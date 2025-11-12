from __future__ import annotations

"""Utilities for registering coding bots with the central registries.

Note:
    Always decorate new coding bot classes with ``@self_coding_managed`` so
    they are automatically registered with the system's helpers.  The decorator
    accepts a ``SelfCodingManager`` instance to reuse existing state across
    instances.

Example:
    >>> @self_coding_managed(
    ...     bot_registry=registry,
    ...     data_bot=data_bot,
    ...     manager=manager,
    ... )
    ... class ExampleBot:
    ...     ...
"""

import contextvars
import importlib.util
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from functools import wraps
import inspect
import json
import logging
import os
import hashlib
import re
import subprocess
import threading
from typing import Iterable, Mapping
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Literal, TypeVar, TYPE_CHECKING
import time

try:  # pragma: no cover - prefer package-relative import
    from menace_sandbox.shared.self_coding_import_guard import (
        self_coding_import_guard,
        self_coding_import_depth,
    )
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    from shared.self_coding_import_guard import (  # type: ignore
        self_coding_import_guard,
        self_coding_import_depth,
    )

try:  # pragma: no cover - prefer package-relative import
    from menace_sandbox.shared.cooperative_init import (
        COOPERATIVE_INIT_KWARGS,
        cooperative_init_call,
    )
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    from shared.cooperative_init import (  # type: ignore
        COOPERATIVE_INIT_KWARGS,
        cooperative_init_call,
    )

try:  # pragma: no cover - prefer package import when available
    from menace_sandbox.self_coding_dependency_probe import ensure_self_coding_ready
except Exception:  # pragma: no cover - fallback when executed from flat layout
    try:
        from self_coding_dependency_probe import ensure_self_coding_ready  # type: ignore
    except Exception:  # pragma: no cover - dependency probe unavailable
        ensure_self_coding_ready = None  # type: ignore[assignment]

try:  # pragma: no cover - prefer package import when available
    from menace_sandbox.self_coding_policy import get_self_coding_policy
except Exception:  # pragma: no cover - fallback when executed from flat layout
    try:
        from self_coding_policy import get_self_coding_policy  # type: ignore
    except Exception:  # pragma: no cover - allow decorator to proceed with defaults
        def get_self_coding_policy():  # type: ignore[override]
            from types import SimpleNamespace

            return SimpleNamespace(
                allowlist=None,
                denylist=frozenset(),
                is_enabled=lambda _name: True,
            )

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal

from shared.provenance_state import (
    PATCH_HASH_CACHE as _PATCH_HASH_CACHE,
    PATCH_HASH_LOCK as _PATCH_HASH_LOCK,
    UNSIGNED_WARNING_CACHE as _UNSIGNED_WARNING_CACHE,
    UNSIGNED_WARNING_LOCK as _UNSIGNED_WARNING_LOCK,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _BootstrapContext:
    """Thread-scoped context shared during nested helper bootstrap."""

    registry: Any = None
    data_bot: Any = None
    manager: Any = None


_BOOTSTRAP_THREAD_STATE = threading.local()


def _push_bootstrap_context(
    *, registry: Any, data_bot: Any, manager: Any
) -> _BootstrapContext:
    """Push a helper context onto the current thread's stack."""

    stack = getattr(_BOOTSTRAP_THREAD_STATE, "stack", None)
    if stack is None:
        stack = []
        _BOOTSTRAP_THREAD_STATE.stack = stack
    context = _BootstrapContext(registry=registry, data_bot=data_bot, manager=manager)
    stack.append(context)
    return context


def _pop_bootstrap_context(context: _BootstrapContext) -> None:
    """Remove ``context`` from the current thread's stack if present."""

    stack = getattr(_BOOTSTRAP_THREAD_STATE, "stack", None)
    if not stack:
        return
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is context:
            stack.pop(index)
            break
    if not stack:
        try:
            delattr(_BOOTSTRAP_THREAD_STATE, "stack")
        except AttributeError:  # pragma: no cover - race safe cleanup
            pass


def _current_bootstrap_context() -> _BootstrapContext | None:
    """Return the innermost helper context for this thread, if any."""

    stack = getattr(_BOOTSTRAP_THREAD_STATE, "stack", None)
    if not stack:
        return None
    return stack[-1]


def _record_cooperative_init_trace(
    instance: object,
    cls: type,
    dropped: tuple[str, ...],
    original_kwargs: Mapping[str, Any],
) -> None:
    """Persist and log cooperative ``__init__`` keyword drops."""

    if not dropped:
        return
    dropped_map = {key: original_kwargs.get(key) for key in dropped}
    trace = getattr(instance, "_cooperative_init_trace", None)
    if not isinstance(trace, list):
        trace = []
    trace.append(
        {
            "class": cls.__qualname__,
            "dropped": tuple(dropped),
            "values": dropped_map,
        }
    )
    if len(trace) > 8:
        del trace[:-8]
    setattr(instance, "_cooperative_init_trace", trace)
    logger.debug(
        "[init-trace] %s received %s unconsumed kwargs: %s",
        cls.__name__,
        len(dropped),
        ", ".join(sorted(dropped)),
    )


_UNSIGNED_COMMIT_PREFIX = "unsigned:"
_SIGNED_PROVENANCE_WARNING_CACHE: set[tuple[str, str]] = set()
_SIGNED_PROVENANCE_WARNING_LOCK = threading.Lock()
_PATCH_PROVENANCE_SERVICE_SENTINEL = object()
_PATCH_PROVENANCE_SERVICE: Any = _PATCH_PROVENANCE_SERVICE_SENTINEL
_SIGNED_PROVENANCE_CACHE: dict[Path, tuple[int, int, tuple["_SignedProvenanceEntry", ...]]] = {}
_PATH_KEY_HINTS: tuple[str, ...] = ("path", "file", "module", "target", "artifact", "source")
_SIGNED_PROVENANCE_CACHE_LOCK = threading.Lock()
_PATCH_HASH_TRACE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "PATCH_HASH_TRACE",
    default=None,
)


def _emit_patch_hash_once(commit: str, *, include_search_hint: bool = False) -> None:
    """Print the derived patch hash only once per ``commit`` value."""

    with _PATCH_HASH_LOCK:
        if commit in _PATCH_HASH_CACHE:
            return
        _PATCH_HASH_CACHE.add(commit)

    print("🐍 PATCH HASH:", commit)
    if include_search_hint:
        print("🧬 Patch being searched:", commit)


@dataclass(slots=True)
class _ProvenanceDecision:
    """Describe the provenance metadata resolved for a coding bot."""

    patch_id: int | None
    commit: str | None
    mode: Literal["signed", "unsigned", "missing"]
    source: str | None = None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.mode != "missing"


@dataclass(slots=True)
class _BootstrapDecision:
    """Capture bootstrap decisions derived from provenance metadata."""

    provenance: _ProvenanceDecision | None = None
    manager: Any = None
    _update_kwargs: Mapping[str, Any] | None = None
    should_update: bool = True
    register_as_coding: bool = True
    hot_swap_active: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - trivial normalisation
        self._update_kwargs = dict(self._update_kwargs or {})

    @property
    def update_kwargs(self) -> dict[str, Any]:
        """Return a copy of the keyword arguments for registry updates."""

        return dict(self._update_kwargs or {})

    def apply(self, *, target: type | None = None) -> None:
        """Materialise provenance details on *target* for diagnostics."""

        if target is None:
            return

        decision = self.provenance
        mode = decision.mode if decision else "missing"
        setattr(target, "_self_coding_provenance", decision)
        setattr(target, "_self_coding_provenance_mode", mode)
        setattr(target, "_self_coding_provenance_source", getattr(decision, "source", None))
        setattr(target, "_self_coding_provenance_reason", getattr(decision, "reason", None))
        setattr(target, "_self_coding_patch_id", getattr(decision, "patch_id", None))
        setattr(target, "_self_coding_commit_hash", getattr(decision, "commit", None))
        setattr(target, "_self_coding_provenance_available", bool(decision and decision.available))
        setattr(target, "_self_coding_provenance_unsigned", bool(decision and decision.mode == "unsigned"))
        if decision and decision.commit:
            _emit_patch_hash_once(decision.commit)


def _registry_hot_swap_active(registry: Any) -> bool:
    """Best-effort helper to query ``registry`` hot swap state."""

    if registry is None:
        return False
    try:
        candidate = getattr(registry, "hot_swap_active", None)
    except Exception:  # pragma: no cover - defensive guard
        return False
    try:
        if callable(candidate):
            return bool(candidate())
        if candidate is not None:
            return bool(candidate)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("failed to evaluate hot_swap_active on %s", registry, exc_info=True)
        return False
    return False


@dataclass(frozen=True)
class _SignedProvenanceEntry:
    """Structured representation of signed provenance metadata."""

    patch_id: int | None
    commit: str | None
    paths: tuple[str, ...] = ()
    path_index: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        object.__setattr__(
            self,
            "path_index",
            frozenset(path.casefold() for path in self.paths if path),
        )

    def matches_path(self, candidate: str | None) -> bool:
        """Return ``True`` when *candidate* matches any recorded path."""

        if not candidate:
            return False
        if candidate in self.paths:
            return True
        return candidate.casefold() in self.path_index


@dataclass(frozen=True)
class _RepositoryContext:
    """Captured repository metadata for provenance resolution."""

    patch_id: int | None
    commit: str | None
    repo_root: Path | None
    relative_path: Path | None
    canonical_path: str | None


def _unsigned_provenance_allowed() -> bool:
    """Return ``True`` when the runtime permits unsigned provenance fallbacks."""

    override = os.getenv("MENACE_ALLOW_UNSIGNED_PROVENANCE")
    if override and override.strip().lower() in {"1", "true", "yes", "on"}:
        return True

    strict = os.getenv("MENACE_REQUIRE_SIGNED_PROVENANCE")
    if strict and strict.strip().lower() in {"1", "true", "yes", "on"}:
        return False

    prov_file = os.getenv("PATCH_PROVENANCE_FILE")
    pubkey = os.getenv("PATCH_PROVENANCE_PUBKEY") or os.getenv(
        "PATCH_PROVENANCE_PUBLIC_KEY"
    )
    if not (prov_file and pubkey):
        return True

    if _signed_provenance_available(prov_file, pubkey):
        return False

    return True


def _signed_provenance_available(prov_file: str, pubkey: str) -> bool:
    """Validate signed provenance configuration returning ``True`` when usable."""

    prov_path = Path(prov_file).expanduser()
    try:
        if not prov_path.exists():
            _warn_signed_provenance_misconfigured(
                prov_file,
                pubkey,
                "file does not exist",
            )
            return False
        if not prov_path.is_file():
            _warn_signed_provenance_misconfigured(
                prov_file,
                pubkey,
                "path is not a regular file",
            )
            return False
        with prov_path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        _warn_signed_provenance_misconfigured(
            prov_file,
            pubkey,
            f"file is not readable ({exc})",
        )
        return False

    pubkey_value = (pubkey or "").strip()
    if not pubkey_value:
        _warn_signed_provenance_misconfigured(
            prov_file,
            pubkey,
            "public key environment variable is empty",
        )
        return False

    if _looks_like_path(pubkey_value):
        key_path = Path(pubkey_value).expanduser()
        try:
            if not key_path.exists():
                _warn_signed_provenance_misconfigured(
                    prov_file,
                    pubkey,
                    "public key path does not exist",
                )
                return False
            if not key_path.is_file():
                _warn_signed_provenance_misconfigured(
                    prov_file,
                    pubkey,
                    "public key path is not a file",
                )
                return False
            with key_path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            _warn_signed_provenance_misconfigured(
                prov_file,
                pubkey,
                f"public key file is not readable ({exc})",
            )
            return False

    return True


def _iter_provenance_dicts(root: Any) -> Iterable[dict[str, Any]]:
    """Yield dictionaries found within ``root`` using an explicit stack."""

    stack: list[Any] = [root]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _canonicalise_repo_path(value: str | os.PathLike[str] | None) -> str | None:
    """Return a repository-relative representation for *value* when possible."""

    if value is None:
        return None
    try:
        text = os.fspath(value)
    except TypeError:
        text = str(value)
    text = text.strip().strip("\"'")
    if not text:
        return None

    if "\\" not in text and "/" not in text and "." in text:
        module_parts = [part for part in text.split(".") if part]
        if module_parts:
            module_path = "/".join(module_parts)
            if not module_path.endswith(".py"):
                module_path = f"{module_path}.py"
            text = module_path

    parts = _split_path_parts(text)
    if not parts:
        return None

    normalised: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned or cleaned == ".":
            continue
        if cleaned == "..":
            if normalised:
                normalised.pop()
            continue
        normalised.append(cleaned)

    if not normalised:
        return None

    return "/".join(normalised)


def _extract_candidate_paths(entry: dict[str, Any]) -> tuple[str, ...]:
    """Derive repository relative path candidates from ``entry`` metadata."""

    seen: set[str] = set()
    stack: list[tuple[Any, str | None]] = [(entry, None)]
    while stack:
        current, hint = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                key_hint = key.lower() if isinstance(key, str) else None
                stack.append((value, key_hint))
        elif isinstance(current, (list, tuple, set)):
            for item in current:
                stack.append((item, hint))
        else:
            if hint is None or not any(token in hint for token in _PATH_KEY_HINTS):
                continue
            try:
                candidate = os.fspath(current)
            except TypeError:
                if isinstance(current, str):
                    candidate = current
                else:
                    continue
            canonical = _canonicalise_repo_path(candidate)
            if canonical:
                seen.add(canonical)

    return tuple(sorted(seen))


def _extract_signed_candidate(entry: dict[str, Any]) -> _SignedProvenanceEntry:
    """Extract structured provenance details from ``entry``."""

    patch_id = _coerce_patch_id(
        entry.get("patch_id")
        or entry.get("patch")
        or entry.get("id")
        or entry.get("vector_id")
    )
    commit = _coerce_commit(
        entry.get("commit")
        or entry.get("commit_hash")
        or entry.get("hash")
        or entry.get("commitId")
    )

    nested = entry.get("patch") or entry.get("metadata") or entry.get("data")
    if isinstance(nested, dict):
        patch_id = patch_id or _coerce_patch_id(
            nested.get("patch_id")
            or nested.get("patch")
            or nested.get("id")
        )
        commit = commit or _coerce_commit(
            nested.get("commit")
            or nested.get("commit_hash")
            or nested.get("hash")
        )

    paths = _extract_candidate_paths(entry)

    return _SignedProvenanceEntry(patch_id, commit, paths)


def _load_signed_provenance_candidates() -> tuple[_SignedProvenanceEntry, ...]:
    """Return cached provenance entries derived from the signed metadata file."""

    prov_file = os.getenv("PATCH_PROVENANCE_FILE")
    if not prov_file:
        return tuple()

    path = Path(prov_file).expanduser()
    try:
        stat_result = path.stat()
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.debug("failed to stat signed provenance file %s: %s", path, exc, exc_info=True)
        return tuple()

    mtime_ns = getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))
    size = getattr(stat_result, "st_size", 0)

    with _SIGNED_PROVENANCE_CACHE_LOCK:
        cached = _SIGNED_PROVENANCE_CACHE.get(path)
        if cached and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]

    payload: Any
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "failed to load signed provenance metadata from %s: %s",
            path,
            exc,
        )
        return tuple()

    seen_keys: set[tuple[int | None, str | None, tuple[str, ...]]] = set()
    candidates: list[_SignedProvenanceEntry] = []
    provenance_keys: tuple[str, ...] = ()
    if isinstance(payload, dict):
        provenance_keys = tuple(str(key) for key in payload.keys())
    elif isinstance(payload, list):
        key_accumulator: set[str] = set()
        for item in payload:
            if isinstance(item, dict):
                key_accumulator.update(str(key) for key in item.keys())
        if key_accumulator:
            provenance_keys = tuple(sorted(key_accumulator))

    for entry in _iter_provenance_dicts(payload):
        candidate = _extract_signed_candidate(entry)
        if candidate.patch_id is None and candidate.commit is None:
            continue
        key = (candidate.patch_id, candidate.commit, candidate.paths)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(candidate)

    result = tuple(candidates)
    with _SIGNED_PROVENANCE_CACHE_LOCK:
        _SIGNED_PROVENANCE_CACHE[path] = (mtime_ns, size, result)

    if not result:
        patch_hash = _PATCH_HASH_TRACE.get()
        if patch_hash:
            _emit_patch_hash_once(patch_hash, include_search_hint=True)
        print("🔎 Available provenance keys:", provenance_keys)
        logger.debug(
            "signed provenance file %s did not yield usable patch metadata", path
        )

    return result


def _commit_matches(reference: str | None, candidate: str | None) -> bool:
    """Return ``True`` when two commit identifiers likely refer to the same commit."""

    if not reference or not candidate:
        return False

    ref = reference.strip().lower()
    cand = candidate.strip().lower()
    if not ref or not cand:
        return False

    if ref == cand:
        return True

    # Allow matching shortened commit hashes (e.g. 7-12 chars) against the full
    # 40 character value.  Git consistently treats prefixes of length >= 7 as
    # unambiguous identifiers, so we mirror that heuristic to honour signed
    # provenance files that only embed abbreviated commits.
    min_length = 7
    if len(ref) >= len(cand) and len(cand) >= min_length and ref.startswith(cand):
        return True
    if len(cand) > len(ref) and len(ref) >= min_length and cand.startswith(ref):
        return True

    return False


def _looks_like_path(value: str) -> bool:
    """Heuristically determine whether *value* represents a filesystem path."""

    if not value:
        return False

    candidate = value.strip()
    if candidate.startswith("-----BEGIN") or candidate.startswith("ssh-"):
        return False
    if candidate.startswith("~"):
        return True
    if os.path.isabs(candidate):
        return True
    if len(candidate) > 2 and candidate[1] == ":" and candidate[2:3] in {"\\", "/"}:
        return True
    if any(sep and sep in candidate for sep in (os.sep, os.path.altsep)):
        return True
    suffix = Path(candidate).suffix.lower()
    if suffix in {".pem", ".pub", ".key", ".der", ".json"}:
        return True
    return False


def _warn_signed_provenance_misconfigured(
    prov_file: str, pubkey: str, reason: str
) -> None:
    """Log a single warning for unusable signed provenance settings."""

    key = (prov_file or "", pubkey or "")
    with _SIGNED_PROVENANCE_WARNING_LOCK:
        if key in _SIGNED_PROVENANCE_WARNING_CACHE:
            return
        _SIGNED_PROVENANCE_WARNING_CACHE.add(key)

    logger.warning(
        "Signed provenance requested via PATCH_PROVENANCE_FILE=%s but %s; "
        "falling back to unsigned provenance.",
        prov_file,
        reason,
    )


def _warn_unsigned_once(name: str) -> None:
    """Log a single warning per *name* when unsigned provenance is generated."""

    with _UNSIGNED_WARNING_LOCK:
        if name in _UNSIGNED_WARNING_CACHE:
            return
        _UNSIGNED_WARNING_CACHE.add(name)
    logger.warning(
        "Provenance metadata unavailable; generating unsigned fingerprint for %s. "
        "Set PATCH_PROVENANCE_FILE and PATCH_PROVENANCE_PUBKEY to enable signed provenance.",
        name,
    )


def _reset_patch_provenance_service() -> None:
    """Reset the cached :class:`PatchProvenanceService` instance."""

    global _PATCH_PROVENANCE_SERVICE
    _PATCH_PROVENANCE_SERVICE = _PATCH_PROVENANCE_SERVICE_SENTINEL


def _get_patch_provenance_service() -> Any | None:
    """Return a cached ``PatchProvenanceService`` instance when available."""

    global _PATCH_PROVENANCE_SERVICE
    if _PATCH_PROVENANCE_SERVICE is _PATCH_PROVENANCE_SERVICE_SENTINEL:
        try:
            module = load_internal("patch_provenance")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency missing
            logger.debug(
                "patch_provenance unavailable while resolving provenance metadata", exc_info=exc
            )
            _PATCH_PROVENANCE_SERVICE = None
        except Exception as exc:  # pragma: no cover - defensive best effort
            logger.warning(
                "failed to import patch_provenance while resolving provenance metadata: %s",
                exc,
                exc_info=True,
            )
            _PATCH_PROVENANCE_SERVICE = None
        else:
            try:
                _PATCH_PROVENANCE_SERVICE = module.PatchProvenanceService()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - service initialisation failure
                logger.warning(
                    "failed to initialise PatchProvenanceService while resolving provenance metadata: %s",
                    exc,
                    exc_info=True,
                )
                _PATCH_PROVENANCE_SERVICE = None
    return _PATCH_PROVENANCE_SERVICE


def _derive_unsigned_provenance(name: str, module_path: str | None) -> tuple[int, str]:
    """Return a deterministic ``(patch_id, commit)`` pair for unsigned updates."""

    identifier = module_path or name
    path = Path(module_path) if module_path else None
    hasher = hashlib.sha256()

    if path and path.exists():
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(131072), b""):
                    if not chunk:
                        break
                    hasher.update(chunk)
        except OSError as exc:  # pragma: no cover - filesystem specific
            logger.warning(
                "Failed to read %s while deriving unsigned provenance: %s", identifier, exc
            )
            seed = f"{identifier}:{time.time_ns()}".encode("utf-8", "replace")
            hasher.update(seed)
    else:
        fallback = identifier.encode("utf-8", "replace")
        hasher.update(fallback)

    digest = hasher.hexdigest()
    seed_bytes = hashlib.blake2s(identifier.encode("utf-8", "replace"), digest_size=4).digest()
    seed_value = int.from_bytes(seed_bytes, "big") or 1
    patch_id = -abs(seed_value)
    commit = f"{_UNSIGNED_COMMIT_PREFIX}{digest}"
    _PATCH_HASH_TRACE.set(commit)
    _emit_patch_hash_once(commit)
    return patch_id, commit


def _split_path_parts(raw_path: str) -> tuple[str, ...]:
    """Return normalised path components for ``raw_path`` across platforms."""

    if not raw_path:
        return ()

    if "\\" in raw_path or re.match(r"^[A-Za-z]:", raw_path):
        pure = PureWindowsPath(raw_path)
        drive = pure.drive
        return tuple(part for part in pure.parts if part and part not in {drive, f"{drive}\\"})

    pure = PurePosixPath(raw_path)
    return pure.parts


def _normalise_module_path(module_path: str | os.PathLike[str] | None) -> Path | None:
    """Attempt to resolve ``module_path`` to an existing :class:`Path` instance."""

    if module_path is None:
        return None

    try:
        raw_path = os.fspath(module_path)
    except TypeError:
        return None

    raw_path = raw_path.strip()
    if not raw_path:
        return None

    candidates: list[Path] = []
    try:
        candidates.append(Path(raw_path))
    except (TypeError, ValueError, OSError):  # pragma: no cover - invalid representation
        pass

    alt = raw_path.replace("\\", os.sep)
    if alt != raw_path:
        try:
            candidates.append(Path(alt))
        except (TypeError, ValueError, OSError):
            pass

    repo_hints: list[str] = []
    for var in ("SANDBOX_REPO_PATH", "MENACE_ROOT"):
        value = os.getenv(var)
        if value:
            repo_hints.append(value)
    for var in ("SANDBOX_REPO_PATHS", "MENACE_ROOTS"):
        value = os.getenv(var)
        if value:
            repo_hints.extend([item for item in value.split(os.pathsep) if item])

    parts = _split_path_parts(raw_path)
    for hint in repo_hints:
        try:
            base = Path(hint)
        except (TypeError, ValueError, OSError):
            continue
        for index in range(len(parts)):
            candidate = base.joinpath(*parts[index:])
            candidates.append(candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            expanded = candidate.expanduser()
        except (RuntimeError, ValueError):  # pragma: no cover - expansion failure
            continue
        try:
            canonical = expanded.resolve(strict=False)
        except (RuntimeError, OSError):  # pragma: no cover - resolution failure
            canonical = expanded
        if canonical in seen:
            continue
        seen.add(canonical)
        if canonical.exists():
            return canonical
    return None


def _discover_repo_root_from_fs(search_root: Path) -> Path | None:
    """Walk up the directory tree looking for a ``.git`` entry."""

    for candidate in (search_root, *search_root.parents):
        git_entry = candidate / ".git"
        try:
            if git_entry.exists():
                return candidate
        except OSError:  # pragma: no cover - filesystem dependent
            logger.debug(
                "failed to stat potential git entry %s while discovering repository root",
                git_entry,
                exc_info=True,
            )
            continue
    return None


def _resolve_git_repository(path: Path) -> Path | None:
    """Return the git repository root for ``path`` or ``None`` when unavailable."""

    search_root = path if path.is_dir() else path.parent
    try:
        output = subprocess.check_output(
            ["git", "-C", str(search_root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.debug(
            "failed to resolve git repository for %s using git executable: %s",
            path,
            exc,
            exc_info=True,
        )
        fallback_root = _discover_repo_root_from_fs(search_root)
        if fallback_root is not None:
            logger.debug(
                "resolved git repository for %s via filesystem fallback: %s",
                path,
                fallback_root,
            )
            return fallback_root
        return None
    root = output.decode("utf-8", "replace").strip()
    if not root:
        fallback_root = _discover_repo_root_from_fs(search_root)
        if fallback_root is not None:
            return fallback_root
        return None
    return Path(root)


def _resolve_git_dir_for_repo(repo_root: Path) -> Path | None:
    """Return the git metadata directory for ``repo_root`` when available."""

    git_entry = repo_root / ".git"
    try:
        if git_entry.is_dir():
            return git_entry
        if git_entry.is_file():
            try:
                content = git_entry.read_text(encoding="utf-8")
            except OSError as exc:  # pragma: no cover - filesystem dependent
                logger.debug(
                    "failed to read indirection file %s while resolving git dir: %s",
                    git_entry,
                    exc,
                    exc_info=True,
                )
                return None
            match = re.search(r"gitdir:\s*(.+)", content)
            if match:
                candidate = (git_entry.parent / match.group(1)).expanduser()
                try:
                    resolved = candidate.resolve(strict=False)
                except (RuntimeError, OSError):  # pragma: no cover - resolution failure
                    resolved = candidate
                if resolved.exists():
                    return resolved
    except OSError:  # pragma: no cover - filesystem dependent
        logger.debug(
            "failed to inspect git metadata entry for %s",
            repo_root,
            exc_info=True,
        )
    return None


def _read_git_ref(git_dir: Path, ref: str) -> str | None:
    """Read ``ref`` from ``git_dir`` returning the stored commit hash."""

    ref_path = git_dir.joinpath(*PurePosixPath(ref).parts)
    try:
        data = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return data or None


def _read_git_packed_ref(git_dir: Path, ref: str) -> str | None:
    """Read ``ref`` from ``packed-refs`` when the loose ref is absent."""

    packed = git_dir / "packed-refs"
    try:
        with packed.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith(("#", "^")):
                    continue
                try:
                    commit_hash, ref_name = line.split(" ", 1)
                except ValueError:
                    continue
                if ref_name == ref:
                    return commit_hash
    except OSError:  # pragma: no cover - filesystem dependent
        return None
    return None


def _read_git_reflog_commit(git_dir: Path, ref: str) -> str | None:
    """Return the newest commit recorded in the reflog for ``ref``."""

    log_path = git_dir / "logs"
    log_path = log_path.joinpath(*PurePosixPath(ref).parts)
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            last_line = None
            for line in handle:
                if line.strip():
                    last_line = line
    except OSError:
        return None
    if not last_line:
        return None
    parts = last_line.strip().split()
    if len(parts) >= 2:
        return parts[1]
    return None


def _read_git_head_commit(repo_root: Path) -> str | None:
    """Best-effort resolution of ``HEAD`` commit without invoking ``git``."""

    git_dir = _resolve_git_dir_for_repo(repo_root)
    if git_dir is None:
        return None

    head_path = git_dir / "HEAD"
    try:
        head_data = head_path.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.debug(
            "failed to read HEAD file from %s while deriving commit fallback: %s",
            head_path,
            exc,
            exc_info=True,
        )
        return None

    if not head_data:
        return None

    if head_data.startswith("ref:"):
        ref = head_data.partition(":")[2].strip()
        commit = _read_git_ref(git_dir, ref) or _read_git_packed_ref(git_dir, ref)
        if commit:
            return _coerce_commit(commit)
        commit = _read_git_reflog_commit(git_dir, ref)
        if commit:
            return _coerce_commit(commit)
        return None

    return _coerce_commit(head_data)


def _relative_repo_path(path: Path, repo_root: Path) -> Path:
    """Return ``path`` relative to ``repo_root`` without filesystem access."""

    try:
        return path.resolve().relative_to(repo_root.resolve())
    except Exception:  # pragma: no cover - fall back to os.path.relpath semantics
        rel = os.path.relpath(str(path), str(repo_root))
        return Path(rel)


def _lookup_patch_by_commit(commit: str, *, log_hint: str) -> tuple[int | None, str | None]:
    """Return ``(patch_id, commit)`` for ``commit`` when recorded."""

    service = _get_patch_provenance_service()
    if service is None:
        return (None, None)

    try:
        record = service.get(commit)
    except Exception as exc:  # pragma: no cover - database access failure
        logger.debug(
            "patch provenance lookup failed for %s (commit=%s): %s", log_hint, commit, exc, exc_info=True
        )
        return (None, None)

    if not record:
        return (None, None)

    patch_id = _coerce_patch_id(record.get("patch_id") or record.get("id"))
    commit_hash = _coerce_commit(record.get("commit") or commit)
    return patch_id, commit_hash


def _resolve_repository_context(
    name: str, module_path: str | os.PathLike[str] | None
) -> _RepositoryContext:
    """Collect repository provenance hints for ``module_path``."""

    path = _normalise_module_path(module_path)
    if path is None:
        return _RepositoryContext(None, None, None, None, None)

    repo_root = _resolve_git_repository(path)
    if repo_root is None:
        return _RepositoryContext(None, None, None, None, None)

    rel_path = _relative_repo_path(path, repo_root)
    rel_path_posix = rel_path.as_posix()
    canonical_path = _canonicalise_repo_path(rel_path_posix)

    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "-n",
                "1",
                "--pretty=format:%H",
                "--",
                rel_path_posix,
            ],
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.debug(
            "failed to derive git commit for %s from %s (repo=%s): %s",
            name,
            path,
            repo_root,
            exc,
            exc_info=True,
        )
        commit = _read_git_head_commit(repo_root)
        if not commit:
            return _RepositoryContext(None, None, repo_root, rel_path, canonical_path)
        logger.debug(
            "using HEAD commit fallback for %s because git log invocation failed", name
        )
    else:
        commit = _coerce_commit(output.decode("utf-8", "replace").strip())
        if not commit:
            return _RepositoryContext(None, None, repo_root, rel_path, canonical_path)

    patch_id, commit_hash = _lookup_patch_by_commit(
        commit, log_hint=f"{name}@{rel_path_posix}"
    )
    if patch_id is None:
        return _RepositoryContext(None, commit, repo_root, rel_path, canonical_path)

    return _RepositoryContext(patch_id, commit_hash or commit, repo_root, rel_path, canonical_path)


def _derive_repository_provenance(
    name: str, module_path: str | os.PathLike[str] | None
) -> tuple[int | None, str | None]:
    """Attempt to recover signed provenance by inspecting the git repository."""

    ctx = _resolve_repository_context(name, module_path)
    return ctx.patch_id, ctx.commit


def _coerce_patch_id(value: object) -> int | None:
    """Best-effort conversion of *value* into an integer patch identifier."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive
        return None
    if not text:
        return None
    try:
        return int(text, 10)
    except (TypeError, ValueError):  # pragma: no cover - invalid representation
        return None


def _coerce_commit(value: object) -> str | None:
    """Return a normalised commit hash representation or ``None``."""

    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive
        return None
    return text or None


def _extract_module_provenance(module: ModuleType | None) -> tuple[int | None, str | None]:
    """Return module-level provenance metadata when available."""

    if module is None:
        return (None, None)

    candidates: list[tuple[int | None, str | None]] = []
    for attr in ("__menace_provenance__", "__provenance__", "MENACE_PROVENANCE"):
        data = getattr(module, attr, None)
        if isinstance(data, dict):
            patch_id = _coerce_patch_id(
                data.get("patch_id") or data.get("patch") or data.get("id")
            )
            commit = _coerce_commit(
                data.get("commit")
                or data.get("commit_hash")
                or data.get("hash")
            )
            candidates.append((patch_id, commit))
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            candidates.append((_coerce_patch_id(data[0]), _coerce_commit(data[1])))
    explicit_patch = _coerce_patch_id(getattr(module, "__patch_id__", None))
    explicit_commit = _coerce_commit(
        getattr(module, "__commit__", None) or getattr(module, "__commit_hash__", None)
    )
    candidates.append((explicit_patch, explicit_commit))
    for patch_id, commit in candidates:
        if patch_id is not None and commit is not None:
            return patch_id, commit
    return (None, None)


def _backfill_commit_from_history(patch_id: int, *, log_hint: str) -> str | None:
    """Attempt to recover a commit hash for ``patch_id`` from stored metadata."""

    try:  # pragma: no cover - optional dependency
        from .patch_provenance import PatchProvenanceService
    except Exception:  # pragma: no cover - best effort recovery
        logger.debug(
            "failed to import PatchProvenanceService while backfilling provenance for %s",
            log_hint,
            exc_info=True,
        )
        return None

    try:
        service = PatchProvenanceService()
        record = service.db.get(patch_id)
    except Exception:  # pragma: no cover - best effort recovery
        logger.debug(
            "failed to load provenance record for patch %s while backfilling %s",
            patch_id,
            log_hint,
            exc_info=True,
        )
        return None

    summary = getattr(record, "summary", None)
    if not summary:
        return None
    try:
        data = json.loads(summary)
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "failed to parse provenance summary for patch %s while backfilling %s",
            patch_id,
            log_hint,
            exc_info=True,
        )
        return None
    commit = _coerce_commit(data.get("commit"))
    return commit


def _resolve_provenance_decision(
    name: str,
    module_path: str | None,
    manager_sources: list[object],
    module_provenance: tuple[int | None, str | None],
) -> _ProvenanceDecision:
    """Resolve provenance metadata for a coding bot registration."""

    patch_id: int | None = None
    commit: str | None = None

    for candidate in manager_sources:
        if candidate is None:
            continue
        patch_id = patch_id or _coerce_patch_id(getattr(candidate, "_last_patch_id", None))
        commit = commit or _coerce_commit(getattr(candidate, "_last_commit_hash", None))
        if patch_id is not None and commit is not None:
            return _ProvenanceDecision(patch_id, commit, "signed", source="manager")

    if patch_id is not None and commit is None:
        commit = _backfill_commit_from_history(patch_id, log_hint=name)
        if commit is not None:
            return _ProvenanceDecision(patch_id, commit, "signed", source="manager")

    module_patch, module_commit = module_provenance
    signed_entries = _load_signed_provenance_candidates()
    allow_unsigned = _unsigned_provenance_allowed()
    if not signed_entries and not allow_unsigned:
        prov_file = os.getenv("PATCH_PROVENANCE_FILE", "")
        pubkey = os.getenv("PATCH_PROVENANCE_PUBKEY") or os.getenv(
            "PATCH_PROVENANCE_PUBLIC_KEY",
            "",
        )
        _warn_signed_provenance_misconfigured(
            prov_file,
            pubkey or "",
            "signed provenance file does not contain usable entries",
        )
        allow_unsigned = True
    if module_patch is not None:
        commit_candidate = module_commit or _backfill_commit_from_history(
            module_patch, log_hint=name
        )
        if commit_candidate is not None:
            return _ProvenanceDecision(
                module_patch, commit_candidate, "signed", source="module"
            )

    if module_patch is None and module_commit is not None:
        patch_candidate, commit_candidate = _lookup_patch_by_commit(
            module_commit, log_hint=f"{name}:module"
        )
        if (
            patch_candidate is None
            and commit_candidate is not None
            and signed_entries
        ):
            for entry in signed_entries:
                cand_patch, cand_commit = entry.patch_id, entry.commit
                if cand_patch is not None and _commit_matches(
                    commit_candidate, cand_commit
                ):
                    commit_value = (
                        commit_candidate
                        if len(commit_candidate or "") >= len(cand_commit or "")
                        else cand_commit
                    )
                    return _ProvenanceDecision(
                        cand_patch, commit_value, "signed", source="signed"
                    )
        if patch_candidate is not None and commit_candidate is not None:
            return _ProvenanceDecision(
                patch_candidate, commit_candidate, "signed", source="module"
            )

    repo_ctx = _resolve_repository_context(name, module_path)
    repo_patch = repo_ctx.patch_id
    repo_commit = repo_ctx.commit
    repo_path = repo_ctx.canonical_path
    if repo_patch is not None and repo_commit is not None:
        return _ProvenanceDecision(repo_patch, repo_commit, "signed", source="repository")

    if repo_path and signed_entries:
        for entry in signed_entries:
            if entry.patch_id is None:
                continue
            if entry.matches_path(repo_path):
                base_commit = entry.commit or repo_commit
                if base_commit is None:
                    continue
                if entry.commit and repo_commit:
                    commit_value = (
                        repo_commit
                        if len(repo_commit) >= len(entry.commit or "")
                        else entry.commit
                    )
                else:
                    commit_value = base_commit
                logger.debug(
                    "resolved signed provenance for %s via repository path match (%s)",
                    name,
                    repo_path,
                )
                return _ProvenanceDecision(
                    entry.patch_id,
                    commit_value,
                    "signed",
                    source="signed:path",
                )

    if repo_commit is not None and signed_entries:
        for entry in signed_entries:
            cand_patch, cand_commit = entry.patch_id, entry.commit
            if cand_patch is not None and _commit_matches(repo_commit, cand_commit):
                commit_value = (
                    repo_commit
                    if len(repo_commit) >= len(cand_commit or "")
                    else cand_commit
                )
                return _ProvenanceDecision(
                    cand_patch, commit_value, "signed", source="signed"
                )

    if module_commit is not None and signed_entries:
        for entry in signed_entries:
            cand_patch, cand_commit = entry.patch_id, entry.commit
            if cand_patch is not None and _commit_matches(module_commit, cand_commit):
                commit_value = (
                    module_commit
                    if len(module_commit) >= len(cand_commit or "")
                    else cand_commit
                )
                return _ProvenanceDecision(
                    cand_patch, commit_value, "signed", source="signed"
                )

    if signed_entries:
        for entry in signed_entries:
            if entry.patch_id is not None and entry.commit is not None:
                return _ProvenanceDecision(
                    entry.patch_id, entry.commit, "signed", source="signed"
                )

    if allow_unsigned:
        unsigned_patch, unsigned_commit = _derive_unsigned_provenance(name, module_path)
        return _ProvenanceDecision(
            unsigned_patch, unsigned_commit, "unsigned", source="unsigned"
        )

    reason = (
        "strict provenance policy requires signed metadata"
        if not allow_unsigned
        else "provenance metadata unavailable"
    )
    return _ProvenanceDecision(None, None, "missing", source=None, reason=reason)

create_context_builder = load_internal("context_builder_util").create_context_builder


class _ThresholdModuleFallback:
    """Gracefully degrade when ``self_coding_thresholds`` is unavailable.

    The Windows command prompt environments that ship with the autonomous
    sandbox frequently omit scientific Python dependencies such as
    :mod:`pydantic`.  Importing :mod:`self_coding_thresholds` in that situation
    raises a :class:`ModuleNotFoundError` which used to abort the import of this
    module entirely, cascading into repeated internalisation retries for coding
    bots.  The fallback implementation below keeps the decorator operational
    while clearly surfacing the degraded behaviour via structured logging.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self._update_logged = False
        self._load_logged = False

    def update_thresholds(self, name: str, *args: Any, **kwargs: Any) -> None:
        if not self._update_logged:
            logger.warning(
                "self_coding_thresholds unavailable; skipping threshold updates (%s)",
                self.reason,
            )
            self._update_logged = True
        else:
            logger.debug(
                "suppressed threshold update for %s due to missing dependencies", name
            )

    def load_config(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not self._load_logged:
            logger.info(
                "returning empty self-coding threshold config because dependencies are missing (%s)",
                self.reason,
            )
            self._load_logged = True
        return {}


try:
    _self_coding_thresholds = load_internal("self_coding_thresholds")
except ModuleNotFoundError as exc:
    fallback = _ThresholdModuleFallback(f"module not found: {exc}")
    update_thresholds = fallback.update_thresholds

    def _load_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return fallback.load_config(*args, **kwargs)

except Exception as exc:  # pragma: no cover - defensive degradation
    fallback = _ThresholdModuleFallback(f"import failure: {exc}")
    update_thresholds = fallback.update_thresholds

    def _load_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return fallback.load_config(*args, **kwargs)

else:
    update_thresholds = _self_coding_thresholds.update_thresholds
    _load_config = _self_coding_thresholds._load_config

_SELF_CODING_MANAGER_SENTINEL = object()
_SELF_CODING_MANAGER_CLS: type[Any] | None | object = _SELF_CODING_MANAGER_SENTINEL


def _import_self_coding_manager_cls() -> type[Any] | None | object:
    """Load ``SelfCodingManager`` lazily to avoid circular imports."""

    try:  # pragma: no cover - optional self-coding dependency
        return load_internal("self_coding_manager").SelfCodingManager
    except ModuleNotFoundError as exc:  # pragma: no cover - degrade gracefully when absent
        fallback_mod = sys.modules.get("menace.self_coding_manager")
        if fallback_mod and hasattr(fallback_mod, "SelfCodingManager"):
            logger.debug(
                "using stub SelfCodingManager from menace.self_coding_manager after import failure",
                exc_info=exc,
            )
            return getattr(fallback_mod, "SelfCodingManager")  # type: ignore[no-any-return]

        logger.warning(
            "self_coding_manager could not be imported; self-coding will run in disabled mode",
            exc_info=exc,
        )
        return None
    except Exception as exc:  # pragma: no cover - degrade gracefully when unavailable
        fallback_mod = sys.modules.get("menace.self_coding_manager")
        if fallback_mod and hasattr(fallback_mod, "SelfCodingManager"):
            logger.debug(
                "using stub SelfCodingManager from menace.self_coding_manager after runtime error",
                exc_info=exc,
            )
            return getattr(fallback_mod, "SelfCodingManager")  # type: ignore[no-any-return]

        logger.warning(
            "self_coding_manager import failed; self-coding will run in disabled mode",
            exc_info=exc,
        )
        return None


def _resolve_self_coding_manager_cls() -> type[Any] | None:
    """Return the cached ``SelfCodingManager`` class when available."""

    global _SELF_CODING_MANAGER_CLS
    if _SELF_CODING_MANAGER_CLS is _SELF_CODING_MANAGER_SENTINEL:
        _SELF_CODING_MANAGER_CLS = _import_self_coding_manager_cls()
    elif _SELF_CODING_MANAGER_CLS is None:
        # A previous attempt failed (for example due to an import cycle).  When
        # the module eventually completes initialisation the attribute becomes
        # available, so re-check ``sys.modules`` for a loaded definition before
        # giving up.
        for mod_name in ("menace_sandbox.self_coding_manager", "self_coding_manager"):
            module = sys.modules.get(mod_name)
            if module and hasattr(module, "SelfCodingManager"):
                _SELF_CODING_MANAGER_CLS = getattr(module, "SelfCodingManager")
                break

    resolved = _SELF_CODING_MANAGER_CLS
    return resolved if isinstance(resolved, type) else None


def _is_self_coding_manager(candidate: Any) -> bool:
    """Return ``True`` when *candidate* is a ``SelfCodingManager`` instance."""

    manager_cls = _resolve_self_coding_manager_cls()
    return isinstance(candidate, manager_cls) if manager_cls is not None else False

_ENGINE_AVAILABLE = True
_ENGINE_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - allow tests to stub engine
    _self_coding_engine = load_internal("self_coding_engine")
except ModuleNotFoundError as exc:  # pragma: no cover - propagate requirement
    fallback_engine = sys.modules.get("menace.self_coding_engine")
    if fallback_engine is not None:
        _self_coding_engine = fallback_engine  # type: ignore[assignment]
        MANAGER_CONTEXT = getattr(
            fallback_engine,
            "MANAGER_CONTEXT",
            contextvars.ContextVar("MANAGER_CONTEXT", default=None),
        )
        _ENGINE_AVAILABLE = True
        _ENGINE_IMPORT_ERROR = None
        logger.debug(
            "using stub self_coding_engine from menace.self_coding_engine after import failure",
            exc_info=exc,
        )
    else:
        _ENGINE_AVAILABLE = False
        _ENGINE_IMPORT_ERROR = exc
        MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT", default=None)
except Exception as exc:  # pragma: no cover - fail fast when engine unavailable
    fallback_engine = sys.modules.get("menace.self_coding_engine")
    if fallback_engine is not None:
        _self_coding_engine = fallback_engine  # type: ignore[assignment]
        MANAGER_CONTEXT = getattr(
            fallback_engine,
            "MANAGER_CONTEXT",
            contextvars.ContextVar("MANAGER_CONTEXT", default=None),
        )
        _ENGINE_AVAILABLE = True
        _ENGINE_IMPORT_ERROR = None
        logger.debug(
            "using stub self_coding_engine from menace.self_coding_engine after runtime error",
            exc_info=exc,
        )
    else:
        _ENGINE_AVAILABLE = False
        _ENGINE_IMPORT_ERROR = exc
        MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT", default=None)
else:
    MANAGER_CONTEXT = getattr(
        _self_coding_engine,
        "MANAGER_CONTEXT",
        contextvars.ContextVar("MANAGER_CONTEXT", default=None),
    )
    if not hasattr(_self_coding_engine, "MANAGER_CONTEXT"):
        _ENGINE_AVAILABLE = False
        _ENGINE_IMPORT_ERROR = AttributeError(
            "self_coding_engine.MANAGER_CONTEXT is not available"
        )

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from menace_sandbox.bot_registry import BotRegistry
    from menace_sandbox.data_bot import DataBot
    from menace_sandbox.evolution_orchestrator import EvolutionOrchestrator
    from menace_sandbox.self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime placeholders
    BotRegistry = Any  # type: ignore
    DataBot = Any  # type: ignore
    EvolutionOrchestrator = Any  # type: ignore
    SelfCodingManager = Any  # type: ignore


if not _ENGINE_AVAILABLE and _ENGINE_IMPORT_ERROR is not None:
    logger.warning(
        "self_coding_engine could not be imported; coding bot helpers are in limited mode",
        exc_info=_ENGINE_IMPORT_ERROR,
    )


def _self_coding_runtime_available() -> bool:
    return _ENGINE_AVAILABLE and _resolve_self_coding_manager_cls() is not None


class _DisabledSelfCodingManager:
    """Fallback manager used when the runtime cannot support self-coding."""

    __slots__ = (
        "bot_registry",
        "data_bot",
        "engine",
        "quick_fix",
        "error_db",
        "evolution_orchestrator",
        "_last_patch_id",
        "_last_commit_hash",
    )

    def __init__(self, *, bot_registry: Any, data_bot: Any) -> None:
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.engine = SimpleNamespace(
            cognition_layer=SimpleNamespace(context_builder=None)
        )
        # Mark quick_fix as initialised so downstream code skips heavy bootstrap.
        self.quick_fix = object()
        self.error_db = None
        self.evolution_orchestrator = None
        self._last_patch_id = None
        self._last_commit_hash = None

    def __bool__(self) -> bool:
        """Report ``False`` so helper heuristics treat the manager as disabled."""

        return False

    def register_patch_cycle(self, *_args: Any, **_kwargs: Any) -> None:
        logger.debug(
            "self-coding disabled; ignoring register_patch_cycle invocation"
        )

    def register(self, *_args: Any, **_kwargs: Any) -> None:
        logger.debug("self-coding disabled; ignoring register invocation")

    def register_bot(self, *_args: Any, **_kwargs: Any) -> None:
        logger.debug(
            "self-coding disabled; ignoring register_bot invocation"
        )

    def run_post_patch_cycle(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        logger.debug(
            "self-coding disabled; ignoring run_post_patch_cycle invocation"
        )
        return {}

    def refresh_quick_fix_context(self) -> Any:
        return getattr(self.engine, "context_builder", None)


_BOOTSTRAP_STATE = threading.local()


def _bootstrap_manager(
    name: str,
    bot_registry: BotRegistry,
    data_bot: DataBot,
) -> Any:
    """Instantiate a ``SelfCodingManager`` with progressive fallbacks."""

    def _disabled_manager(reason: str) -> Any:
        guidance = (
            "Re-entrant bootstrap detected; returning disabled manager "
            "temporarily—internalisation will retry after "
            f"{name} completes."
        )
        message = (
            f"SelfCodingManager bootstrap skipped for {name}: {reason}. {guidance}"
        )
        print(message)
        logger.warning(message)
        return _DisabledSelfCodingManager(
            bot_registry=bot_registry,
            data_bot=data_bot,
        )

    depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
    if depth > 0:
        return _disabled_manager(f"re-entrant initialisation depth={depth}")

    active = set(getattr(_BOOTSTRAP_STATE, "names", set()))
    if name in active:
        return _disabled_manager("bootstrap already in progress")

    active.add(name)
    _BOOTSTRAP_STATE.names = active
    _BOOTSTRAP_STATE.depth = depth + 1

    def _release_guard() -> None:
        current = set(getattr(_BOOTSTRAP_STATE, "names", set()))
        current.discard(name)
        if current:
            _BOOTSTRAP_STATE.names = current
        elif hasattr(_BOOTSTRAP_STATE, "names"):
            delattr(_BOOTSTRAP_STATE, "names")

    try:
        manager_cls = _resolve_self_coding_manager_cls()
        if manager_cls is None:
            raise RuntimeError("self-coding runtime is unavailable")

        try:
            return manager_cls(  # type: ignore[call-arg]
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
        except TypeError:
            pass

        code_db_cls = _load_optional_module("code_database").CodeDB
        memory_cls = _load_optional_module("gpt_memory").GPTMemoryManager
        engine_mod = _load_optional_module(
            "self_coding_engine", fallback="menace.self_coding_engine"
        )
        pipeline_mod = _load_optional_module(
            "model_automation_pipeline", fallback="menace.model_automation_pipeline"
        )
        ctx_builder = create_context_builder()
        engine = engine_mod.SelfCodingEngine(
            code_db_cls(),
            memory_cls(),
            context_builder=ctx_builder,
        )
        pipeline = pipeline_mod.ModelAutomationPipeline(
            context_builder=ctx_builder,
            bot_registry=bot_registry,
        )
        manager_mod = _load_optional_module(
            "self_coding_manager", fallback="menace.self_coding_manager"
        )
        return manager_mod.SelfCodingManager(
            engine,
            pipeline,
            bot_name=name,
            data_bot=data_bot,
            bot_registry=bot_registry,
        )
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - heavy bootstrap fallback
        raise RuntimeError(f"manager bootstrap failed: {exc}") from exc
    finally:
        _release_guard()
        current_depth = getattr(_BOOTSTRAP_STATE, "depth", 1) - 1
        if current_depth > 0:
            _BOOTSTRAP_STATE.depth = current_depth
        elif hasattr(_BOOTSTRAP_STATE, "depth"):
            delattr(_BOOTSTRAP_STATE, "depth")


def _load_optional_module(name: str, *, fallback: str | None = None) -> Any:
    """Attempt to load *name* falling back to ``fallback`` module when present."""

    try:
        return load_internal(name)
    except ModuleNotFoundError as exc:
        if fallback:
            module = sys.modules.get(fallback)
            if module is not None:
                logger.debug(
                    "using stub module %s for %s after import error",
                    fallback,
                    name,
                    exc_info=exc,
                )
                return module
        raise
    except Exception as exc:
        if fallback:
            module = sys.modules.get(fallback)
            if module is not None:
                logger.debug(
                    "using stub module %s for %s after runtime error",
                    fallback,
                    name,
                    exc_info=exc,
                )
                return module
        raise


F = TypeVar("F", bound=Callable[..., Any])


if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from vector_service.context_builder import ContextBuilder
else:  # pragma: no cover - runtime placeholder avoids hard dependency
    ContextBuilder = Any  # type: ignore


def manager_generate_helper(
    manager: SelfCodingManager,
    description: str,
    *,
    context_builder: "ContextBuilder",
    **kwargs: Any,
) -> str:
    """Invoke :meth:`SelfCodingEngine.generate_helper` under a manager token."""

    if not _ENGINE_AVAILABLE:
        message = "Self-coding engine is unavailable"
        if _ENGINE_IMPORT_ERROR is not None:
            message = f"{message}: {_ENGINE_IMPORT_ERROR}"
        raise RuntimeError(message)

    if context_builder is None:  # pragma: no cover - defensive
        raise TypeError("context_builder is required")

    engine = getattr(manager, "engine", None)
    if engine is None:  # pragma: no cover - defensive guard
        raise RuntimeError("manager must provide an engine for helper generation")

    token = MANAGER_CONTEXT.set(manager)
    previous_builder = getattr(engine, "context_builder", None)
    try:
        engine.context_builder = context_builder
        return engine.generate_helper(description, **kwargs)
    finally:
        engine.context_builder = previous_builder
        MANAGER_CONTEXT.reset(token)


def _resolve_helpers(
    obj: Any,
    registry: BotRegistry | None,
    data_bot: DataBot | None,
    orchestrator: EvolutionOrchestrator | None,
    manager: SelfCodingManager | None,
) -> tuple[
    BotRegistry,
    DataBot,
    EvolutionOrchestrator | None,
    str,
    SelfCodingManager,
]:
    """Resolve helper objects for *obj*.

    ``BotRegistry`` and ``DataBot`` are mandatory helpers.  When available, the
    existing ``EvolutionOrchestrator`` reference is also returned so callers can
    reuse or extend it.  ``manager`` takes precedence over any existing
    ``manager`` attribute on the object.  A :class:`RuntimeError` is raised if
    any mandatory helper cannot be resolved.
    """

    manager = manager or getattr(obj, "manager", None)

    if registry is None:
        registry = getattr(obj, "bot_registry", None)
        if registry is None and manager is not None:
            registry = getattr(manager, "bot_registry", None)
    if registry is None:
        raise RuntimeError("BotRegistry is required but was not provided")

    if data_bot is None:
        data_bot = getattr(obj, "data_bot", None)
        if data_bot is None and manager is not None:
            data_bot = getattr(manager, "data_bot", None)
    if data_bot is None:
        raise RuntimeError("DataBot is required but was not provided")

    if orchestrator is None:
        orchestrator = getattr(obj, "evolution_orchestrator", None)
        if orchestrator is None and manager is not None:
            orchestrator = getattr(manager, "evolution_orchestrator", None)

    try:
        module_path = inspect.getfile(obj.__class__)
    except Exception:  # pragma: no cover - best effort
        module_path = ""

    if manager is None:
        if _self_coding_runtime_available():
            try:
                name_local = getattr(
                    obj,
                    "name",
                    getattr(obj, "bot_name", obj.__class__.__name__),
                )
                manager = _bootstrap_manager(name_local, registry, data_bot)
            except Exception as exc:
                print(
                    f"[debug] Bootstrap failed during _resolve_helpers for {name_local} due to: {exc}"
                )
                logger.warning(
                    "SelfCodingManager bootstrap failed for %s: %s",
                    name_local,
                    exc,
                )
                manager = _DisabledSelfCodingManager(
                    bot_registry=registry,
                    data_bot=data_bot,
                )
        else:
            print(
                "[debug] Bootstrap failed during _resolve_helpers: self-coding runtime unavailable"
            )
            manager = _DisabledSelfCodingManager(
                bot_registry=registry,
                data_bot=data_bot,
            )

    return registry, data_bot, orchestrator, module_path, manager


def _ensure_threshold_entry(name: str, thresholds: Any) -> None:
    """Persist default threshold config for *name* when missing."""

    try:
        bots = (_load_config(None) or {}).get("bots", {})
    except Exception:  # pragma: no cover - best effort
        bots = {}
    if name in bots:
        return
    try:  # pragma: no cover - best effort
        update_thresholds(
            name,
            roi_drop=getattr(thresholds, "roi_drop", None),
            error_increase=getattr(thresholds, "error_threshold", None),
            test_failure_increase=getattr(thresholds, "test_failure_threshold", None),
        )
    except Exception:
        logger.exception("failed to persist thresholds for %s", name)


def self_coding_managed(
    *,
    bot_registry: BotRegistry,
    data_bot: DataBot,
    manager: SelfCodingManager | None = None,
) -> Callable[[type], type]:
    """Class decorator registering bots with helper services.

    ``bot_registry`` and ``data_bot`` may be provided as instances or callables
    that create the helper lazily.  When ``manager`` is supplied, instances will
    default to this :class:`SelfCodingManager` when resolving helpers.  The bot's
    name is registered with ``bot_registry`` the first time an instance is
    created so importing a module does not eagerly construct the helper
    dependencies.

    To limit which bots are wrapped, set ``MENACE_SELF_CODING_ALLOWLIST`` and
    ``MENACE_SELF_CODING_DENYLIST`` environment variables to comma separated
    lists of bot names.  By default every bot is eligible unless explicitly
    denied.  When an allowlist is provided only the listed bots are wrapped.
    """

    if bot_registry is None or data_bot is None:
        raise RuntimeError("BotRegistry and DataBot instances are required")

    def decorator(cls: type) -> type:
        orig_init = cls.__init__  # type: ignore[attr-defined]

        name = getattr(cls, "name", getattr(cls, "bot_name", cls.__name__))
        policy = get_self_coding_policy()
        if not policy.is_enabled(name):
            if name in getattr(policy, "denylist", frozenset()):
                logger.info(
                    "self_coding_managed disabled for bot %s (denylist=%s)",
                    name,
                    sorted(policy.denylist),
                )
            else:
                allowlist = getattr(policy, "allowlist", None)
                logger.info(
                    "self_coding_managed disabled for bot %s (allowlist=%s)",
                    name,
                    sorted(allowlist) if allowlist else [],
                )
            return cls
        print(f"[debug] self_coding_managed wrapping bot={name}")
        try:
            module_path = inspect.getfile(cls)
        except Exception:  # pragma: no cover - best effort
            module_path = ""
        if not module_path:
            module_path = getattr(cls, "__module__", "")
        manager_instance = manager
        module_provenance = _extract_module_provenance(sys.modules.get(cls.__module__))
        update_kwargs: dict[str, Any] = {}
        should_update = True
        register_as_coding = True
        register_as_coding_local = register_as_coding
        decision: _BootstrapDecision | None = None
        resolved_registry: BotRegistry | None = None
        resolved_data_bot: DataBot | None = None
        # ``ModelAutomationPipeline`` and other decorated bots lazily resolve
        # their helpers which can trigger nested bootstrap invocations within
        # the same thread (for example, when an orchestrator loads additional
        # managed bots while the current one is still initialising).  A plain
        # ``Lock`` would deadlock in this scenario because the second
        # invocation would block waiting for the already-held lock.  Use an
        # ``RLock`` so re-entrant access from the same thread succeeds while
        # still providing mutual exclusion across threads.
        bootstrap_lock = threading.RLock()
        bootstrap_event = threading.Event()
        bootstrap_in_progress = False
        bootstrap_error: BaseException | None = None
        bootstrap_done = False

        def _resolve_candidate(candidate: Any) -> Any:
            if getattr(candidate, "__self_coding_lazy__", False):
                return candidate()
            return candidate

        def _bootstrap_helpers() -> tuple[BotRegistry, DataBot]:
            nonlocal manager_instance, update_kwargs, should_update, register_as_coding
            nonlocal decision, resolved_registry, resolved_data_bot, bootstrap_done
            nonlocal bootstrap_in_progress, bootstrap_error

            if bootstrap_done and resolved_registry is not None and resolved_data_bot is not None:
                return resolved_registry, resolved_data_bot

            wait_for_completion = False
            with bootstrap_lock:
                if bootstrap_done and resolved_registry is not None and resolved_data_bot is not None:
                    return resolved_registry, resolved_data_bot
                if bootstrap_in_progress:
                    wait_for_completion = True
                else:
                    bootstrap_in_progress = True
                    bootstrap_event.clear()

            if wait_for_completion:
                bootstrap_event.wait()
                if bootstrap_error is not None:
                    raise bootstrap_error
                if not bootstrap_done or resolved_registry is None or resolved_data_bot is None:
                    raise RuntimeError("Bot helper bootstrap did not complete successfully")
                return resolved_registry, resolved_data_bot

            context: _BootstrapContext | None = None
            try:
                registry_obj = _resolve_candidate(bot_registry)
                data_bot_obj = _resolve_candidate(data_bot)
                context = _push_bootstrap_context(
                    registry=registry_obj,
                    data_bot=data_bot_obj,
                    manager=manager_instance,
                )
                if registry_obj is None or data_bot_obj is None:
                    raise RuntimeError("BotRegistry and DataBot instances are required")

                roi_t = err_t = None
                if hasattr(data_bot_obj, "reload_thresholds"):
                    try:
                        t = data_bot_obj.reload_thresholds(name)
                        roi_t = getattr(t, "roi_drop", None)
                        err_t = getattr(t, "error_threshold", None)
                        _ensure_threshold_entry(name, t)
                    except Exception:  # pragma: no cover - best effort
                        logger.exception("threshold reload failed for %s", name)

                manager_local = manager_instance
                runtime_available = _self_coding_runtime_available()
                if context is not None:
                    context.manager = manager_local
                if manager_local is None and runtime_available:
                    ready = True
                    missing: Iterable[str] = ()
                    if callable(ensure_self_coding_ready):
                        try:
                            ready, missing = ensure_self_coding_ready()
                        except Exception:  # pragma: no cover - best effort diagnostics
                            ready = False
                            missing = ()
                            logger.debug(
                                "self-coding dependency probe failed during manager bootstrap for %s",
                                name,
                                exc_info=True,
                            )
                    if ready:
                        try:
                            manager_local = manager_local or _bootstrap_manager(
                                name,
                                registry_obj,
                                data_bot_obj,
                            )
                        except RuntimeError as exc:
                            register_as_coding_local = False
                            logger.warning(
                                "self-coding manager unavailable for %s; %s",
                                name,
                                exc,
                            )
                            logger.debug(
                                "self-coding manager bootstrap failed for %s", name, exc_info=exc
                            )
                        else:
                            register_as_coding_local = True
                            if context is not None:
                                context.manager = manager_local
                    else:
                        register_as_coding_local = False
                    if missing:
                        logger.warning(
                            "Self-coding dependencies missing (%s); manager registration for %s is blocked and bootstrap will keep retrying until these modules are installed.",
                            ", ".join(sorted(missing)),
                            name,
                        )
                else:
                    register_as_coding_local = register_as_coding

                decision_local = decision
                update_kwargs_local = dict(update_kwargs)
                orchestrator_factory: Callable[..., Any] | None = None
                should_update_local = should_update
                if decision_local is None:
                    manager_sources: list[Any] = []
                    for candidate in (
                        manager_local,
                        getattr(manager_local, "bot_registry", None),
                        getattr(manager_local, "data_bot", None),
                        registry_obj,
                        data_bot_obj,
                    ):
                        if candidate is not None:
                            manager_sources.append(candidate)

                    provenance_decision = _resolve_provenance_decision(
                        name,
                        module_path,
                        manager_sources,
                        module_provenance,
                    )
                    if provenance_decision.mode == "missing":
                        reason = provenance_decision.reason or "provenance metadata unavailable"
                        logger.warning(
                            "Skipping bot update for %s because provenance metadata is unavailable (%s)",
                            name,
                            reason,
                        )
                        should_update_local = False
                        register_as_coding_local = False
                    else:
                        if (
                            provenance_decision.patch_id is not None
                            and "patch_id" not in update_kwargs_local
                        ):
                            update_kwargs_local["patch_id"] = provenance_decision.patch_id
                        if (
                            provenance_decision.commit is not None
                            and "commit" not in update_kwargs_local
                        ):
                            update_kwargs_local["commit"] = provenance_decision.commit
                        if provenance_decision.mode == "unsigned":
                            _warn_unsigned_once(name)
                    decision_local = _BootstrapDecision(
                        provenance=provenance_decision,
                        manager=manager_local,
                        _update_kwargs=update_kwargs_local,
                        should_update=should_update_local,
                        register_as_coding=register_as_coding_local,
                        hot_swap_active=_registry_hot_swap_active(registry_obj),
                    )

                hot_swap_active = False
                if decision_local:
                    decision_local.apply(target=cls)
                    update_kwargs_local = decision_local.update_kwargs
                    should_update_local = decision_local.should_update
                    register_as_coding_local = decision_local.register_as_coding
                    manager_local = decision_local.manager
                    hot_swap_active = decision_local.hot_swap_active
                    if context is not None:
                        context.manager = manager_local

                orchestrator_factory = update_kwargs_local.pop(
                    "orchestrator_factory", None
                )

                if (
                    register_as_coding_local
                    and manager_local is not None
                    and hasattr(manager_local, "register")
                ):
                    try:
                        manager_local.register(name, cls)
                    except Exception:  # pragma: no cover - best effort
                        logger.exception("manager registration failed for %s", name)

                if registry_obj is not None:
                    if register_as_coding_local:
                        try:
                            register_kwargs = dict(update_kwargs_local)
                            if (
                                decision_local is not None
                                and decision_local.provenance is not None
                                and "provenance" not in register_kwargs
                            ):
                                register_kwargs["provenance"] = decision_local.provenance
                            if orchestrator_factory is None and manager_local is not None:
                                orchestrator_factory = getattr(
                                    manager_local,
                                    "orchestrator_factory",
                                    None,
                                )
                            if orchestrator_factory is not None:
                                try:
                                    register_sig = inspect.signature(
                                        registry_obj.register_bot
                                    )
                                except (TypeError, ValueError):  # pragma: no cover - best effort
                                    register_sig = None
                                include_factory = False
                                if register_sig is None:
                                    include_factory = True
                                else:
                                    params = register_sig.parameters
                                    if "orchestrator_factory" in params:
                                        include_factory = True
                                    else:
                                        include_factory = any(
                                            param.kind
                                            == inspect.Parameter.VAR_KEYWORD
                                            for param in params.values()
                                        )
                                if include_factory:
                                    register_kwargs["orchestrator_factory"] = (
                                        orchestrator_factory
                                    )
                            registry_obj.register_bot(
                                name,
                                module_path,
                                manager=manager_local,
                                roi_threshold=roi_t,
                                error_threshold=err_t,
                                **register_kwargs,
                            )
                        except Exception:  # pragma: no cover - best effort
                            logger.exception("bot registration failed for %s", name)
                    else:
                        logger.debug(
                            "Skipping bot registration for %s because it is not in self-coding mode",
                            name,
                        )

                    patch_id = getattr(cls, "_self_coding_patch_id", None)
                    if patch_id is not None and hot_swap_active:
                        try:
                            registry_obj.mark_bot_patch_in_progress(name, patch_id)
                        except Exception:  # pragma: no cover - best effort
                            logger.exception(
                                "failed to mark bot patch %s as in progress for %s",
                                patch_id,
                                name,
                            )

                    if should_update_local:
                        try:
                            registry_obj.update_bot(name, module_path, **update_kwargs_local)
                        except Exception:  # pragma: no cover - best effort
                            logger.exception("bot update failed for %s", name)

                if context is not None:
                    context.manager = manager_local

                with bootstrap_lock:
                    cls.bot_registry = registry_obj  # type: ignore[attr-defined]
                    cls.data_bot = data_bot_obj  # type: ignore[attr-defined]
                    cls.manager = manager_local  # type: ignore[attr-defined]
                    cls._self_coding_manual_mode = not register_as_coding_local

                    manager_instance = manager_local
                    update_kwargs = update_kwargs_local
                    should_update = should_update_local
                    register_as_coding = register_as_coding_local
                    decision = decision_local
                    resolved_registry = registry_obj
                    resolved_data_bot = data_bot_obj
                    bootstrap_done = True
                    bootstrap_error = None
                    bootstrap_in_progress = False
                    bootstrap_event.set()
                return registry_obj, data_bot_obj
            except BaseException as exc:
                with bootstrap_lock:
                    bootstrap_error = exc
                    bootstrap_in_progress = False
                    bootstrap_event.set()
                raise
            finally:
                if context is not None:
                    _pop_bootstrap_context(context)

        cls.bot_registry = None  # type: ignore[attr-defined]
        cls.data_bot = None  # type: ignore[attr-defined]
        cls.manager = manager_instance  # type: ignore[attr-defined]
        cls._self_coding_manual_mode = True

        @wraps(orig_init)
        def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
            context = _current_bootstrap_context()
            if (
                context is not None
                and context.registry is not None
                and context.data_bot is not None
            ):
                registry_obj = context.registry
                data_bot_obj = context.data_bot
                manager_default = context.manager or manager_instance
            else:
                registry_obj, data_bot_obj = _bootstrap_helpers()
                manager_default = manager_instance
            init_kwargs = dict(kwargs)
            orchestrator: EvolutionOrchestrator | None = init_kwargs.get(
                "evolution_orchestrator"
            )
            manager_local: SelfCodingManager | None = init_kwargs.get(
                "manager", manager_default
            )

            cooperative_init_call(
                orig_init,
                self,
                *args,
                injected_keywords=COOPERATIVE_INIT_KWARGS,
                logger=logger,
                cls=cls,
                kwarg_trace=_record_cooperative_init_trace,
                **init_kwargs,
            )
            try:
                (
                    registry,
                    d_bot,
                    orchestrator,
                    _module_path,
                    manager_local,
                ) = _resolve_helpers(
                    self, registry_obj, data_bot_obj, orchestrator, manager_local
                )
            except RuntimeError as exc:
                raise RuntimeError(f"{cls.__name__}: {exc}") from exc

            name_local = getattr(self, "name", getattr(self, "bot_name", name))
            thresholds = None
            if hasattr(d_bot, "reload_thresholds"):
                try:
                    thresholds = d_bot.reload_thresholds(name_local)
                    update_thresholds(
                        name_local,
                        roi_drop=thresholds.roi_drop,
                        error_increase=thresholds.error_threshold,
                        test_failure_increase=thresholds.test_failure_threshold,
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to initialise thresholds for %s", name_local
                    )
            manual_mode = getattr(cls, "_self_coding_manual_mode", False)
            manager_is_active = _is_self_coding_manager(manager_local)
            if (
                _self_coding_runtime_available()
                and not manual_mode
                and not manager_is_active
            ):
                logger.warning(
                    "%s: self-coding runtime detected but manager unavailable; "
                    "running without autonomous patching",
                    name_local,
                )
                manual_mode = True
            self.manager = manager_local

            if not manager_is_active:
                if orchestrator is not None:
                    self.evolution_orchestrator = orchestrator
                return

            if not _self_coding_runtime_available():
                if orchestrator is not None:
                    self.evolution_orchestrator = orchestrator
                return

            orchestrator_boot_failed = False
            if orchestrator is None:
                try:
                    _capital_module = _load_optional_module(
                        "capital_management_bot",
                        fallback="menace.capital_management_bot",
                    )
                    CapitalManagementBot = _capital_module.CapitalManagementBot
                    _improvement_module = _load_optional_module(
                        "self_improvement.engine",
                        fallback="menace.self_improvement.engine",
                    )
                    SelfImprovementEngine = _improvement_module.SelfImprovementEngine
                    _evolution_manager_module = _load_optional_module(
                        "system_evolution_manager",
                        fallback="menace.system_evolution_manager",
                    )
                    SystemEvolutionManager = (
                        _evolution_manager_module.SystemEvolutionManager
                    )
                    _eo_module = _load_optional_module(
                        "evolution_orchestrator",
                        fallback="menace.evolution_orchestrator",
                    )
                    _EO = _eo_module.EvolutionOrchestrator

                    capital = CapitalManagementBot(data_bot=d_bot)
                    builder = create_context_builder()
                    improv = SelfImprovementEngine(
                        context_builder=builder,
                        data_bot=d_bot,
                        bot_name=name_local,
                    )
                    bot_list: list[str] = []
                    try:
                        bot_list = list(getattr(registry, "graph", {}).keys())
                    except Exception:
                        bot_list = []
                    evol_mgr = SystemEvolutionManager(bot_list)
                    orchestrator = _EO(
                        data_bot=d_bot,
                        capital_bot=capital,
                        improvement_engine=improv,
                        evolution_manager=evol_mgr,
                        selfcoding_manager=manager_local,
                    )
                except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: EvolutionOrchestrator dependencies unavailable: %s",
                        cls.__name__,
                        exc,
                    )
                    orchestrator_boot_failed = True
                    orchestrator = None
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        f"{cls.__name__}: EvolutionOrchestrator is required but could not be instantiated"
                    ) from exc

            if orchestrator is not None:
                self.evolution_orchestrator = orchestrator
                try:
                    manager_local.evolution_orchestrator = orchestrator
                except Exception:
                    pass
            elif orchestrator_boot_failed:
                self.evolution_orchestrator = None

            if getattr(manager_local, "quick_fix", None) is None:
                try:
                    _quick_fix_module = _load_optional_module(
                        "quick_fix_engine", fallback="menace.quick_fix_engine"
                    )
                    QuickFixEngine = _quick_fix_module.QuickFixEngine
                    ErrorDB = _load_optional_module(
                        "error_bot", fallback="menace.error_bot"
                    ).ErrorDB
                    _helper_fn = _load_optional_module(
                        "self_coding_manager", fallback="menace.self_coding_manager"
                    )._manager_generate_helper_with_builder
                except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: QuickFixEngine dependencies unavailable: %s",
                        cls.__name__,
                        exc,
                    )
                    manager_local.quick_fix = manager_local.quick_fix or None
                    ErrorDB = None
                    QuickFixEngine = None
                except Exception as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: QuickFixEngine initialisation failed: %s",
                        cls.__name__,
                        exc,
                    )
                    manager_local.quick_fix = manager_local.quick_fix or None
                    ErrorDB = None
                    QuickFixEngine = None
                if QuickFixEngine is not None and ErrorDB is not None:
                    engine = getattr(manager_local, "engine", None)
                    clayer = getattr(engine, "cognition_layer", None)
                    if clayer is None:
                        logger.warning(
                            "%s: QuickFixEngine requires a cognition_layer; skipping bootstrap",
                            cls.__name__,
                        )
                    else:
                        try:
                            builder = clayer.context_builder
                        except AttributeError as exc:
                            logger.warning(
                                "%s: QuickFixEngine missing context_builder: %s",
                                cls.__name__,
                                exc,
                            )
                        else:
                            error_db = getattr(self, "error_db", None) or getattr(
                                manager_local, "error_db", None
                            )
                            if error_db is None:
                                try:
                                    error_db = ErrorDB()
                                except Exception as exc:  # pragma: no cover
                                    logger.warning(
                                        "%s: failed to initialise ErrorDB for QuickFixEngine: %s",
                                        cls.__name__,
                                        exc,
                                    )
                                    error_db = None
                            if error_db is not None:
                                try:
                                    manager_local.quick_fix = QuickFixEngine(
                                        error_db,
                                        manager_local,
                                        context_builder=builder,
                                        helper_fn=_helper_fn,
                                    )
                                    manager_local.error_db = error_db
                                except Exception as exc:  # pragma: no cover - instantiation errors
                                    logger.warning(
                                        "%s: failed to initialise QuickFixEngine: %s",
                                        cls.__name__,
                                        exc,
                                    )
                                    manager_local.quick_fix = manager_local.quick_fix or None
            registries_seen = getattr(cls, "_self_coding_registry_ids", set())
            should_register = isinstance(registries_seen, set) and (
                id(registry) not in registries_seen
            )
            if should_register:
                try:
                    registry.register_bot(
                        name_local,
                        roi_threshold=getattr(thresholds, "roi_drop", None),
                        error_threshold=getattr(thresholds, "error_threshold", None),
                        manager=manager_local,
                        data_bot=d_bot,
                        is_coding_bot=True,
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("bot registration failed for %s", name_local)
                else:
                    registries_seen.add(id(registry))
                    cls._self_coding_registry_ids = registries_seen
            if orchestrator is not None:
                try:
                    orchestrator.register_bot(name_local)
                    logger.info("registered %s with EvolutionOrchestrator", name_local)
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "evolution orchestrator registration failed for %s", name_local
                    )
            if d_bot and getattr(d_bot, "db", None):
                try:
                    roi = float(d_bot.roi(name_local)) if hasattr(d_bot, "roi") else 0.0
                    d_bot.db.log_eval(name_local, "roi", roi)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed logging roi for %s", name_local)
                try:
                    d_bot.db.log_eval(name_local, "errors", 0.0)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed logging errors for %s", name_local)

        wrapped_init.__cooperative_safe__ = True  # type: ignore[attr-defined]
        cls.__init__ = wrapped_init  # type: ignore[assignment]

        for method_name in ("run", "execute"):
            orig_method = getattr(cls, method_name, None)
            if callable(orig_method):

                @wraps(orig_method)
                def wrapped_method(self, *args: Any, _orig=orig_method, **kwargs: Any):
                    start = time.time()
                    errors = 0
                    try:
                        result = _orig(self, *args, **kwargs)
                    except Exception:
                        errors = 1
                        raise
                    finally:
                        response_time = time.time() - start
                        try:
                            d_bot_local = getattr(self, "data_bot", None)
                            if d_bot_local is None:
                                manager = getattr(self, "manager", None)
                                d_bot_local = getattr(manager, "data_bot", None)
                            if d_bot_local:
                                name_local2 = getattr(
                                    self,
                                    "name",
                                    getattr(self, "bot_name", name),
                                )
                                d_bot_local.collect(
                                    name_local2,
                                    response_time=response_time,
                                    errors=errors,
                                )
                        except Exception:  # pragma: no cover - best effort
                            logger.exception("failed logging metrics for %s", name)
                    return result

                setattr(cls, method_name, wrapped_method)

        return cls

    return decorator
