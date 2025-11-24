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

import contextlib
import contextvars
import importlib.util
import sys
from collections import deque
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
from typing import Any, Callable, Iterator, Literal, TypeVar, TYPE_CHECKING
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

try:  # pragma: no cover - shared bus optional during tests
    from menace_sandbox.shared_event_bus import event_bus as _SHARED_EVENT_BUS
except Exception:  # pragma: no cover - flat layout fallback
    try:
        from shared_event_bus import event_bus as _SHARED_EVENT_BUS  # type: ignore
    except Exception:  # pragma: no cover - optional dependency missing
        _SHARED_EVENT_BUS = None

logger = logging.getLogger(__name__)
_COMM_BOT_BOOTSTRAP_STATE: dict[str, Any] | None = None


def _get_bootstrap_wait_timeout() -> float | None:
    """Return the maximum time (in seconds) to wait for helper bootstrap.

    The timeout is extended beyond the previous hard-coded 10s to accommodate
    slower initialisation paths.  It can be tuned via
    ``MENACE_BOOTSTRAP_WAIT_SECS``; specify ``"none"`` to disable timeouts.
    Invalid overrides fall back to a sensible default while emitting a warning
    for visibility.
    """

    default_timeout = 300.0
    raw_timeout = os.getenv("MENACE_BOOTSTRAP_WAIT_SECS")
    if not raw_timeout:
        return default_timeout
    if raw_timeout.lower() == "none":
        return None
    try:
        return max(1.0, float(raw_timeout))
    except ValueError:
        logger.warning(
            "Invalid MENACE_BOOTSTRAP_WAIT_SECS=%r; using default %ss",
            raw_timeout,
            default_timeout,
        )
        return default_timeout


_BOOTSTRAP_WAIT_TIMEOUT = _get_bootstrap_wait_timeout()


@dataclass(eq=False)
class _BootstrapContext:
    """Thread-scoped context shared during nested helper bootstrap."""

    registry: Any = None
    data_bot: Any = None
    manager: Any = None
    sentinel: Any = None
    pipeline: Any = None


@dataclass(eq=False)
class _BootstrapContextGuard:
    """Defers popping a ``_BootstrapContext`` until explicitly released."""

    context: _BootstrapContext | None
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.context is not None:
            _pop_bootstrap_context(self.context)


_BOOTSTRAP_THREAD_STATE = threading.local()
_SENTINEL_UNSET = object()


def _push_bootstrap_context(
    *, registry: Any, data_bot: Any, manager: Any, pipeline: Any | None = None
) -> _BootstrapContext:
    """Push a helper context onto the current thread's stack."""

    stack = getattr(_BOOTSTRAP_THREAD_STATE, "stack", None)
    if stack is None:
        stack = []
        _BOOTSTRAP_THREAD_STATE.stack = stack
    if pipeline is None:
        pipeline = getattr(_BOOTSTRAP_STATE, "pipeline", None)
    if pipeline is None:
        pipeline = _current_pipeline_context()
    if pipeline is None and stack:
        try:
            pipeline = stack[-1].pipeline
        except Exception:
            pipeline = None
    context = _BootstrapContext(
        registry=registry,
        data_bot=data_bot,
        manager=manager,
        sentinel=manager,
        pipeline=pipeline,
    )
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


def _is_bootstrap_owner(candidate: Any) -> bool:
    """Return ``True`` if *candidate* represents the bootstrap owner sentinel."""

    return bool(getattr(candidate, "_bootstrap_owner_marker", False))


def _is_bootstrap_placeholder(candidate: Any) -> bool:
    """Return ``True`` when *candidate* represents a bootstrap placeholder."""

    if candidate is None:
        return False
    if isinstance(candidate, _BootstrapManagerSentinel) or _is_bootstrap_owner(candidate):
        return True
    return bool(getattr(candidate, "_self_coding_bootstrap_placeholder", False))


def _mark_bootstrap_placeholder(candidate: Any) -> None:
    """Flag *candidate* so placeholder propagation can safely override it."""

    if candidate is None:
        return
    try:
        setattr(candidate, "_self_coding_bootstrap_placeholder", True)
    except Exception:  # pragma: no cover - best effort marking
        logger.debug(
            "failed to mark %s as a bootstrap placeholder", candidate, exc_info=True
        )


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

_PIPELINE_CONTEXT = contextvars.ContextVar("PIPELINE_CONTEXT", default=None)
_HELPER_MANAGER_OVERRIDE = contextvars.ContextVar(
    "SELF_CODING_HELPER_MANAGER_OVERRIDE", default=None
)


def _current_helper_manager_override() -> Any | None:
    """Return the manager override active for helper bootstrap, if any."""

    try:
        return _HELPER_MANAGER_OVERRIDE.get()
    except LookupError:  # pragma: no cover - unset context
        return None


def _push_helper_manager_override(
    manager: Any | None,
) -> contextvars.Token[Any] | None:
    """Expose *manager* via the helper-override context variable."""

    if manager is None:
        return None
    try:
        return _HELPER_MANAGER_OVERRIDE.set(manager)
    except LookupError:  # pragma: no cover - context var best effort
        return None


def _reset_helper_manager_override(
    token: contextvars.Token[Any] | None,
) -> None:
    """Reset the helper override context to *token* when provided."""

    if token is None:
        return
    try:
        _HELPER_MANAGER_OVERRIDE.reset(token)
    except LookupError:  # pragma: no cover - defensive reset
        pass


@contextlib.contextmanager
def fallback_helper_manager(
    *, bot_registry: Any, data_bot: Any
) -> Iterator[Any]:
    """Seed helper bootstrap with a disabled manager placeholder."""

    manager = _DisabledSelfCodingManager(
        bot_registry=bot_registry,
        data_bot=data_bot,
        bootstrap_placeholder=True,
    )
    token = _push_helper_manager_override(manager)
    try:
        yield manager
    finally:
        _reset_helper_manager_override(token)


def _emit_disabled_manager_metric(payload: Mapping[str, Any]) -> None:
    """Emit telemetry for disabled-manager fallbacks via available hooks."""

    emitted = False
    manager_context = None
    if MANAGER_CONTEXT is not None:
        try:
            manager_context = MANAGER_CONTEXT.get()
        except LookupError:  # pragma: no cover - unset context
            manager_context = None
    metric_hook = getattr(manager_context, "emit_metric", None)
    if callable(metric_hook):
        try:
            metric_hook("self_coding.disabled_manager", payload)
            emitted = True
        except Exception:  # pragma: no cover - telemetry best effort
            logger.debug("manager-context telemetry emission failed", exc_info=True)
    if not emitted and _SHARED_EVENT_BUS is not None:
        publish = getattr(_SHARED_EVENT_BUS, "publish", None)
        if callable(publish):
            try:
                publish("self_coding.disabled_manager", payload)
                emitted = True
            except Exception:  # pragma: no cover - bus best effort
                logger.debug("event bus telemetry emission failed", exc_info=True)
    if emitted:
        return
    logger.info(
        "disabled manager telemetry emitted via logger",
        extra={
            "disabled_manager": payload,
            "event": "self_coding.disabled_manager",
        },
    )


def _current_pipeline_context() -> Any | None:
    """Return the pipeline currently exposed via ``pipeline_context_scope``."""

    try:
        return _PIPELINE_CONTEXT.get()
    except LookupError:  # pragma: no cover - context var unset
        return None


@contextlib.contextmanager
def pipeline_context_scope(pipeline: Any | None) -> Iterator[None]:
    """Temporarily expose *pipeline* to nested helper bootstrap logic."""

    if pipeline is None:
        yield None
        return
    token = None
    try:
        token = _PIPELINE_CONTEXT.set(pipeline)
    except Exception:  # pragma: no cover - context vars best effort
        token = None
    try:
        yield None
    finally:
        if token is not None:
            try:
                _PIPELINE_CONTEXT.reset(token)
            except Exception:  # pragma: no cover - context vars best effort
                logger.debug(
                    "failed to reset pipeline context", exc_info=True
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


def normalise_manager_arg(
    manager: "SelfCodingManager | None",
    owner: object | Mapping[str, object] | None,
    *,
    fallback: "SelfCodingManager | None" = None,
) -> "SelfCodingManager | None":
    """Return ``manager`` unless ``None`` falling back to ``owner`` or ``fallback``.

    ``manager`` arguments frequently originate from sentinels that deliberately
    evaluate ``False`` (for example when proxying disabled self-coding
    runtimes).  Treat ``None`` as the only signal to replace the supplied
    manager.  ``owner`` may be a class, module, or globals dict that exposes a
    ``manager`` attribute.  ``fallback`` provides a final default when neither
    the argument nor ``owner`` supply a concrete manager.
    """

    if manager is not None:
        return manager
    candidate = None
    if owner is not None:
        if isinstance(owner, Mapping):
            candidate = owner.get("manager")
        else:
            candidate = getattr(owner, "manager", None)
    if candidate is not None:
        return candidate
    return fallback


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
        "manager",
        "initial_manager",
        "_pending_manager_helpers",
        "_lazy_helper_attrs",
        "_registered_bot_attrs",
        "_bots",
        "_bot_attribute_order",
        "_should_defer_manager_helpers",
        "_last_patch_id",
        "_last_commit_hash",
        "_bootstrap_owner_marker",
        "_self_coding_bootstrap_placeholder",
        "_bootstrap_owner_token",
        "_bootstrap_runtime_marker",
        "_manager",
    )

    def __init__(
        self,
        *,
        bot_registry: Any,
        data_bot: Any,
        bootstrap_owner: bool = False,
        bootstrap_placeholder: bool = False,
        bootstrap_runtime: bool = False,
    ) -> None:
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.engine = SimpleNamespace(
            cognition_layer=SimpleNamespace(context_builder=None)
        )
        # Mark quick_fix as initialised so downstream code skips heavy bootstrap.
        self.quick_fix = object()
        self.error_db = None
        self.evolution_orchestrator = None
        self.manager: Any | None = None
        self.initial_manager: Any | None = None
        self._pending_manager_helpers: dict[str, Any] = {}
        self._lazy_helper_attrs: set[str] = set()
        self._registered_bot_attrs: set[str] = set()
        self._bots: list[Any] = []
        self._bot_attribute_order: tuple[str, ...] = ()
        self._should_defer_manager_helpers = False
        self._last_patch_id = None
        self._last_commit_hash = None
        self._bootstrap_owner_marker = bootstrap_owner
        self._self_coding_bootstrap_placeholder = bootstrap_placeholder
        self._bootstrap_owner_token: Any | None = None
        self._bootstrap_runtime_marker = bootstrap_runtime
        self._manager: Any | None = None

    def __bool__(self) -> bool:
        """Return ``True`` whenever acting as a bootstrap guard."""

        return bool(self._bootstrap_owner_marker or self._bootstrap_runtime_marker)

    @property
    def bootstrap_runtime_active(self) -> bool:
        """Return ``True`` when the placeholder acts as a runtime shim."""

        return bool(self._bootstrap_runtime_marker)

    def mark_bootstrap_runtime(self) -> None:
        """Flag this placeholder so helpers treat it as a runtime sentinel."""

        self._bootstrap_runtime_marker = True

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


class _BootstrapOwnerSentinel(_DisabledSelfCodingManager):
    """Sentinel exposed while the fallback bootstrap pipeline is active.

    The sentinel deliberately evaluates truthy so downstream helpers treat it as
    an active manager while ``_bootstrap_manager`` is still constructing the
    real ``SelfCodingManager``.  This ensures bootstrap heuristics do not try to
    re-enter while the fallback pipeline is wiring itself up.
    """

    __slots__ = _DisabledSelfCodingManager.__slots__ + (
        "_promotion_callbacks",
        "_resolved_manager",
        "_bootstrap_state_guard",
        "_owner_active",
        "_pipeline_promoter",
        "_bootstrap_owner_delegate",
        "_extra_manager_sentinels",
    )

    def __init__(self, *, bot_registry: Any, data_bot: Any) -> None:
        super().__init__(
            bot_registry=bot_registry,
            data_bot=data_bot,
            bootstrap_owner=True,
        )
        self._promotion_callbacks: list[Callable[[Any], None]] = []
        self._resolved_manager: Any | None = None
        self._bootstrap_state_guard: Callable[[], None] | None = None
        self._owner_active = False
        self._pipeline_promoter: Callable[[Any], None] | None = None
        self._bootstrap_owner_delegate: Any | None = None
        self._extra_manager_sentinels: list[Any] = []

    def __bool__(self) -> bool:  # pragma: no cover - trivial truthiness
        """Always evaluate truthy so helper heuristics keep deferring."""

        return True

    @property
    def bootstrap_owner_active(self) -> bool:
        """Return ``True`` while this sentinel guards fallback bootstrap."""

        return self._owner_active

    def claim_bootstrap_state_guard(self, guard: Callable[[], None]) -> bool:
        """Adopt *guard* so the owner controls bootstrap state cleanup."""

        if not callable(guard):
            return False
        self._bootstrap_state_guard = guard
        self._owner_active = True
        return True

    def bind_pipeline_promoter(self, callback: Callable[[Any], None]) -> None:
        """Remember the *callback* used to promote pipeline managers."""

        if callable(callback):
            self._pipeline_promoter = callback

    def _release_bootstrap_state(self) -> None:
        guard = self._bootstrap_state_guard
        self._bootstrap_state_guard = None
        self._owner_active = False
        if callable(guard):
            try:
                guard()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug(
                    "bootstrap owner sentinel failed to release guard", exc_info=True
                )

    def add_promotion_callback(self, callback: Callable[[Any], None]) -> None:
        """Subscribe *callback* to manager promotion events."""

        if not callable(callback):
            return
        if self._resolved_manager is not None:
            try:
                callback(self._resolved_manager)
            except Exception:  # pragma: no cover - callbacks best-effort
                logger.debug("bootstrap owner callback failed", exc_info=True)
            return
        self._promotion_callbacks.append(callback)

    def add_delegate_callback(self, callback: Callable[[Any], None]) -> None:
        """Mirror the sentinel API so decorators can subscribe uniformly."""

        self.add_promotion_callback(callback)

    def mark_ready(
        self,
        real_manager: Any,
        *,
        sentinel: Any | None = None,
        extra_sentinels: Iterable[Any] | None = None,
    ) -> None:
        """Swap this sentinel out for *real_manager* once bootstrap completes."""

        logger.debug(
            "promoting bootstrap owner sentinel with %s", real_manager
        )
        if sentinel is not None:
            attach_delegate = getattr(sentinel, "attach_delegate", None)
            if callable(attach_delegate):
                try:
                    attach_delegate(real_manager)
                except Exception:  # pragma: no cover - best effort linkage
                    logger.debug(
                        "bootstrap owner failed to attach sentinel delegate", exc_info=True
                    )
        self._resolved_manager = real_manager
        if real_manager is not None:
            try:
                self.bot_registry = getattr(real_manager, "bot_registry", self.bot_registry)
            except Exception:  # pragma: no cover - best effort injection
                logger.debug(
                    "bootstrap owner failed to refresh bot_registry", exc_info=True
                )
            try:
                self.data_bot = getattr(real_manager, "data_bot", self.data_bot)
            except Exception:  # pragma: no cover - best effort injection
                logger.debug(
                    "bootstrap owner failed to refresh data_bot", exc_info=True
                )

        if extra_sentinels:
            for candidate in extra_sentinels:
                if candidate is None:
                    continue
                if any(existing is candidate for existing in self._extra_manager_sentinels):
                    continue
                self._extra_manager_sentinels.append(candidate)

        promoter = self._pipeline_promoter
        if promoter is None and real_manager is not None:
            pipeline = getattr(real_manager, "pipeline", None)
            promoter = lambda mgr, *, extra_sentinels=None: _promote_pipeline_manager(  # type: ignore[assignment]
                pipeline,
                mgr,
                sentinel,
                extra_sentinels=extra_sentinels,
            )

        if callable(promoter) and real_manager is not None:
            logger.debug(
                "promoting fallback pipeline references for %s", real_manager
            )
            try:
                promote_kwargs: dict[str, Any] = {}
                if self._extra_manager_sentinels:
                    promote_kwargs["extra_sentinels"] = tuple(
                        dict.fromkeys(self._extra_manager_sentinels)
                    )
                promoter(real_manager, **promote_kwargs)
            except Exception:  # pragma: no cover - promotion must be best effort
                logger.debug(
                    "bootstrap owner pipeline promotion failed", exc_info=True
                )
            else:
                logger.debug(
                    "fallback pipeline promotion finished for %s", real_manager
                )

        callbacks = tuple(self._promotion_callbacks)
        self._promotion_callbacks.clear()
        self._release_bootstrap_state()
        for callback in callbacks:
            try:
                callback(real_manager)
            except Exception:  # pragma: no cover - callbacks must be best-effort
                logger.debug("bootstrap owner callback failed", exc_info=True)


_BOOTSTRAP_STATE = threading.local()
_DEFERRED_SENTINEL_CALLBACKS: set[int] = set()


def _register_bootstrap_helper_callback(callback: Callable[[Any], None]) -> None:
    if not callable(callback):
        return
    callbacks = getattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks", None)
    if callbacks is None:
        callbacks = []
        _BOOTSTRAP_STATE.helper_promotion_callbacks = callbacks
    callbacks.append(callback)


def _deregister_bootstrap_helper_callback(callback: Callable[[Any], None]) -> None:
    callbacks = getattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks", None)
    if not callbacks:
        return
    try:
        callbacks.remove(callback)
    except ValueError:
        return
    if not callbacks and hasattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks"):
        delattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks")


def _consume_bootstrap_helper_callbacks() -> list[Callable[[Any], None]]:
    callbacks = getattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks", None)
    if not callbacks:
        return []
    try:
        delattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks")
    except AttributeError:
        pass
    return list(callbacks)


@dataclass
class _ManagerBootstrapPlan:
    manager: Any | None
    sentinel: Any | None
    defer: bool
    should_bootstrap: bool
    seed_owner_sentinel: bool = False


def _should_bootstrap_manager(candidate: Any) -> _ManagerBootstrapPlan:
    """Return strategy hints for *candidate* based on sentinel state."""

    sentinel_manager = getattr(_BOOTSTRAP_STATE, "sentinel_manager", None)
    bootstrap_depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
    candidate_sentinel = candidate if _is_bootstrap_placeholder(candidate) else None
    plan = _ManagerBootstrapPlan(
        manager=candidate,
        sentinel=candidate_sentinel,
        defer=False,
        should_bootstrap=candidate is None,
    )
    if candidate is None and bootstrap_depth > 0:
        placeholder = None
        if _is_bootstrap_placeholder(sentinel_manager):
            placeholder = sentinel_manager
        if placeholder is not None:
            plan.manager = placeholder
            plan.sentinel = placeholder
        else:
            plan.seed_owner_sentinel = True
        plan.defer = True
        plan.should_bootstrap = False
        return plan
    if candidate_sentinel is None:
        return plan
    owner_active = bool(getattr(candidate_sentinel, "bootstrap_owner_active", False))
    owner_marker = _is_bootstrap_owner(candidate_sentinel)
    owner_registered = owner_marker and candidate_sentinel is sentinel_manager
    owner_in_use = owner_active
    if owner_marker and (bootstrap_depth > 0 or owner_registered):
        owner_in_use = True
    if owner_registered and bootstrap_depth > 0:
        plan.defer = True
        plan.should_bootstrap = False
        plan.manager = candidate_sentinel
        return plan
    if owner_in_use:
        plan.defer = True
        plan.should_bootstrap = False
        plan.manager = candidate_sentinel
        return plan
    plan.manager = None
    plan.should_bootstrap = True
    return plan


def _spawn_nested_bootstrap_owner(
    bot_registry: Any, data_bot: Any
) -> Any:
    """Return a bootstrap owner sentinel for nested helper bootstrap flows."""

    try:
        return _BootstrapOwnerSentinel(
            bot_registry=bot_registry,
            data_bot=data_bot,
        )
    except Exception:  # pragma: no cover - fallback to disabled sentinel
        logger.debug(
            "failed to construct bootstrap owner sentinel; falling back to disabled manager",
            exc_info=True,
        )
        return _DisabledSelfCodingManager(
            bot_registry=bot_registry,
            data_bot=data_bot,
            bootstrap_owner=True,
            bootstrap_placeholder=True,
        )


def _using_bootstrap_sentinel(candidate: Any | None = None) -> bool:
    """Return ``True`` when the current bootstrap flow relies on a sentinel."""

    def _sentinel_guard_active(value: Any | None) -> bool:
        if value is None:
            return False
        if not _is_bootstrap_owner(value):
            return False
        return bool(getattr(value, "bootstrap_owner_active", False))

    if _is_bootstrap_placeholder(candidate) and _sentinel_guard_active(candidate):
        return True
    sentinel_manager = getattr(_BOOTSTRAP_STATE, "sentinel_manager", None)
    if sentinel_manager is None:
        return False
    if candidate is sentinel_manager and _sentinel_guard_active(candidate):
        return True
    depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
    return depth > 0 and _sentinel_guard_active(sentinel_manager)


class _BootstrapManagerSentinel(_DisabledSelfCodingManager):
    """Truthful proxy used while the real manager is initialising."""

    __slots__ = _DisabledSelfCodingManager.__slots__ + (
        "_delegate",
        "_delegate_callbacks",
        "_owner_registry",
        "_owner_data_bot",
        "_bootstrap_owner_delegate",
    )

    def __init__(self, *, bot_registry: Any, data_bot: Any) -> None:
        super().__init__(bot_registry=bot_registry, data_bot=data_bot)
        self._delegate: Any | None = None
        self._delegate_callbacks: list[Callable[[Any], None]] = []
        self._owner_registry = bot_registry
        self._owner_data_bot = data_bot
        self._bootstrap_owner_delegate: Any | None = None

    def __bool__(self) -> bool:  # pragma: no cover - truthiness is trivial
        return True

    def __getattr__(self, attr: str) -> Any:
        delegate = self._delegate
        if delegate is not None:
            return getattr(delegate, attr)
        raise AttributeError(attr)

    def resolve(self) -> Any:
        """Return the concrete manager once attached, falling back to ``self``."""

        return self._delegate or self

    def claim_bootstrap_state_guard(
        self, guard: Callable[[], None] | None
    ) -> bool:
        """Delegate guard ownership to the bootstrap owner sentinel when present."""

        if guard is None:
            return False
        delegate = self._bootstrap_owner_delegate
        claim_guard = getattr(delegate, "claim_bootstrap_state_guard", None)
        if not callable(claim_guard):
            return False
        try:
            return bool(claim_guard(guard))
        except Exception:  # pragma: no cover - delegate coordination best-effort
            logger.debug(
                "bootstrap manager sentinel failed to claim guard", exc_info=True
            )
        return False

    def _release_bootstrap_state(self) -> None:
        """Proxy guard release requests to the bootstrap owner sentinel."""

        delegate = self._bootstrap_owner_delegate
        release_guard = getattr(delegate, "_release_bootstrap_state", None)
        if not callable(release_guard):
            return
        try:
            release_guard()
        except Exception:  # pragma: no cover - delegate coordination best-effort
            logger.debug(
                "bootstrap manager sentinel failed to release guard", exc_info=True
            )

    def add_delegate_callback(self, callback: Callable[[Any], None]) -> None:
        """Subscribe *callback* to manager promotion events."""

        if not callable(callback):
            return
        delegate = self._delegate
        if delegate is not None:
            try:
                callback(delegate)
            except Exception:  # pragma: no cover - callbacks must be best-effort
                logger.debug("sentinel delegate callback failed", exc_info=True)
            return
        self._delegate_callbacks.append(callback)

    def attach_delegate(self, real_manager: Any) -> None:
        """Attach *real_manager* and notify any pending callbacks."""

        self._delegate = real_manager
        if real_manager is not None:
            registry = getattr(real_manager, "bot_registry", None)
            data_bot = getattr(real_manager, "data_bot", None)
            if registry is None and self._owner_registry is not None:
                try:
                    setattr(real_manager, "bot_registry", self._owner_registry)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("delegate registry injection failed", exc_info=True)
            if data_bot is None and self._owner_data_bot is not None:
                try:
                    setattr(real_manager, "data_bot", self._owner_data_bot)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("delegate data bot injection failed", exc_info=True)
            self.bot_registry = getattr(real_manager, "bot_registry", self.bot_registry)
            self.data_bot = getattr(real_manager, "data_bot", self.data_bot)

        callbacks = tuple(self._delegate_callbacks)
        self._delegate_callbacks.clear()
        for callback in callbacks:
            try:
                callback(real_manager)
            except Exception:  # pragma: no cover - callbacks are best-effort
                logger.debug("sentinel delegate callback failed", exc_info=True)


class _BootstrapManagerPromise:
    """Coordinates re-entrant helpers awaiting an active bootstrap."""

    __slots__ = ("waiters", "fallback_factory")

    def __init__(
        self, *, fallback_factory: Callable[[], Any] | None = None
    ) -> None:
        self.waiters: list[Any] = []
        self.fallback_factory = fallback_factory

    def add_waiter(self, sentinel: Any) -> bool:
        """Register *sentinel* to be promoted once the owner resolves."""

        if sentinel is None:
            return False
        attach_delegate = getattr(sentinel, "attach_delegate", None)
        if not callable(attach_delegate):
            return False
        self.waiters.append(sentinel)
        return True

    def resolve(self, manager: Any | None) -> None:
        """Promote all waiters with *manager* or the fallback placeholder."""

        if manager is None and callable(self.fallback_factory):
            try:
                manager = self.fallback_factory()
            except Exception:  # pragma: no cover - fallback must stay best-effort
                logger.debug(
                    "failed to construct fallback manager for re-entrant bootstrap waiters",
                    exc_info=True,
                )
                manager = None
        waiters = tuple(self.waiters)
        self.waiters.clear()
        if manager is None and not waiters:
            return
        for sentinel in waiters:
            attach_delegate = getattr(sentinel, "attach_delegate", None)
            if not callable(attach_delegate):
                continue
            try:
                attach_delegate(manager)
            except Exception:  # pragma: no cover - callbacks must be best-effort
                logger.debug(
                    "re-entrant bootstrap sentinel promotion failed", exc_info=True
                )


def _peek_owner_promise(owner_guard: Any) -> _BootstrapManagerPromise | None:
    """Return the promise registered for *owner_guard* when present."""

    promises = getattr(_BOOTSTRAP_STATE, "owner_promises", None)
    if not promises:
        return None
    return promises.get(owner_guard)


def _ensure_owner_promise(
    owner_guard: Any, fallback_factory: Callable[[], Any] | None = None
) -> _BootstrapManagerPromise:
    """Return or create the promise coordinating re-entrant waiters."""

    promises = getattr(_BOOTSTRAP_STATE, "owner_promises", None)
    if promises is None:
        promises = {}
        _BOOTSTRAP_STATE.owner_promises = promises
    promise = promises.get(owner_guard)
    if promise is None:
        promise = _BootstrapManagerPromise(fallback_factory=fallback_factory)
        promises[owner_guard] = promise
    elif promise.fallback_factory is None and callable(fallback_factory):
        promise.fallback_factory = fallback_factory
    return promise


def _settle_owner_promise(owner_guard: Any, manager: Any | None) -> None:
    """Resolve and discard the promise registered for *owner_guard*."""

    promises = getattr(_BOOTSTRAP_STATE, "owner_promises", None)
    if not promises:
        return
    promise = promises.pop(owner_guard, None)
    if promise is None:
        return
    try:
        promise.resolve(manager)
    finally:
        if not promises and hasattr(_BOOTSTRAP_STATE, "owner_promises"):
            delattr(_BOOTSTRAP_STATE, "owner_promises")


def _create_bootstrap_manager_sentinel(
    *, bot_registry: Any, data_bot: Any
) -> _BootstrapManagerSentinel:
    """Return a sentinel manager used to guard the legacy bootstrap path."""

    return _BootstrapManagerSentinel(
        bot_registry=bot_registry,
        data_bot=data_bot,
    )


def _activate_bootstrap_sentinel(manager: Any | None) -> Callable[[], None]:
    """Temporarily expose *manager* as the active bootstrap sentinel."""

    previous_sentinel = getattr(_BOOTSTRAP_STATE, "sentinel_manager", _SENTINEL_UNSET)
    sentinel_was_set = manager is not None
    if sentinel_was_set:
        _BOOTSTRAP_STATE.sentinel_manager = manager
    depth_missing = not hasattr(_BOOTSTRAP_STATE, "depth")
    previous_depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
    depth_overridden = previous_depth <= 0
    if depth_overridden:
        _BOOTSTRAP_STATE.depth = 1

    def _restore() -> None:
        if depth_overridden:
            if depth_missing:
                try:
                    delattr(_BOOTSTRAP_STATE, "depth")
                except AttributeError:  # pragma: no cover - best effort cleanup
                    pass
            else:
                _BOOTSTRAP_STATE.depth = previous_depth
        if sentinel_was_set:
            if previous_sentinel is _SENTINEL_UNSET:
                try:
                    delattr(_BOOTSTRAP_STATE, "sentinel_manager")
                except AttributeError:  # pragma: no cover - best effort cleanup
                    pass
            else:
                _BOOTSTRAP_STATE.sentinel_manager = previous_sentinel

    return _restore


def _claim_bootstrap_owner_guard(
    manager: Any, restore_callback: Callable[[], None]
) -> tuple[bool, Callable[[], None] | None]:
    """Mark *manager* as the active bootstrap owner when supported."""

    owner_guard_attached = False
    release_owner_guard = getattr(manager, "_release_bootstrap_state", None)
    claim_guard = getattr(manager, "claim_bootstrap_state_guard", None)
    if callable(claim_guard):
        try:
            owner_guard_attached = bool(claim_guard(restore_callback))
        except Exception:  # pragma: no cover - best effort coordination
            logger.debug(
                "sentinel failed to claim bootstrap guard", exc_info=True
            )
    return owner_guard_attached, release_owner_guard


_BOOTSTRAP_HELPER_ATTR_HINTS = (
    "comms_bot",
    "synthesis_bot",
    "diagnostic_manager",
    "planner",
    "aggregator",
    "helper",
    "helpers",
    "_helpers",
    "bots",
    "_bots",
)


def _iter_nested_bootstrap_values(value: Any) -> Iterator[Any]:
    if value is None:
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_nested_bootstrap_values(nested)
        return
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        for nested in value:
            yield from _iter_nested_bootstrap_values(nested)
        return
    yield value


def _iter_bootstrap_helper_candidates(root: Any) -> Iterator[Any]:
    if root is None:
        return
    for attr in _BOOTSTRAP_HELPER_ATTR_HINTS:
        try:
            value = getattr(root, attr)
        except Exception:
            continue
        for candidate in _iter_nested_bootstrap_values(value):
            yield candidate
    mapping = getattr(root, "__dict__", None)
    if isinstance(mapping, dict):
        for value in mapping.values():
            for candidate in _iter_nested_bootstrap_values(value):
                yield candidate


def _looks_like_helper_candidate(candidate: Any) -> bool:
    if candidate is None:
        return False
    if isinstance(candidate, (str, bytes, bytearray, int, float, complex, bool)):
        return False
    if isinstance(candidate, (list, tuple, set, frozenset, Mapping)):
        return False
    for attr in ("manager", "initial_manager", "bot_registry", "data_bot"):
        try:
            if hasattr(candidate, attr):
                return True
        except Exception:
            continue
    try:
        getattr(candidate, "_self_coding_pending_manager")
    except Exception:
        return False
    return True


def _looks_like_pipeline_candidate(candidate: Any) -> bool:
    """Return ``True`` when *candidate* resembles a pipeline instance."""

    if candidate is None:
        return False
    cls = getattr(candidate, "__class__", None)
    name = getattr(cls, "__name__", "")
    if not name or "Pipeline" not in name:
        return False
    if not name.endswith("ModelAutomationPipeline"):
        return False
    for attr in ("context_builder", "manager", "_bot_attribute_order"):
        if not hasattr(candidate, attr):
            return False
    return True


def _is_model_automation_pipeline_class(candidate: Any) -> bool:
    """Return ``True`` when *candidate* represents the pipeline class."""

    name = getattr(candidate, "__name__", "")
    if not name:
        return False
    return name == "ModelAutomationPipeline" or name.endswith("ModelAutomationPipeline")


def _resolve_bootstrap_pipeline_candidate(pipeline: Any | None) -> Any | None:
    """Return an existing pipeline involved in the current bootstrap flow."""

    if _looks_like_pipeline_candidate(pipeline):
        return pipeline
    context_candidate = _current_pipeline_context()
    if _looks_like_pipeline_candidate(context_candidate):
        return context_candidate
    context = _current_bootstrap_context()
    if context is None:
        return None
    candidate = getattr(context, "pipeline", None)
    if _looks_like_pipeline_candidate(candidate):
        return candidate
    return None


def _propagate_placeholder_to_helpers(
    root: Any,
    placeholder: Any,
    visited: set[int],
) -> None:
    pending: deque[Any] = deque()
    for candidate in _iter_bootstrap_helper_candidates(root):
        pending.append(candidate)
    while pending:
        candidate = pending.popleft()
        if candidate is None:
            continue
        key = id(candidate)
        if key in visited:
            continue
        visited.add(key)
        if not _looks_like_helper_candidate(candidate):
            continue
        updated = False
        for attr in ("manager", "initial_manager"):
            try:
                current = getattr(candidate, attr)
            except Exception:
                current = None
            if current is placeholder:
                continue
            if current is not None and not _is_bootstrap_placeholder(current):
                continue
            try:
                setattr(candidate, attr, placeholder)
            except Exception:  # pragma: no cover - best effort assignment
                logger.debug(
                    "failed to propagate %s placeholder on %s",
                    attr,
                    candidate,
                    exc_info=True,
                )
            else:
                updated = True
        if updated:
            for nested in _iter_bootstrap_helper_candidates(candidate):
                pending.append(nested)


def _seed_placeholder_bootstrap_fields(target: Any, placeholder: Any) -> bool:
    """Ensure *target* can accept *placeholder* even before ``__init__`` runs.

    The shim installs manager placeholders twice: once before ``__init__`` to
    prevent recursive helper bootstrap and once afterwards when the pipeline
    propagates the sentinel through its hierarchy.  Pre-seeding the internal
    bookkeeping structures keeps `_activate_deferred_helpers` and related
    routines safe to call even though the pipeline has not finished
    initialising yet.
    """

    if target is None or placeholder is None:
        return False
    seeded = False
    defaults: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("_pending_manager_helpers", dict),
        ("_lazy_helper_attrs", set),
        ("_registered_bot_attrs", set),
        ("_bots", list),
    )
    for attr, factory in defaults:
        if hasattr(target, attr):
            continue
        try:
            object.__setattr__(target, attr, factory())
        except Exception:  # pragma: no cover - best effort defaults
            logger.debug(
                "failed to seed %s before manager placeholder injection on %s",
                attr,
                target,
                exc_info=True,
            )
        else:
            seeded = True
    if not hasattr(target, "_bot_attribute_order"):
        order = tuple(getattr(target, "_BOT_ATTRIBUTE_ORDER", ()))
        try:
            object.__setattr__(target, "_bot_attribute_order", order)
        except Exception:  # pragma: no cover - best effort defaults
            logger.debug(
                "failed to seed _bot_attribute_order before bootstrap on %s",
                target,
                exc_info=True,
            )
        else:
            seeded = True
    if not hasattr(target, "_should_defer_manager_helpers"):
        try:
            object.__setattr__(target, "_should_defer_manager_helpers", False)
        except Exception:
            logger.debug(
                "failed to seed _should_defer_manager_helpers on %s",
                target,
                exc_info=True,
            )
        else:
            seeded = True
    try:
        object.__setattr__(target, "_manager", placeholder)
    except Exception:  # pragma: no cover - direct assignment best effort
        logger.debug(
            "failed to bind bootstrap placeholder directly on %s", target, exc_info=True
        )
    else:
        seeded = True
    if hasattr(target, "initial_manager"):
        try:
            object.__setattr__(target, "initial_manager", placeholder)
        except Exception:  # pragma: no cover - best effort linkage
            logger.debug(
                "failed to seed initial_manager placeholder on %s",
                target,
                exc_info=True,
            )
    logger.debug(
        "pre-seeding manager placeholder state for %s to avoid recursive bootstrap",
        target,
    )
    return seeded


def _assign_bootstrap_manager_placeholder(
    target: Any,
    placeholder: Any,
    *,
    propagate_nested: bool = False,
    _visited: set[int] | None = None,
) -> bool:
    """Assign *placeholder* to ``manager`` and ``initial_manager`` on *target*."""

    if placeholder is None or target is None:
        return False
    assigned_any = False
    propagate_immediately = propagate_nested or _is_bootstrap_placeholder(placeholder)
    for attr in ("manager", "initial_manager"):
        try:
            current = getattr(target, attr)
        except Exception:
            current = None
        if current is placeholder:
            continue
        if current is not None and not _is_bootstrap_placeholder(current):
            continue
        try:
            setattr(target, attr, placeholder)
        except Exception:  # pragma: no cover - best effort assignment
            logger.debug(
                "failed to assign %s placeholder on %s", attr, target, exc_info=True
            )
        else:
            assigned_any = True
    if not assigned_any:
        seeded = _seed_placeholder_bootstrap_fields(target, placeholder)
        if not seeded:
            try:
                object.__setattr__(target, "_should_defer_manager_helpers", False)
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "failed to disable helper deferral after placeholder fallback on %s",
                    target,
                    exc_info=True,
                )
    if propagate_immediately:
        visited = _visited or set()
        visited.add(id(target))
        _propagate_placeholder_to_helpers(target, placeholder, visited)
    return assigned_any


def _seed_existing_pipeline_placeholder(
    pipeline: Any,
    placeholder: Any,
    *,
    allow_disabled_manager: bool = False,
) -> contextvars.Token[Any] | None:
    """Install *placeholder* on *pipeline* and expose it via ``MANAGER_CONTEXT``.

    Existing pipelines may be upgraded after construction, so the sentinel is
    applied again to keep the bootstrap shim and the runtime hierarchy in sync.
    """

    if pipeline is None or placeholder is None:
        return None
    placeholder_is_bootstrap = _is_bootstrap_placeholder(placeholder)
    disabled_manager_allowed = allow_disabled_manager and isinstance(
        placeholder, _DisabledSelfCodingManager
    )
    if not placeholder_is_bootstrap and not disabled_manager_allowed:
        return None
    _seed_placeholder_bootstrap_fields(pipeline, placeholder)
    _assign_bootstrap_manager_placeholder(
        pipeline,
        placeholder,
        propagate_nested=placeholder_is_bootstrap,
    )
    context_manager = _resolve_bootstrap_owner(placeholder) or placeholder
    if context_manager is None:
        return None
    try:
        return MANAGER_CONTEXT.set(context_manager)
    except Exception:  # pragma: no cover - MANAGER_CONTEXT best effort
        logger.debug(
            "failed to expose pipeline manager placeholder via context", exc_info=True
        )
        return None


def _resolve_bootstrap_owner(candidate: Any) -> Any | None:
    """Return the bootstrap owner sentinel associated with *candidate*."""

    if candidate is None:
        return None
    owner = getattr(candidate, "_bootstrap_owner_delegate", None)
    if owner is not None:
        return owner
    if _is_bootstrap_owner(candidate):
        return candidate
    return None


@contextlib.contextmanager
def _inject_manager_placeholder_during_init(
    pipeline_cls: type[Any], placeholder: Any
) -> Iterator[bool]:
    """Temporarily inject *placeholder* before ``pipeline_cls.__init__`` executes.

    The placeholder is installed before ``__init__`` and again afterwards when
    helper promotion walks the hierarchy.  Seeding the internal fields before
    the first assignment prevents recursive bootstrap attempts from helpers
    constructed during ``__init__``.
    """

    if placeholder is None:
        yield False
        return
    try:
        original_init = pipeline_cls.__init__  # type: ignore[attr-defined]
    except AttributeError:
        yield False
        return
    if not callable(original_init):
        yield False
        return

    @wraps(original_init)
    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        local_placeholder = getattr(
            _wrapped_init, "_self_coding_bootstrap_placeholder", placeholder
        )
        token = None
        if local_placeholder is not None:
            _seed_placeholder_bootstrap_fields(self, local_placeholder)
            _assign_bootstrap_manager_placeholder(self, local_placeholder)
            try:
                token = MANAGER_CONTEXT.set(local_placeholder)
            except Exception:  # pragma: no cover - context vars may be unavailable
                token = None
        try:
            return original_init(self, *args, **kwargs)
        finally:
            if token is not None:
                try:
                    MANAGER_CONTEXT.reset(token)
                except Exception:  # pragma: no cover - defensive reset
                    logger.debug("manager context reset failed", exc_info=True)

    setattr(_wrapped_init, "_self_coding_bootstrap_placeholder", placeholder)

    try:
        setattr(pipeline_cls, "__init__", _wrapped_init)
    except Exception:  # pragma: no cover - best effort instrumentation
        logger.debug(
            "failed to patch %s for manager placeholder injection", pipeline_cls,
            exc_info=True,
        )
        yield False
        return
    try:
        yield True
    finally:
        try:
            setattr(pipeline_cls, "__init__", original_init)
        except Exception:  # pragma: no cover - best effort restoration
            logger.debug(
                "failed to restore original __init__ on %s", pipeline_cls, exc_info=True
            )


@dataclass(slots=True)
class _PipelineShimHandle:
    """Tracks shim instrumentation state and its cleanup callback."""

    applied: bool
    release: Callable[[], None] | None = None


@contextlib.contextmanager
def _pipeline_manager_placeholder_shim(
    pipeline_cls: type[Any], placeholder: Any
) -> Iterator[_PipelineShimHandle]:
    """Assign *placeholder* early and expose a consistent bootstrap sentinel."""

    owner_sentinel = _resolve_bootstrap_owner(placeholder)
    release_callbacks: list[Callable[[], None]] = []
    released = False

    def _release_state() -> None:
        nonlocal released
        if released:
            return
        released = True
        while release_callbacks:
            callback = release_callbacks.pop()
            try:
                callback()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("pipeline shim cleanup failed", exc_info=True)

    previous_sentinel = getattr(_BOOTSTRAP_STATE, "sentinel_manager", _SENTINEL_UNSET)
    depth_missing = not hasattr(_BOOTSTRAP_STATE, "depth")
    previous_depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
    depth_overridden = previous_depth <= 0
    sentinel_candidate = owner_sentinel or placeholder
    context_manager = sentinel_candidate
    if context_manager is not None:
        try:
            token = MANAGER_CONTEXT.set(context_manager)
        except Exception:  # pragma: no cover - MANAGER_CONTEXT may be absent
            token = None
        else:
            release_callbacks.append(lambda tok=token: _reset_manager_context(tok))
    if depth_overridden:
        _BOOTSTRAP_STATE.depth = 1
        release_callbacks.append(
            lambda missing=depth_missing, previous=previous_depth: _restore_bootstrap_depth(
                missing, previous
            )
        )
    if sentinel_candidate is not None:
        try:
            _BOOTSTRAP_STATE.sentinel_manager = sentinel_candidate
        except Exception:  # pragma: no cover - best effort override
            logger.debug(
                "failed to seed sentinel manager placeholder during shim", exc_info=True
            )
        else:
            release_callbacks.append(
                lambda previous=previous_sentinel: _restore_bootstrap_sentinel(previous)
            )
    exit_stack = contextlib.ExitStack()
    patched_any = False
    try:
        candidates: list[type[Any]] = []
        if isinstance(pipeline_cls, type):
            candidates.append(pipeline_cls)
            try:
                for base_cls in getattr(pipeline_cls, "__mro__", ()):
                    if not isinstance(base_cls, type):
                        continue
                    if base_cls is pipeline_cls:
                        continue
                    if not _is_model_automation_pipeline_class(base_cls):
                        continue
                    if any(existing is base_cls for existing in candidates):
                        continue
                    candidates.append(base_cls)
            except Exception:  # pragma: no cover - MRO inspection best effort
                logger.debug(
                    "failed to inspect %s MRO for bootstrap instrumentation", pipeline_cls,
                    exc_info=True,
                )
        for candidate_cls in candidates:
            patched = exit_stack.enter_context(
                _inject_manager_placeholder_during_init(candidate_cls, placeholder)
            )
            patched_any = patched_any or bool(patched)
        handle = _PipelineShimHandle(
            applied=patched_any,
            release=_release_state if release_callbacks else None,
        )
        yield handle
    finally:
        exit_stack.close()
        _release_state()


def _reset_manager_context(token: contextvars.Token[Any] | None) -> None:
    """Reset ``MANAGER_CONTEXT`` back to *token* if provided."""

    if token is None:
        return
    try:
        MANAGER_CONTEXT.reset(token)
    except Exception:  # pragma: no cover - defensive cleanup
        logger.debug("manager context reset failed after pipeline shim", exc_info=True)


@contextlib.contextmanager
def _runtime_manager_context(
    manager: Any,
) -> Iterator[contextvars.Token[Any] | None]:
    """Expose *manager* via ``MANAGER_CONTEXT`` for the duration of the block."""

    if manager is None:
        yield None
        return
    token: contextvars.Token[Any] | None
    try:
        token = MANAGER_CONTEXT.set(manager)
    except Exception:  # pragma: no cover - MANAGER_CONTEXT best effort
        logger.debug(
            "failed to expose runtime manager during pipeline bootstrap", exc_info=True
        )
        token = None
    try:
        yield token
    finally:
        _reset_manager_context(token)


def _restore_bootstrap_depth(depth_missing: bool, previous_depth: int) -> None:
    """Restore bootstrap depth bookkeeping after shim finalises."""

    if depth_missing:
        try:
            delattr(_BOOTSTRAP_STATE, "depth")
        except AttributeError:  # pragma: no cover - best effort cleanup
            pass
        return
    _BOOTSTRAP_STATE.depth = previous_depth


def _restore_bootstrap_sentinel(previous_sentinel: Any) -> None:
    """Restore the previously active bootstrap sentinel placeholder."""

    if previous_sentinel is _SENTINEL_UNSET:
        try:
            delattr(_BOOTSTRAP_STATE, "sentinel_manager")
        except AttributeError:  # pragma: no cover - best effort cleanup
            pass
        return
    _BOOTSTRAP_STATE.sentinel_manager = previous_sentinel


def _promote_pipeline_manager(
    pipeline: Any,
    manager: Any,
    sentinel: Any,
    extra_sentinels: Iterable[Any] | None = None,
) -> None:
    """Swap sentinel references for *manager* across the pipeline hierarchy."""

    if pipeline is None or manager is None or manager is sentinel:
        return

    registry_ref = getattr(manager, "bot_registry", None)
    data_bot_ref = getattr(manager, "data_bot", None)
    finalized_targets: set[int] = set()
    sentinel_candidates: list[Any] = []
    if sentinel is not None:
        sentinel_candidates.append(sentinel)
        delegate = getattr(sentinel, "_bootstrap_owner_delegate", None)
        if delegate is not None:
            sentinel_candidates.append(delegate)
    if extra_sentinels:
        for candidate in extra_sentinels:
            if candidate is None:
                continue
            sentinel_candidates.append(candidate)
    sentinel_ids = {id(candidate) for candidate in sentinel_candidates}

    helper_promotions: list[str] = []

    if sentinel is not None:
        logger.debug(
            "promoting bootstrap sentinel %s for pipeline %s", sentinel, pipeline
        )

    def _describe_helper(target: Any) -> str:
        if target is None:
            return "<unknown>"
        for attr in ("name", "bot_name"):
            try:
                value = getattr(target, attr)
            except Exception:
                continue
            if value:
                return str(value)
        try:
            return target.__class__.__name__
        except Exception:
            return repr(target)

    def _record_helper(target: Any) -> None:
        if target is None:
            return
        helper_promotions.append(_describe_helper(target))

    def _refresh_placeholder_reference(placeholder: Any) -> None:
        if placeholder is None or placeholder is manager:
            return
        for attr, replacement in (
            ("manager", manager),
            ("bot_registry", registry_ref),
            ("data_bot", data_bot_ref),
        ):
            if replacement is None:
                continue
            try:
                setattr(placeholder, attr, replacement)
            except Exception:  # pragma: no cover - best effort update
                logger.debug(
                    "failed to refresh %s on bootstrap sentinel %s",
                    attr,
                    placeholder,
                    exc_info=True,
                )

    for placeholder in sentinel_candidates:
        _refresh_placeholder_reference(placeholder)

    def _is_placeholder(value: Any) -> bool:
        if value is None:
            return False
        if id(value) in sentinel_ids:
            return True
        if getattr(value, "bootstrap_runtime_active", False):
            return True
        return _is_bootstrap_owner(value)

    def _invoke_finalizer(target: Any) -> None:
        if target is None:
            return
        key = id(target)
        if key in finalized_targets:
            return
        finalizer = getattr(target, "_finalize_self_coding_bootstrap", None)
        pending_manager = bool(getattr(target, "_self_coding_pending_manager", False))
        if not callable(finalizer):
            return
        kwargs: dict[str, Any] = {}
        if registry_ref is not None:
            kwargs["registry"] = registry_ref
        if data_bot_ref is not None:
            kwargs["data_bot"] = data_bot_ref
        try:
            finalizer(manager, **kwargs)
        except TypeError:
            try:
                finalizer(manager)
            except Exception:  # pragma: no cover - finalizer errors are non-fatal
                logger.debug(
                    "manager finaliser failed for %s", target, exc_info=True
                )
            else:
                finalized_targets.add(key)
        except Exception:  # pragma: no cover - finaliser errors are non-fatal
            logger.debug("manager finaliser failed for %s", target, exc_info=True)
        else:
            finalized_targets.add(key)
        finally:
            if pending_manager:
                try:
                    setattr(target, "_self_coding_pending_manager", False)
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.debug(
                        "failed to reset pending manager flag on %s", target, exc_info=True
                    )

    def _rewire_helper_attributes(target: Any) -> bool:
        manager_rewired = False
        if target is None:
            return manager_rewired
        for attr, replacement in (
            ("manager", manager),
            ("bot_registry", registry_ref),
            ("data_bot", data_bot_ref),
        ):
            if replacement is None:
                continue
            try:
                current = getattr(target, attr)
            except Exception:
                continue
            if not _is_placeholder(current):
                continue
            try:
                setattr(target, attr, replacement)
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "failed to promote %s attribute on %s", attr, target, exc_info=True
                )
            else:
                if attr == "manager":
                    manager_rewired = True
        try:
            initial_candidate = getattr(target, "initial_manager")
        except Exception:
            initial_candidate = None
        if _is_placeholder(initial_candidate):
            try:
                setattr(target, "initial_manager", manager)
            except Exception:  # pragma: no cover - best effort update
                logger.debug(
                    "failed to promote initial_manager on %s", target, exc_info=True
                )
        try:
            finalizer_attr = getattr(target, "_finalize_self_coding_bootstrap")
        except Exception:
            finalizer_attr = None
        if finalizer_attr is None:
            return manager_rewired
        owner = getattr(finalizer_attr, "__self__", None)
        if _is_placeholder(owner) or _is_placeholder(finalizer_attr):
            try:
                delattr(target, "_finalize_self_coding_bootstrap")
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug(
                    "failed to reset _finalize_self_coding_bootstrap on %s",
                    target,
                    exc_info=True,
                )
        return manager_rewired

    def _swap_manager(target: Any) -> None:
        nonlocal finalized_targets
        if target is None:
            return
        finalizer_targets: list[Any] = []
        manager_rewired = _rewire_helper_attributes(target)
        try:
            current_manager = getattr(target, "manager", None)
        except Exception:  # pragma: no cover - best effort
            current_manager = None
        if _is_placeholder(current_manager):
            try:
                setattr(target, "manager", manager)
            except Exception:  # pragma: no cover - best effort reassignment
                logger.debug("failed to promote manager on %s", target, exc_info=True)
            else:
                finalizer_targets.append(target)
                _record_helper(target)
        elif manager_rewired:
            finalizer_targets.append(target)
            _record_helper(target)
        owner = getattr(target, "__class__", None)
        if owner is not None:
            try:
                owner_current = getattr(owner, "manager", None)
            except Exception:  # pragma: no cover - class level guard
                logger.debug(
                    "manager promotion inspection failed for %s class", owner, exc_info=True
                )
            else:
                if _is_placeholder(owner_current):
                    try:
                        setattr(owner, "manager", manager)
                    except Exception:  # pragma: no cover - class level guard
                        logger.debug(
                            "failed to promote manager on %s class", owner, exc_info=True
                        )
                    else:
                        finalizer_targets.append(owner)
                        _record_helper(owner)
        if getattr(target, "_self_coding_pending_manager", False):
            finalizer_targets.append(target)
        if owner is not None and getattr(owner, "_self_coding_pending_manager", False):
            finalizer_targets.append(owner)
        for final_target in finalizer_targets:
            _invoke_finalizer(final_target)

    _swap_manager(pipeline)

    seen: set[int] = {id(pipeline)}
    pending: deque[Any] = deque()

    def _enqueue_helpers(root: Any) -> None:
        if root is None:
            return
        for candidate in _iter_bootstrap_helper_candidates(root):
            if candidate is None:
                continue
            key = id(candidate)
            if key in seen or key in sentinel_ids:
                continue
            if isinstance(candidate, (str, bytes, bytearray, int, float, complex, bool)):
                continue
            seen.add(key)
            pending.append(candidate)

    _enqueue_helpers(pipeline)

    while pending:
        candidate = pending.popleft()
        _swap_manager(candidate)
        _invoke_finalizer(candidate)
        _enqueue_helpers(candidate)

    def _promote_registry_metadata() -> None:
        if registry_ref is None:
            return
        graph = getattr(registry_ref, "graph", None)
        if graph is None:
            return
        nodes = getattr(graph, "nodes", None)
        if nodes is None:
            return
        entries: list[tuple[Any, Any]] = []
        items = getattr(nodes, "items", None)
        if callable(items):
            try:
                entries = list(items())
            except Exception:  # pragma: no cover - best effort view coercion
                entries = []
        if not entries:
            try:
                keys = list(nodes)
            except Exception:  # pragma: no cover - view iteration may fail
                keys = []
            for key in keys:
                try:
                    entries.append((key, nodes[key]))
                except Exception:  # pragma: no cover - best effort access
                    logger.debug(
                        "failed to inspect registry node %s", key, exc_info=True
                    )
        for _name, payload in entries:
            if isinstance(payload, Mapping):
                for attr in ("selfcoding_manager", "manager"):
                    current = payload.get(attr)
                    if not _is_placeholder(current):
                        continue
                    try:
                        payload[attr] = manager
                    except Exception:  # pragma: no cover - registry mutation best effort
                        logger.debug(
                            "failed to promote %s on registry entry %s", attr, _name,
                            exc_info=True,
                        )
                continue
            for attr in ("selfcoding_manager", "manager"):
                try:
                    current = getattr(payload, attr)
                except Exception:
                    continue
                if not _is_placeholder(current):
                    continue
                try:
                    setattr(payload, attr, manager)
                except Exception:  # pragma: no cover - best effort
                    logger.debug(
                        "failed to promote %s on registry payload %s", attr, payload,
                        exc_info=True,
                    )

    _promote_registry_metadata()

    if sentinel is not None:
        helper_summary = ", ".join(dict.fromkeys(helper_promotions))
        logger.debug(
            "bootstrap sentinel promotion completed for %s (helpers=%s)",
            pipeline,
            helper_summary or "<none>",
        )

    finalize_helpers = getattr(pipeline, "finalize_helpers", None)
    if callable(finalize_helpers):
        try:
            finalize_helpers(manager)
        except Exception:  # pragma: no cover - finalizer failures are non-fatal
            logger.debug("pipeline finalize_helpers hook failed", exc_info=True)

    # Allow downstream consumers to refresh their cached manager reference when
    # promotion succeeds without importing heavy modules eagerly.
    refresh = getattr(pipeline, "_attach_information_synthesis_manager", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:  # pragma: no cover - downstream refresh should not block
            logger.debug("pipeline reattachment hook failed", exc_info=True)


def prepare_pipeline_for_bootstrap(
    *,
    pipeline_cls: type[Any],
    context_builder: Any,
    bot_registry: Any,
    data_bot: Any,
    bootstrap_manager: Any | None = None,
    bootstrap_runtime_manager: Any | None = None,
    force_manager_kwarg: bool = False,
    manager_override: Any | None = None,
    manager_sentinel: Any | None = None,
    sentinel_factory: Callable[[], Any] | None = None,
    extra_manager_sentinels: Iterable[Any] | None = None,
    **pipeline_kwargs: Any,
) -> tuple[Any, Callable[[Any], None]]:
    """Instantiate *pipeline_cls* with a bootstrap sentinel manager.

    See ``_prepare_pipeline_for_bootstrap_impl`` for full parameter details.
    """

    try:
        from menace_sandbox.relevancy_radar import relevancy_import_hook_guard
    except Exception:  # pragma: no cover - flat layout fallback
        try:
            from relevancy_radar import relevancy_import_hook_guard  # type: ignore
        except Exception:  # pragma: no cover - relevancy radar unavailable
            @contextlib.contextmanager
            def relevancy_import_hook_guard():
                yield

    with relevancy_import_hook_guard():
        return _prepare_pipeline_for_bootstrap_impl(
            pipeline_cls=pipeline_cls,
            context_builder=context_builder,
            bot_registry=bot_registry,
            data_bot=data_bot,
            bootstrap_manager=bootstrap_manager,
            bootstrap_runtime_manager=bootstrap_runtime_manager,
            force_manager_kwarg=force_manager_kwarg,
            manager_override=manager_override,
            manager_sentinel=manager_sentinel,
            sentinel_factory=sentinel_factory,
            extra_manager_sentinels=extra_manager_sentinels,
            **pipeline_kwargs,
        )


def _prepare_pipeline_for_bootstrap_impl(
    *,
    pipeline_cls: type[Any],
    context_builder: Any,
    bot_registry: Any,
    data_bot: Any,
    bootstrap_manager: Any | None = None,
    bootstrap_runtime_manager: Any | None = None,
    force_manager_kwarg: bool = False,
    manager_override: Any | None = None,
    manager_sentinel: Any | None = None,
    sentinel_factory: Callable[[], Any] | None = None,
    extra_manager_sentinels: Iterable[Any] | None = None,
    **pipeline_kwargs: Any,
) -> tuple[Any, Callable[[Any], None]]:
    """Instantiate *pipeline_cls* with a bootstrap sentinel manager.

    The returned ``ModelAutomationPipeline`` instance uses a temporary sentinel
    manager so callers can safely construct nested bots while the real
    ``SelfCodingManager`` is still initialising.  The accompanying callback must
    be invoked once the concrete manager is ready so that all sentinel
    references are promoted atomically.
    ``bootstrap_manager`` ensures the supplied sentinel is treated as the active
    bootstrap manager for the duration of the pipeline construction even when a
    different context already holds the guard.  ``bootstrap_runtime_manager``
    exposes a temporary concrete manager to nested helpers while the pipeline is
    initialising so attribute lookups never encounter ``None`` even though the
    sentinel placeholders remain authoritative.  ``force_manager_kwarg`` keeps
    the ``manager`` keyword argument supplied via ``pipeline_kwargs`` intact so
    constructors can observe the provided runtime manager even when bootstrap
    sentinels are in play.  ``manager_override`` allows callers to reuse an
    existing sentinel instead of creating a new one when
    nested helper bootstrap is already in progress.
    ``manager_sentinel`` forces a specific placeholder to be passed to the
    pipeline constructor even when a different sentinel guards the bootstrap
    context so callers can ensure downstream helpers see a consistent manager
    reference.  ``sentinel_factory`` can be used to lazily construct a sentinel
    so callers can track when bootstrap contexts adopt it.  When helpers attach
    temporary managers that should be promoted alongside the bootstrap sentinel,
    include them via ``extra_manager_sentinels`` so promotion callbacks replace
    every placeholder atomically.  The runtime manager is only used during
    construction; once the pipeline instance exists the sentinel bookkeeping is
    reapplied so promotion callbacks behave as before.  Callers that need a
    specific temporary manager can supply it via ``bootstrap_runtime_manager``.
    Additional keyword arguments are forwarded to ``pipeline_cls`` during
    instantiation.
    """

    def _typeerror_rejects_manager(exc: TypeError) -> bool:
        message = str(exc)
        if not message:
            return False
        lowered = message.lower()
        return "manager" in lowered and "unexpected" in lowered

    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    slow_hook_threshold = 0.05

    def _log_timing(label: str, start_time: float, **payload: Any) -> None:
        if not debug_enabled:
            return
        elapsed = time.perf_counter() - start_time
        details = ", ".join(
            f"{key}={value}" for key, value in payload.items() if value is not None
        )
        logger.debug(
            "bootstrap %s elapsed=%.3fs%s",
            label,
            elapsed,
            f" ({details})" if details else "",
        )

    def _log_slow_hook(label: str, start_time: float, **payload: Any) -> None:
        if not debug_enabled:
            return
        elapsed = time.perf_counter() - start_time
        if elapsed < slow_hook_threshold:
            return
        details = ", ".join(
            f"{key}={value}" for key, value in payload.items() if value is not None
        )
        logger.debug("bootstrap %s took %.3fs%s", label, elapsed, f" ({details})" if details else "")

    def _select_placeholder(*candidates: Any) -> Any | None:
        for candidate in candidates:
            if _is_bootstrap_placeholder(candidate):
                return candidate
        return None

    def _concrete_runtime(candidate: Any | None) -> Any | None:
        if candidate is None:
            return None
        if _is_bootstrap_placeholder(candidate):
            return None
        return candidate

    manager_kwarg_value = pipeline_kwargs.get("manager")
    manager_kwarg_placeholder = manager_kwarg_value
    sentinel_selection_start = time.perf_counter()
    sentinel_manager = _select_placeholder(
        manager_override,
        manager_sentinel,
        bootstrap_manager,
        manager_kwarg_placeholder,
    )
    if sentinel_manager is None and sentinel_factory is not None:
        candidate = sentinel_factory()
        if _is_bootstrap_placeholder(candidate):
            sentinel_manager = candidate
    if sentinel_manager is None:
        sentinel_manager = _create_bootstrap_manager_sentinel(
            bot_registry=bot_registry,
            data_bot=data_bot,
        )
    _log_timing(
        "sentinel selection",
        sentinel_selection_start,
        manager_kwarg_supplied=manager_kwarg_value is not None,
        manager_override=bool(manager_override),
        sentinel_factory=bool(sentinel_factory),
    )
    manager_placeholder = sentinel_manager
    shim_manager_placeholder = _select_placeholder(
        manager_override,
        manager_sentinel,
        bootstrap_manager,
        manager_placeholder,
    )
    if not _is_bootstrap_placeholder(shim_manager_placeholder):
        owner_candidate = _spawn_nested_bootstrap_owner(
            bot_registry=bot_registry,
            data_bot=data_bot,
        )
        if _is_bootstrap_placeholder(owner_candidate):
            shim_manager_placeholder = owner_candidate
        else:
            shim_manager_placeholder = manager_placeholder
    extra_candidates: list[Any] = []
    if extra_manager_sentinels:
        for candidate in extra_manager_sentinels:
            if candidate is None:
                continue
            extra_candidates.append(candidate)
    if (
        shim_manager_placeholder is not None
        and shim_manager_placeholder is not manager_placeholder
    ):
        extra_candidates.append(shim_manager_placeholder)
    managed_extra_manager_sentinels: tuple[Any, ...] | None = None
    sentinel_activation_start = time.perf_counter()
    restore_sentinel_state = _activate_bootstrap_sentinel(sentinel_manager)
    _log_timing(
        "activate bootstrap sentinel",
        sentinel_activation_start,
        placeholder_attached=_is_bootstrap_placeholder(sentinel_manager),
    )
    owner_guard_start = time.perf_counter()
    (
        owner_guard_attached,
        release_owner_guard,
    ) = _claim_bootstrap_owner_guard(sentinel_manager, restore_sentinel_state)
    _log_timing(
        "claim bootstrap owner guard",
        owner_guard_start,
        owner_guard_attached=owner_guard_attached,
    )
    manual_restore_pending = not owner_guard_attached
    sentinel_state_released = False
    shim_release_callback: Callable[[], None] | None = None

    def _finalize_bootstrap_state(*, due_to_failure: bool = False) -> None:
        nonlocal manual_restore_pending, owner_guard_attached, sentinel_state_released
        if sentinel_state_released:
            return
        if due_to_failure and owner_guard_attached:
            released = False
            if callable(release_owner_guard):
                try:
                    release_owner_guard()
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.debug(
                        "bootstrap owner sentinel guard release failed", exc_info=True
                    )
                else:
                    released = True
            owner_guard_attached = released
        if manual_restore_pending or not owner_guard_attached:
            restore_sentinel_state()
            manual_restore_pending = False
            owner_guard_attached = False
            sentinel_state_released = True
    context: _BootstrapContext | None = None
    pipeline: Any | None = None
    last_error: Exception | None = None
    runtime_manager = bootstrap_runtime_manager
    manager_kwarg_forced = bool(force_manager_kwarg and manager_kwarg_value is not None)
    if manager_kwarg_forced:
        runtime_manager = manager_kwarg_value
    if _is_bootstrap_placeholder(runtime_manager):
        runtime_manager = None
    if runtime_manager is None:
        runtime_manager = _concrete_runtime(manager_kwarg_value)
    if runtime_manager is None:
        runtime_manager = _concrete_runtime(bootstrap_manager)
    if runtime_manager is None:
        runtime_manager = _DisabledSelfCodingManager(
            bot_registry=bot_registry,
            data_bot=data_bot,
            bootstrap_placeholder=True,
            bootstrap_runtime=True,
        )
    runtime_manager_placeholder: Any | None = None
    if runtime_manager is not None:
        if getattr(runtime_manager, "bootstrap_runtime_active", False):
            runtime_manager_placeholder = runtime_manager
        elif _is_bootstrap_placeholder(runtime_manager):
            runtime_manager_placeholder = runtime_manager
    if runtime_manager_placeholder is not None:
        extra_candidates.append(runtime_manager_placeholder)
    if extra_candidates:
        managed_extra_manager_sentinels = tuple(dict.fromkeys(extra_candidates))

    try:
        context_start = time.perf_counter()
        context = _push_bootstrap_context(
            registry=bot_registry,
            data_bot=data_bot,
            manager=runtime_manager,
        )
        _log_slow_hook(
            "context push",
            context_start,
            registry_present=bool(bot_registry),
            data_bot_present=bool(data_bot),
        )
        _log_timing(
            "context push",
            context_start,
            registry_present=bool(bot_registry),
            data_bot_present=bool(data_bot),
        )
        if context is not None and sentinel_manager is not None:
            context.sentinel = sentinel_manager
        init_kwargs = dict(pipeline_kwargs)
        init_kwargs.setdefault("context_builder", context_builder)
        if bot_registry is not None:
            init_kwargs.setdefault("bot_registry", bot_registry)
        if data_bot is not None:
            init_kwargs.setdefault("data_bot", data_bot)
        current_manager_value = init_kwargs.get("manager")
        manager_kwarg_supplied = "manager" in pipeline_kwargs
        should_override_manager = (
            "manager" not in init_kwargs
            or current_manager_value is None
            or (
                not (manager_kwarg_forced and manager_kwarg_supplied)
                and _is_bootstrap_placeholder(current_manager_value)
            )
        )
        if should_override_manager:
            init_kwargs["manager"] = runtime_manager

        variants: tuple[tuple[str, ...], ...] = (
            ("context_builder", "bot_registry", "data_bot", "manager"),
            ("context_builder", "bot_registry", "manager"),
            ("context_builder", "manager"),
            ("context_builder",),
        )
        static_items = {
            key: value
            for key, value in init_kwargs.items()
            if key
            not in {"context_builder", "bot_registry", "data_bot", "manager"}
        }
        available_variants: tuple[tuple[str, ...], ...] = tuple(
            keys for keys in variants if all(key in init_kwargs for key in keys)
        )
        manager_variants = tuple(keys for keys in available_variants if "manager" in keys)
        managerless_variants = tuple(
            keys for keys in available_variants if "manager" not in keys
        )

        manager_rejected = False
        constructor_loop_start = time.perf_counter()

        def _attempt_constructor(keys: tuple[str, ...]) -> bool:
            nonlocal pipeline, last_error, manager_rejected
            call_kwargs = {key: init_kwargs[key] for key in keys}
            call_kwargs.update(static_items)
            start_time = time.perf_counter()
            try:
                pipeline = pipeline_cls(**call_kwargs)
            except TypeError as exc:
                last_error = exc
                if (
                    not manager_rejected
                    and "manager" in keys
                    and _typeerror_rejects_manager(exc)
                ):
                    manager_rejected = True
                _log_slow_hook(
                    "pipeline constructor failure",
                    start_time,
                    keys=keys,
                    manager_rejected=manager_rejected,
                )
                _log_timing(
                    "pipeline constructor failure",
                    start_time,
                    keys=keys,
                    manager_rejected=manager_rejected,
                )
                return False
            _log_slow_hook("pipeline construction", start_time, keys=keys)
            _log_timing("pipeline construction", start_time, keys=keys)
            return True

        shim_context: contextlib.AbstractContextManager[Any]
        if shim_manager_placeholder is None:
            shim_context = contextlib.nullcontext(_PipelineShimHandle(False, None))
        else:
            shim_context = _pipeline_manager_placeholder_shim(
                pipeline_cls, shim_manager_placeholder
            )
        shim_release_candidate: Callable[[], None] | None = None

        def _runtime_context_manager() -> contextlib.AbstractContextManager[
            contextvars.Token[Any] | None
        ]:
            if runtime_manager is None:
                return contextlib.nullcontext(None)
            return _runtime_manager_context(runtime_manager)

        with shim_context as shim_handle:
            if isinstance(shim_handle, _PipelineShimHandle):
                shim_release_candidate = shim_handle.release
            else:
                candidate_release = getattr(shim_handle, "release", None)
                if callable(candidate_release):
                    shim_release_candidate = candidate_release
            with _runtime_context_manager():
                for keys in manager_variants:
                    if _attempt_constructor(keys):
                        break
        _log_timing(
            "constructor attempt loop",
            constructor_loop_start,
            attempts=len(manager_variants),
            manager_rejected=manager_rejected,
            shim_placeholder=bool(shim_manager_placeholder),
        )
        managerless_placeholder = None
        if pipeline is None:
            if callable(shim_release_candidate):
                try:
                    shim_release_candidate()
                except Exception:  # pragma: no cover - cleanup best effort
                    logger.debug(
                        "failed to release pipeline shim placeholder", exc_info=True
                    )
            shim_release_candidate = None
        if pipeline is None and manager_rejected and managerless_variants:
            managerless_placeholder = _select_placeholder(
                shim_manager_placeholder,
                manager_placeholder,
                sentinel_manager,
            )
            if managerless_placeholder is None:
                managerless_placeholder = runtime_manager
            if managerless_placeholder is None:
                managerless_placeholder = _DisabledSelfCodingManager(
                    bot_registry=bot_registry,
                    data_bot=data_bot,
                    bootstrap_placeholder=True,
                )
            _mark_bootstrap_placeholder(managerless_placeholder)
            logger.debug(
                "retrying %s bootstrap without manager kwarg using placeholder %s",
                pipeline_cls,
                type(managerless_placeholder),
            )
            managerless_context: contextlib.AbstractContextManager[Any]
            managerless_context = _pipeline_manager_placeholder_shim(
                pipeline_cls, managerless_placeholder
            )
            managerless_loop_start = time.perf_counter()
            with managerless_context as shim_handle:
                if isinstance(shim_handle, _PipelineShimHandle):
                    shim_release_candidate = shim_handle.release
                else:
                    candidate_release = getattr(shim_handle, "release", None)
                    if callable(candidate_release):
                        shim_release_candidate = candidate_release
                with _runtime_context_manager():
                    for keys in managerless_variants:
                        if _attempt_constructor(keys):
                            shim_manager_placeholder = managerless_placeholder
                            break
            _log_timing(
                "managerless constructor loop",
                managerless_loop_start,
                attempts=len(managerless_variants),
                managerless_placeholder=bool(managerless_placeholder),
            )
        if pipeline is None:
            if callable(shim_release_candidate):
                try:
                    shim_release_candidate()
                except Exception:  # pragma: no cover - cleanup best effort
                    logger.debug(
                        "failed to release pipeline shim placeholder", exc_info=True
                    )
            if last_error is None:
                raise RuntimeError("pipeline bootstrap failed")
            raise last_error
        if callable(shim_release_candidate):
            shim_release_callback = shim_release_candidate
        _assign_bootstrap_manager_placeholder(
            pipeline,
            manager_placeholder,
            propagate_nested=True,
        )
        _log_timing(
            "assign bootstrap manager placeholder",
            constructor_loop_start,
            propagate_nested=True,
            has_context=bool(context),
        )
        if context is not None and context.pipeline is None:
            context.pipeline = pipeline
    finally:
        if context is not None:
            _pop_bootstrap_context(context)
        if pipeline is None:
            _finalize_bootstrap_state(due_to_failure=True)

    try:
        current_manager = getattr(pipeline, "manager", None)
    except Exception:  # pragma: no cover - introspection guard
        current_manager = None
    if current_manager is not sentinel_manager:
        try:
            setattr(pipeline, "manager", sentinel_manager)
        except Exception:  # pragma: no cover - best effort assignment
            logger.debug(
                "pipeline manager sentinel attachment failed for %s", pipeline, exc_info=True
            )

    def _promote(
        manager: Any,
        *,
        extra_sentinels: Iterable[Any] | None = None,
    ) -> None:
        nonlocal shim_release_callback
        promote_start = time.perf_counter()
        extra_sentinel_candidates = tuple(extra_sentinels or ())
        aggregation_start = time.perf_counter()
        try:
            combined: list[Any] = []
            seen: set[int] = set()
            sentinel_groups: tuple[Iterable[Any] | None, ...] = (
                managed_extra_manager_sentinels,
                extra_sentinel_candidates,
            )
            for group in sentinel_groups:
                if not group:
                    continue
                for candidate in group:
                    if candidate is None:
                        continue
                    key = id(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    combined.append(candidate)
            placeholder_extras: list[Any] = []
            if (
                shim_manager_placeholder is not None
                and shim_manager_placeholder is not manager_placeholder
            ):
                placeholder_extras.append(shim_manager_placeholder)
            if combined:
                placeholder_extras.extend(combined)
            _log_timing(
                "promotion aggregation",
                aggregation_start,
                managed_extra=len(managed_extra_manager_sentinels or ()),
                runtime_extra=len(extra_sentinel_candidates),
            )
            promotion_apply_start = time.perf_counter()
            _promote_pipeline_manager(
                pipeline,
                manager,
                manager_placeholder,
                extra_sentinels=placeholder_extras or None,
            )
            _log_timing(
                "pipeline promotion application",
                promotion_apply_start,
                placeholder_extras=len(placeholder_extras),
            )
        finally:
            if callable(shim_release_callback):
                try:
                    shim_release_callback()
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.debug("pipeline shim release failed", exc_info=True)
                shim_release_callback = None
            _deregister_bootstrap_helper_callback(_promote)
            _finalize_bootstrap_state()
            _log_slow_hook(
                "sentinel promotion",
                promote_start,
                extra_sentinels=len(extra_sentinel_candidates),
            )

    register_callback = any(
        _is_bootstrap_placeholder(candidate)
        for candidate in (
            manager_placeholder,
            bootstrap_manager,
            manager_override,
            sentinel_manager,
            shim_manager_placeholder,
            runtime_manager_placeholder,
        )
    )
    if register_callback:
        _register_bootstrap_helper_callback(_promote)

    return pipeline, _promote


def _bootstrap_manager(
    name: str,
    bot_registry: BotRegistry,
    data_bot: DataBot,
    *,
    pipeline: Any | None = None,
    pipeline_manager: Any | None = None,
    pipeline_promoter: Callable[[Any], None] | None = None,
    disabled_manager_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    """Instantiate a ``SelfCodingManager`` with progressive fallbacks."""

    def _disabled_manager(reason: str, *, reentrant: bool = False) -> Any:
        placeholder: Any | None = None
        sentinel_reused = False
        if reentrant:
            placeholder = getattr(_BOOTSTRAP_STATE, "sentinel_manager", None)
            if placeholder is None:
                placeholder = sentinel_manager
            sentinel_reused = placeholder is not None
            guidance = (
                "Re-entrant bootstrap detected; reusing bootstrap sentinel "
                "temporarily—internalisation will retry after "
                f"{name} completes."
            )
        else:
            guidance = (
                "Bootstrap already active; returning disabled manager "
                "until the current owner finishes."
            )
        depth_level = getattr(_BOOTSTRAP_STATE, "depth", 0)
        message = (
            f"SelfCodingManager bootstrap skipped for {name}: {reason}. {guidance}"
        )
        payload = {
            "owner": name,
            "reason": reason,
            "bootstrap_depth": depth_level,
            "reentrant": bool(reentrant),
            "sentinel_reused": sentinel_reused,
            "message": message,
        }
        print(message)
        logger.warning(message, extra={"disabled_manager": payload})
        _emit_disabled_manager_metric(payload)
        if callable(disabled_manager_callback):
            try:
                disabled_manager_callback(payload)
            except Exception:  # pragma: no cover - callback best effort
                logger.debug("disabled manager callback failed", exc_info=True)
        if reentrant and placeholder is not None:
            _track_extra_sentinel(placeholder)
            return placeholder
        disabled_manager = _DisabledSelfCodingManager(
            bot_registry=bot_registry,
            data_bot=data_bot,
            bootstrap_placeholder=True,
        )
        _track_extra_sentinel(disabled_manager)
        return disabled_manager

    def _resolve_owner_token(candidate: Any) -> Any | None:
        if candidate in (None, _SENTINEL_UNSET):
            return None
        if isinstance(candidate, str):
            return candidate
        try:
            return getattr(candidate, "_bootstrap_owner_token", None)
        except Exception:  # pragma: no cover - attribute introspection best effort
            return None

    depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
    sentinel_active = _is_bootstrap_placeholder(
        getattr(_BOOTSTRAP_STATE, "sentinel_manager", None)
    )
    if depth > 0 and not sentinel_active:
        logger.debug(
            "restoring missing bootstrap sentinel for %s (depth=%s)", name, depth
        )
        sentinel_active = True

    pipeline = _resolve_bootstrap_pipeline_candidate(pipeline)
    if pipeline_manager is None and pipeline is not None:
        try:
            pipeline_manager = getattr(pipeline, "manager", None)
        except Exception:
            pipeline_manager = None

    sentinel_manager = _create_bootstrap_manager_sentinel(
        bot_registry=bot_registry,
        data_bot=data_bot,
    )
    try:
        sentinel_manager._bootstrap_owner_token = name
    except Exception:  # pragma: no cover - attribute assignment best effort
        logger.debug("failed to assign bootstrap owner token", exc_info=True)
    previous_sentinel = getattr(_BOOTSTRAP_STATE, "sentinel_manager", _SENTINEL_UNSET)
    _BOOTSTRAP_STATE.sentinel_manager = sentinel_manager
    fallback_manager_sentinels: list[Any] = []

    def _resolve_fallback_owner(owner_token: Any) -> Any | None:
        if owner_token in (None, _SENTINEL_UNSET):
            return None
        stack = getattr(_BOOTSTRAP_STATE, "fallback_owner_stack", None)
        if not stack:
            return None
        for candidate in reversed(stack):
            try:
                delegate = getattr(candidate, "_bootstrap_owner_delegate", None)
            except Exception:  # pragma: no cover - best effort attribute access
                delegate = None
            if delegate is owner_token:
                return candidate
        return None

    def _seed_comm_bot_bootstrap_override(
        runtime_manager: Any | None,
    ) -> Callable[[], None] | None:
        if runtime_manager is None:
            return None
        module = sys.modules.get(
            "menace_sandbox.communication_maintenance_bot"
        ) or sys.modules.get("communication_maintenance_bot")
        if module is None:
            return None
        setter = getattr(module, "set_bootstrap_manager_for_comm_bot", None)
        if not callable(setter):
            return None
        try:
            setter(
                runtime_manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
        except Exception:  # pragma: no cover - hook should stay best effort
            logger.debug(
                "failed to seed communication maintenance bootstrap override",
                exc_info=True,
            )
            return None

        def _reset() -> None:
            try:
                setter(None)
            except Exception:  # pragma: no cover - hook cleanup best effort
                logger.debug(
                    "failed to clear communication maintenance bootstrap override",
                    exc_info=True,
                )

        return _reset

    def _push_fallback_owner(owner: Any) -> bool:
        if not _is_bootstrap_owner(owner):
            return False
        stack = getattr(_BOOTSTRAP_STATE, "fallback_owner_stack", None)
        if stack is None:
            stack = []
            _BOOTSTRAP_STATE.fallback_owner_stack = stack
        stack.append(owner)
        return True

    def _pop_fallback_owner(owner: Any) -> None:
        stack = getattr(_BOOTSTRAP_STATE, "fallback_owner_stack", None)
        if not stack:
            return
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is owner:
                stack.pop(index)
                break
        if not stack and hasattr(_BOOTSTRAP_STATE, "fallback_owner_stack"):
            delattr(_BOOTSTRAP_STATE, "fallback_owner_stack")

    def _track_extra_sentinel(candidate: Any) -> None:
        if candidate is None:
            return
        if not _is_bootstrap_placeholder(candidate):
            return
        if any(existing is candidate for existing in fallback_manager_sentinels):
            return
        fallback_manager_sentinels.append(candidate)

    if pipeline_manager is not None:
        _track_extra_sentinel(pipeline_manager)
    next_depth = depth + 1
    logger.debug(
        "installed bootstrap sentinel %s for %s (depth=%s)",
        sentinel_manager,
        name,
        next_depth,
    )

    owner_guard = previous_sentinel
    if owner_guard is _SENTINEL_UNSET:
        owner_guard = name
    owns_reentrant_promise = False
    reentrant_promises_settled = False

    def _settle_reentrant_promises(manager_value: Any | None) -> None:
        nonlocal reentrant_promises_settled
        if not owns_reentrant_promise or reentrant_promises_settled:
            return
        _settle_owner_promise(owner_guard, manager_value)
        reentrant_promises_settled = True
    owner_depths = getattr(_BOOTSTRAP_STATE, "owner_depths", None)
    if owner_depths is None:
        owner_depths = {}
        _BOOTSTRAP_STATE.owner_depths = owner_depths
    current_owner_depth = owner_depths.get(owner_guard, 0)
    guard_token = _resolve_owner_token(owner_guard)
    guard_matches_owner = guard_token == name and _is_bootstrap_placeholder(owner_guard)
    if current_owner_depth > 0:
        if previous_sentinel is _SENTINEL_UNSET:
            try:
                delattr(_BOOTSTRAP_STATE, "sentinel_manager")
            except AttributeError:
                pass
        else:
            _BOOTSTRAP_STATE.sentinel_manager = previous_sentinel
        if guard_matches_owner:
            logger.info(
                "reusing bootstrap sentinel for %s (owner=%s depth=%s)",
                name,
                guard_token,
                current_owner_depth,
            )
            _track_extra_sentinel(owner_guard)
            return owner_guard
        fallback_owner = None
        if owner_guard is not name:
            fallback_owner = _resolve_fallback_owner(owner_guard)
        if fallback_owner is not None:
            _track_extra_sentinel(fallback_owner)
            return fallback_owner
        promise = _peek_owner_promise(owner_guard)
        placeholder_active = _is_bootstrap_placeholder(previous_sentinel)
        if (
            promise is not None
            and placeholder_active
            and promise.add_waiter(sentinel_manager)
        ):
            _track_extra_sentinel(sentinel_manager)
            return sentinel_manager
        return _disabled_manager("bootstrap already in progress", reentrant=True)

    owner_depths[owner_guard] = current_owner_depth + 1
    _BOOTSTRAP_STATE.depth = next_depth

    def _build_reentrant_placeholder() -> Any:
        return _DisabledSelfCodingManager(
            bot_registry=bot_registry,
            data_bot=data_bot,
            bootstrap_placeholder=True,
        )

    owner_promise = _ensure_owner_promise(
        owner_guard,
        fallback_factory=_build_reentrant_placeholder,
    )
    owns_reentrant_promise = owner_promise is not None

    def _release_guard() -> None:
        owner_depths_local = getattr(_BOOTSTRAP_STATE, "owner_depths", None)
        if not owner_depths_local:
            return
        depth_value = owner_depths_local.get(owner_guard, 0)
        if depth_value <= 1:
            owner_depths_local.pop(owner_guard, None)
        else:
            owner_depths_local[owner_guard] = depth_value - 1
        if owner_depths_local:
            _BOOTSTRAP_STATE.owner_depths = owner_depths_local
        elif hasattr(_BOOTSTRAP_STATE, "owner_depths"):
            delattr(_BOOTSTRAP_STATE, "owner_depths")

    bootstrap_owner: _BootstrapOwnerSentinel | None = None
    owner_context: _BootstrapContext | None = None
    placeholder_manager: Any = sentinel_manager

    def _link_bootstrap_owner(candidate: Any) -> None:
        if bootstrap_owner is None or candidate is None:
            return
        if candidate is bootstrap_owner:
            return
        if not _is_bootstrap_placeholder(candidate):
            return
        sentinel_candidate: _BootstrapManagerSentinel | None = None
        if isinstance(candidate, _BootstrapManagerSentinel):
            sentinel_candidate = candidate
        _assign_bootstrap_manager_placeholder(
            candidate,
            bootstrap_owner,
            propagate_nested=True,
        )
        if sentinel_candidate is not None:
            attach_delegate = getattr(sentinel_candidate, "attach_delegate", None)
            if callable(attach_delegate):
                try:
                    attach_delegate(bootstrap_owner)
                except Exception:  # pragma: no cover - best effort linkage
                    logger.debug(
                        "failed to rebind bootstrap sentinel %s to owner", sentinel_candidate,
                        exc_info=True,
                    )
        try:
            setattr(candidate, "_bootstrap_owner_delegate", bootstrap_owner)
        except Exception:  # pragma: no cover - best effort linkage
            logger.debug(
                "failed to link bootstrap delegate %s to owner sentinel", candidate,
                exc_info=True,
            )
        try:
            setattr(bootstrap_owner, "_bootstrap_owner_delegate", candidate)
        except Exception:  # pragma: no cover - best effort linkage
            logger.debug(
                "failed to link bootstrap owner sentinel to %s", candidate,
                exc_info=True,
            )

    try:
        manager_cls = _resolve_self_coding_manager_cls()
        if manager_cls is None:
            raise RuntimeError("self-coding runtime is unavailable")

        context: _BootstrapContext | None = None
        manager: Any = None
        try:
            context = _push_bootstrap_context(
                registry=bot_registry,
                data_bot=data_bot,
                manager=sentinel_manager,
                pipeline=pipeline,
            )
            manager = manager_cls(  # type: ignore[call-arg]
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
        except TypeError:
            manager = None
        except Exception as exc:
            logger.debug(
                "self-coding manager bootstrap via %s failed for %s",
                getattr(manager_cls, "__name__", manager_cls),
                name,
                exc_info=exc,
            )
            raise
        else:
            _promote_pipeline_manager(
                getattr(manager, "pipeline", None),
                manager,
                sentinel_manager,
            )
            _settle_reentrant_promises(manager)
            return manager
        finally:
            if context is not None:
                _pop_bootstrap_context(context)

        try:
            bootstrap_owner = _BootstrapOwnerSentinel(
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
        except Exception as exc:  # pragma: no cover - sentinel creation must succeed
            bootstrap_owner = None
            logger.warning(
                "failed to construct bootstrap owner sentinel for %s; falling back to disabled placeholder",
                name,
                exc_info=exc,
            )
        if bootstrap_owner is None:
            placeholder_manager = _DisabledSelfCodingManager(
                bot_registry=bot_registry,
                data_bot=data_bot,
                bootstrap_owner=True,
                bootstrap_placeholder=True,
            )
            bootstrap_owner = placeholder_manager
        else:
            placeholder_manager = bootstrap_owner
        _track_extra_sentinel(placeholder_manager)
        _link_bootstrap_owner(sentinel_manager)
        owner_sentinel_restore: Callable[[], None] | None = None
        pipeline_placeholder_token = None
        owner_placeholder_registered = _push_fallback_owner(placeholder_manager)
        try:
            owner_sentinel_restore = _activate_bootstrap_sentinel(placeholder_manager)
        except Exception:  # pragma: no cover - bootstrap guard best effort
            owner_sentinel_restore = None
        try:
            logger.debug(
                "activated bootstrap owner sentinel %s for %s",
                placeholder_manager,
                name,
            )
            owner_context = _push_bootstrap_context(
                registry=bot_registry,
                data_bot=data_bot,
                manager=placeholder_manager,
                pipeline=pipeline,
            )
            if owner_context is not None:
                owner_context.sentinel = placeholder_manager
                if pipeline is not None and owner_context.pipeline is None:
                    owner_context.pipeline = pipeline
            try:
                code_db_cls = _load_optional_module("code_database").CodeDB
                memory_cls = _load_optional_module("gpt_memory").GPTMemoryManager
                engine_mod = _load_optional_module(
                    "self_coding_engine", fallback="menace.self_coding_engine"
                )
                pipeline_mod = _load_optional_module(
                    "model_automation_pipeline", fallback="menace.model_automation_pipeline"
                )
                if pipeline is None:
                    ctx_builder = create_context_builder()
                else:
                    ctx_builder = getattr(pipeline, "context_builder", None)
                    if ctx_builder is None:
                        ctx_builder = create_context_builder()
                engine = engine_mod.SelfCodingEngine(
                    code_db_cls(),
                    memory_cls(),
                    context_builder=ctx_builder,
                )
                pipeline_cls = getattr(pipeline_mod, "ModelAutomationPipeline", None)
                if pipeline_cls is None:
                    raise RuntimeError("ModelAutomationPipeline is unavailable")
                local_pipeline = pipeline
                promote: Callable[[Any], None] | None = pipeline_promoter
                if local_pipeline is None:
                    fallback_runtime_manager = pipeline_manager
                    if _is_bootstrap_placeholder(fallback_runtime_manager):
                        fallback_runtime_manager = None
                    if fallback_runtime_manager is None:
                        fallback_runtime_manager = placeholder_manager
                    marker = None
                    if fallback_runtime_manager is not None:
                        marker = getattr(
                            fallback_runtime_manager,
                            "mark_bootstrap_runtime",
                            None,
                        )
                    if callable(marker):
                        try:
                            marker()
                        except Exception:  # pragma: no cover - best effort flagging
                            logger.debug(
                                "failed to mark fallback runtime manager as bootstrap placeholder",
                                exc_info=True,
                            )
                    helper_override_token = None
                    comm_override_reset: Callable[[], None] | None = None
                    try:
                        if isinstance(
                            fallback_runtime_manager, _DisabledSelfCodingManager
                        ):
                            helper_override_token = _push_helper_manager_override(
                                fallback_runtime_manager
                            )
                        comm_override_reset = _seed_comm_bot_bootstrap_override(
                            fallback_runtime_manager
                        )
                        local_pipeline, promote = prepare_pipeline_for_bootstrap(
                            pipeline_cls=pipeline_cls,
                            context_builder=ctx_builder,
                            bot_registry=bot_registry,
                            data_bot=data_bot,
                            bootstrap_manager=placeholder_manager,
                            bootstrap_runtime_manager=fallback_runtime_manager,
                            force_manager_kwarg=True,
                            manager_override=placeholder_manager,
                            manager_sentinel=placeholder_manager,
                            extra_manager_sentinels=(
                                tuple(dict.fromkeys(fallback_manager_sentinels))
                                if fallback_manager_sentinels
                                else None
                            ),
                            manager=fallback_runtime_manager,
                        )
                    finally:
                        if helper_override_token is not None:
                            _reset_helper_manager_override(helper_override_token)
                        if comm_override_reset is not None:
                            try:
                                comm_override_reset()
                            except Exception:
                                logger.debug(
                                    "communication maintenance override cleanup failed",
                                    exc_info=True,
                                )
                else:
                    placeholder_for_pipeline = sentinel_manager or placeholder_manager
                    pipeline_placeholder_token = _seed_existing_pipeline_placeholder(
                        local_pipeline,
                        placeholder_for_pipeline,
                    )
                if not callable(promote):
                    def _default_promote(
                        real_manager: Any,
                        *,
                        extra_sentinels: Iterable[Any] | None = None,
                    ) -> None:
                        combined: list[Any] = []
                        seen: set[int] = set()
                        for group in (fallback_manager_sentinels, extra_sentinels):
                            if not group:
                                continue
                            for candidate in group:
                                if candidate is None:
                                    continue
                                key = id(candidate)
                                if key in seen:
                                    continue
                                seen.add(key)
                                combined.append(candidate)
                        _promote_pipeline_manager(
                            local_pipeline,
                            real_manager,
                            sentinel_manager,
                            extra_sentinels=combined or None,
                        )

                    promote = _default_promote
                else:
                    delegate = promote

                    def _wrapped_promote(
                        real_manager: Any,
                        *,
                        extra_sentinels: Iterable[Any] | None = None,
                        _extras: Iterable[Any] | None = fallback_manager_sentinels,
                    ) -> None:
                        combined: list[Any] = []
                        seen: set[int] = set()
                        for group in (_extras, extra_sentinels):
                            if not group:
                                continue
                            for candidate in group:
                                if candidate is None:
                                    continue
                                key = id(candidate)
                                if key in seen:
                                    continue
                                seen.add(key)
                                combined.append(candidate)
                        try:
                            delegate(real_manager, extra_sentinels=combined or None)
                        except TypeError:
                            delegate(real_manager)

                    promote = _wrapped_promote

                bind_promoter = getattr(bootstrap_owner, "bind_pipeline_promoter", None)
                if callable(bind_promoter) and callable(promote):
                    bind_promoter(promote)
                pipeline = local_pipeline
                pipeline_manager = getattr(pipeline, "manager", None)
                if placeholder_manager is not None and not _is_bootstrap_placeholder(
                    pipeline_manager
                ):
                    _assign_bootstrap_manager_placeholder(
                        pipeline,
                        placeholder_manager,
                        propagate_nested=True,
                    )
                    pipeline_manager = getattr(pipeline, "manager", None)
                if _is_bootstrap_placeholder(pipeline_manager):
                    _link_bootstrap_owner(pipeline_manager)
                    if pipeline_manager is not bootstrap_owner:
                        _assign_bootstrap_manager_placeholder(
                            pipeline,
                            bootstrap_owner,
                            propagate_nested=True,
                        )
            finally:
                if owner_context is not None:
                    _pop_bootstrap_context(owner_context)
            manager_mod = _load_optional_module(
                "self_coding_manager", fallback="menace.self_coding_manager"
            )
            manager = manager_mod.SelfCodingManager(
                engine,
                pipeline,
                bot_name=name,
                data_bot=data_bot,
                bot_registry=bot_registry,
            )
            helper_callbacks = _consume_bootstrap_helper_callbacks()
            attach_delegate = getattr(sentinel_manager, "attach_delegate", None)
            if callable(attach_delegate):
                try:
                    attach_delegate(manager)
                except Exception:  # pragma: no cover - best effort bridge
                    logger.debug("sentinel delegate attachment failed", exc_info=True)
            callback_queue: list[Callable[[Any], None]] = []
            if helper_callbacks:
                callback_queue.extend(helper_callbacks)
            if callable(promote):
                def _promote_with_extras(
                    real_manager: Any,
                    *,
                    _promote: Callable[[Any], None] = promote,
                    _extras: Iterable[Any] | None = fallback_manager_sentinels,
                ) -> None:
                    try:
                        _promote(real_manager, extra_sentinels=_extras or None)
                    except TypeError:
                        _promote(real_manager)

                callback_queue.append(_promote_with_extras)
            ordered_callbacks: list[Callable[[Any], None]] = []
            seen_callbacks: set[int] = set()
            for callback in callback_queue:
                if not callable(callback):
                    continue
                key = id(callback)
                if key in seen_callbacks:
                    continue
                seen_callbacks.add(key)
                ordered_callbacks.append(callback)
            for callback in ordered_callbacks:
                try:
                    callback(manager)
                except Exception:  # pragma: no cover - promotion must stay best-effort
                    logger.debug(
                        "legacy bootstrap failed to promote pipeline helpers", exc_info=True
                    )
            if pipeline is not None:
                try:
                    promoted_manager = getattr(pipeline, "manager", None)
                except Exception:
                    promoted_manager = None
                if promoted_manager is not manager:
                    raise AssertionError(
                        "pipeline manager promotion failed to expose the real manager"
                    )
            mark_ready = getattr(bootstrap_owner, "mark_ready", None)
            if callable(mark_ready):
                mark_ready(
                    manager,
                    sentinel=sentinel_manager,
                    extra_sentinels=fallback_manager_sentinels or None,
                )
            logger.debug(
                "bootstrap owner sentinel released for %s", name
            )
            _settle_reentrant_promises(manager)
            return manager
        finally:
            if owner_sentinel_restore is not None:
                owner_sentinel_restore()
            if pipeline_placeholder_token is not None:
                try:
                    MANAGER_CONTEXT.reset(pipeline_placeholder_token)
                except Exception:  # pragma: no cover - context vars best effort
                    logger.debug(
                        "failed to reset existing pipeline manager context", exc_info=True
                    )
            if owner_placeholder_registered:
                _pop_fallback_owner(placeholder_manager)
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - heavy bootstrap fallback
        raise RuntimeError(f"manager bootstrap failed: {exc}") from exc
    finally:
        _settle_reentrant_promises(None)
        _release_guard()
        current_depth = getattr(_BOOTSTRAP_STATE, "depth", 1) - 1
        if current_depth > 0:
            _BOOTSTRAP_STATE.depth = current_depth
        elif hasattr(_BOOTSTRAP_STATE, "depth"):
            delattr(_BOOTSTRAP_STATE, "depth")
        if previous_sentinel is _SENTINEL_UNSET:
            try:
                delattr(_BOOTSTRAP_STATE, "sentinel_manager")
            except AttributeError:
                pass
        else:
            _BOOTSTRAP_STATE.sentinel_manager = previous_sentinel
        logger.debug(
            "restored bootstrap sentinel for %s (depth=%s)",
            name,
            getattr(_BOOTSTRAP_STATE, "depth", 0),
        )


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

    context = _current_bootstrap_context()

    if manager is None:
        manager = getattr(obj, "manager", None)
        if manager is None and context is not None:
            context_manager = getattr(context, "manager", None)
            if context_manager is not None:
                manager = context_manager
        if manager is None and _using_bootstrap_sentinel():
            manager = getattr(_BOOTSTRAP_STATE, "sentinel_manager", None)

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
                pipeline_hint = None
                if _looks_like_pipeline_candidate(obj):
                    pipeline_hint = obj
                else:
                    candidate_pipeline = getattr(obj, "pipeline", None)
                    if _looks_like_pipeline_candidate(candidate_pipeline):
                        pipeline_hint = candidate_pipeline
                if pipeline_hint is None and context is not None:
                    candidate_pipeline = getattr(context, "pipeline", None)
                    if _looks_like_pipeline_candidate(candidate_pipeline):
                        pipeline_hint = candidate_pipeline
                manager = _bootstrap_manager(
                    name_local,
                    registry,
                    data_bot,
                    pipeline=pipeline_hint,
                )
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
        register_deferred = False
        deferred_registration: dict[str, Any] | None = None
        decision: _BootstrapDecision | None = None
        resolved_registry: BotRegistry | None = None
        resolved_data_bot: DataBot | None = None
        is_pipeline_cls = _is_model_automation_pipeline_class(cls)
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

        def _bootstrap_helpers(
            *,
            pipeline: Any | None = None,
            override_manager: Any | None = None,
        ) -> tuple[BotRegistry, DataBot, _BootstrapContextGuard | None]:
            nonlocal manager_instance, update_kwargs, should_update, register_as_coding
            nonlocal decision, resolved_registry, resolved_data_bot, bootstrap_done
            nonlocal bootstrap_in_progress, bootstrap_error
            nonlocal register_deferred, deferred_registration

            if bootstrap_done and resolved_registry is not None and resolved_data_bot is not None:
                return resolved_registry, resolved_data_bot, None

            wait_for_completion = False
            with bootstrap_lock:
                if bootstrap_done and resolved_registry is not None and resolved_data_bot is not None:
                    return resolved_registry, resolved_data_bot, None
                if bootstrap_in_progress:
                    wait_for_completion = True
                else:
                    bootstrap_in_progress = True
                    bootstrap_event.clear()

            if wait_for_completion:
                active_context = _current_bootstrap_context()
                if active_context is not None and (
                    getattr(active_context, "manager", None) is not None
                    or getattr(active_context, "pipeline", None) is not None
                ):
                    logger.debug(
                        "%s: reusing active bootstrap context (manager=%s, pipeline=%s)",
                        name,
                        getattr(active_context, "manager", None),
                        getattr(active_context, "pipeline", None),
                    )
                    return (
                        getattr(active_context, "registry", None),
                        getattr(active_context, "data_bot", None),
                        None,
                    )
                if not bootstrap_event.wait(timeout=_BOOTSTRAP_WAIT_TIMEOUT):
                    timeout_error = TimeoutError(
                        "Bot helper bootstrap timed out waiting for prior initialisation"
                    )
                    with bootstrap_lock:
                        bootstrap_error = timeout_error
                        bootstrap_in_progress = False
                        bootstrap_done = False
                        bootstrap_event.set()
                    logger.error(
                        "Bootstrap coordination stalled; falling back to fail-fast behaviour",
                        exc_info=timeout_error,
                    )
                    raise timeout_error
                if bootstrap_error is not None:
                    raise bootstrap_error
                if not bootstrap_done or resolved_registry is None or resolved_data_bot is None:
                    raise RuntimeError("Bot helper bootstrap did not complete successfully")
                return resolved_registry, resolved_data_bot, None

            context: _BootstrapContext | None = None
            context_guard: _BootstrapContextGuard | None = None
            context_guard_transferred = False
            nested_owner_placeholder: Any | None = None
            nested_owner_token = None
            nested_owner_previous = _SENTINEL_UNSET
            nested_owner_installed = False
            parent_context = _current_bootstrap_context()
            contextual_manager = None
            contextual_placeholder: Any | None = None
            contextual_sentinel: Any | None = None
            override_for_context = override_manager
            if override_for_context is None:
                override_for_context = _current_helper_manager_override()
            pipeline_for_context = _current_pipeline_context()
            if not _looks_like_pipeline_candidate(pipeline_for_context):
                pipeline_for_context = None
            if pipeline_for_context is None and pipeline is not None and _looks_like_pipeline_candidate(pipeline):
                pipeline_for_context = pipeline
            if parent_context is not None:
                contextual_manager = parent_context.manager
                contextual_sentinel = getattr(parent_context, "sentinel", None)
                if pipeline_for_context is None:
                    pipeline_for_context = getattr(parent_context, "pipeline", None)
                if contextual_manager is not None and _is_bootstrap_placeholder(
                    contextual_manager
                ):
                    contextual_placeholder = contextual_manager
                if (
                    contextual_placeholder is None
                    and contextual_sentinel is not None
                    and _is_bootstrap_placeholder(contextual_sentinel)
                ):
                    contextual_placeholder = contextual_sentinel
            depth_level = getattr(_BOOTSTRAP_STATE, "depth", 0)
            sentinel_guard_active = False
            if contextual_sentinel is not None:
                sentinel_guard_active = bool(
                    _is_bootstrap_owner(contextual_sentinel)
                    and getattr(contextual_sentinel, "bootstrap_owner_active", False)
                )
            if (
                depth_level > 0
                and contextual_sentinel is not None
                and contextual_sentinel is not contextual_manager
                and sentinel_guard_active
            ):
                contextual_manager = contextual_sentinel
            try:
                registry_obj = _resolve_candidate(bot_registry)
                data_bot_obj = _resolve_candidate(data_bot)
                sentinel_manager = getattr(
                    _BOOTSTRAP_STATE, "sentinel_manager", None
                )
                if sentinel_manager is None and contextual_placeholder is not None:
                    sentinel_manager = contextual_placeholder
                manager_for_context = contextual_manager
                if manager_for_context is None and override_for_context is not None:
                    manager_for_context = override_for_context
                if manager_for_context is None and contextual_sentinel is not None:
                    manager_for_context = contextual_sentinel
                if manager_for_context is None:
                    manager_for_context = (
                        manager_instance
                        if manager_instance is not None
                        else sentinel_manager
                    )
                context = _push_bootstrap_context(
                    registry=registry_obj,
                    data_bot=data_bot_obj,
                    manager=manager_for_context,
                    pipeline=pipeline_for_context,
                )
                if (
                    context is not None
                    and pipeline_for_context is not None
                    and _looks_like_pipeline_candidate(pipeline_for_context)
                ):
                    context_guard = _BootstrapContextGuard(context)
                if context is not None and context.pipeline is None:
                    context.pipeline = pipeline_for_context
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

                manager_local = manager_for_context
                if manager_local is None:
                    manager_local = (
                        manager_instance
                        if manager_instance is not None
                        else sentinel_manager
                    )
                sentinel_placeholder = False

                def _refresh_bootstrap_strategy(candidate: Any) -> None:
                    nonlocal manager_local, sentinel_placeholder, strategy
                    strategy = _should_bootstrap_manager(candidate)
                    manager_local = strategy.manager
                    sentinel_placeholder = strategy.sentinel is not None

                strategy: _ManagerBootstrapPlan
                _refresh_bootstrap_strategy(manager_local)
                runtime_available = _self_coding_runtime_available()
                register_deferred_local = register_deferred
                sentinel_in_use = False
                owner_sentinel_active = False
                if (
                    strategy.seed_owner_sentinel
                    and manager_local is None
                    and registry_obj is not None
                    and data_bot_obj is not None
                ):
                    nested_owner_placeholder = _spawn_nested_bootstrap_owner(
                        registry_obj,
                        data_bot_obj,
                    )
                    manager_local = nested_owner_placeholder
                    _refresh_bootstrap_strategy(manager_local)
                    sentinel_manager = nested_owner_placeholder
                    nested_owner_previous = getattr(
                        _BOOTSTRAP_STATE, "sentinel_manager", _SENTINEL_UNSET
                    )
                    try:
                        _BOOTSTRAP_STATE.sentinel_manager = nested_owner_placeholder
                    except Exception:  # pragma: no cover - best effort exposure
                        logger.debug(
                            "%s: failed to expose nested bootstrap sentinel",
                            name,
                            exc_info=True,
                        )
                        nested_owner_installed = False
                    else:
                        nested_owner_installed = True
                    try:
                        nested_owner_token = MANAGER_CONTEXT.set(
                            nested_owner_placeholder
                        )
                    except Exception:  # pragma: no cover - context vars best effort
                        nested_owner_token = None
                    logger.debug(
                        "%s: installed nested bootstrap owner sentinel at depth=%s",
                        name,
                        getattr(_BOOTSTRAP_STATE, "depth", 0),
                    )
                    if context is not None:
                        context.manager = nested_owner_placeholder
                        context.sentinel = nested_owner_placeholder
                    register_as_coding_local = False
                    register_deferred_local = True

                    def _deferred_finalize(
                        real_manager: Any,
                        *,
                        _registry: Any = registry_obj,
                        _data_bot: Any = data_bot_obj,
                    ) -> None:
                        try:
                            _finalize_self_coding_bootstrap(
                                real_manager,
                                registry=_registry,
                                data_bot=_data_bot,
                            )
                        except TypeError:
                            _finalize_self_coding_bootstrap(real_manager)

                    _register_bootstrap_helper_callback(_deferred_finalize)
                if strategy.defer and strategy.sentinel is not None:
                    logger.debug(
                        "%s: sentinel manager active at bootstrap depth=%s; deferring real manager promotion",
                        name,
                        getattr(_BOOTSTRAP_STATE, "depth", 0),
                    )
                    subscribe = getattr(
                        _finalize_self_coding_bootstrap,
                        "_subscribe_to_deferred_promotion",
                        None,
                    )
                    if callable(subscribe):
                        subscribe(strategy.sentinel)
                    if context is not None:
                        context.manager = manager_local
                        if _is_bootstrap_placeholder(manager_local):
                            context.sentinel = manager_local
                    register_as_coding_local = False
                    register_deferred_local = True
                    sentinel_in_use = True
                else:
                    sentinel_in_use = _using_bootstrap_sentinel(manager_local)
                if getattr(manager_local, "bootstrap_runtime_active", False):
                    sentinel_in_use = True
                if _is_bootstrap_owner(manager_local):
                    owner_sentinel_active = bool(
                        getattr(manager_local, "bootstrap_owner_active", False)
                    )
                    if owner_sentinel_active:
                        sentinel_in_use = True
                    elif sentinel_in_use:
                        logger.debug(
                            "%s: bootstrap owner sentinel detected without active guard; awaiting promotion",
                            name,
                        )
                if sentinel_in_use:
                    subscribe = getattr(
                        _finalize_self_coding_bootstrap,
                        "_subscribe_to_deferred_promotion",
                        None,
                    )
                    if callable(subscribe):
                        subscribe(manager_local)
                    register_as_coding_local = False
                    register_deferred_local = True
                placeholder_manager = (
                    manager_local is None or not _is_self_coding_manager(manager_local)
                )
                if placeholder_manager and not strategy.should_bootstrap:
                    register_as_coding_local = False
                    register_deferred_local = True
                if sentinel_placeholder and not strategy.defer:
                    logger.debug(
                        "%s: stale sentinel manager detected; triggering manager bootstrap",
                        name,
                    )
                    manager_local = strategy.manager
                if context is not None:
                    context.manager = manager_local
                    if _is_bootstrap_placeholder(manager_local):
                        context.sentinel = manager_local
                if (not sentinel_in_use) and strategy.should_bootstrap and runtime_available:
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
                            if manager_local is None:
                                context_pipeline = (
                                    getattr(context, "pipeline", None)
                                    if context is not None
                                    else None
                                )
                                manager_local = _bootstrap_manager(
                                    name,
                                    registry_obj,
                                    data_bot_obj,
                                    pipeline=context_pipeline,
                                )
                                _refresh_bootstrap_strategy(manager_local)
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
                    if not sentinel_in_use:
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

                register_kwargs: dict[str, Any] | None = None
                if (
                    registry_obj is not None
                    and (register_as_coding_local or register_deferred_local)
                ):
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
                                    param.kind == inspect.Parameter.VAR_KEYWORD
                                    for param in params.values()
                                )
                        if include_factory:
                            register_kwargs["orchestrator_factory"] = orchestrator_factory

                deferred_payload_local: dict[str, Any] | None = None
                if registry_obj is not None:
                    if register_deferred_local or (
                        placeholder_manager and not register_as_coding_local
                    ):
                        deferred_payload_local = {
                            "register_kwargs": dict(register_kwargs or {}),
                            "roi_threshold": roi_t,
                            "error_threshold": err_t,
                            "should_update": should_update_local,
                            "update_kwargs": dict(update_kwargs_local),
                        }
                    elif register_as_coding_local:
                        try:
                            registry_obj.register_bot(
                                name,
                                module_path,
                                manager=manager_local,
                                roi_threshold=roi_t,
                                error_threshold=err_t,
                                **(register_kwargs or {}),
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

                    if should_update_local and not register_deferred_local:
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
                    cls._self_coding_manual_mode = (
                        register_deferred_local or not register_as_coding_local
                    )
                    try:
                        cls._self_coding_pending_manager = register_deferred_local  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - defensive
                        logger.debug(
                            "%s: failed to flag pending manager promotion", cls, exc_info=True
                        )

                    manager_instance = manager_local
                    update_kwargs = update_kwargs_local
                    should_update = should_update_local
                    register_as_coding = register_as_coding_local
                    register_deferred = register_deferred_local
                    deferred_registration = deferred_payload_local
                    decision = decision_local
                    resolved_registry = registry_obj
                    resolved_data_bot = data_bot_obj
                    bootstrap_done = True
                    bootstrap_error = None
                    bootstrap_in_progress = False
                    bootstrap_event.set()
                context_guard_transferred = bool(context_guard)
                return registry_obj, data_bot_obj, context_guard
            except BaseException as exc:
                with bootstrap_lock:
                    bootstrap_error = exc
                    bootstrap_in_progress = False
                    bootstrap_event.set()
                raise
            finally:
                if nested_owner_token is not None:
                    try:
                        MANAGER_CONTEXT.reset(nested_owner_token)
                    except Exception:  # pragma: no cover - best effort reset
                        logger.debug(
                            "%s: failed to reset nested manager context", name,
                            exc_info=True,
                        )
                if nested_owner_installed:
                    if nested_owner_previous is _SENTINEL_UNSET:
                        try:
                            delattr(_BOOTSTRAP_STATE, "sentinel_manager")
                        except AttributeError:
                            pass
                    else:
                        _BOOTSTRAP_STATE.sentinel_manager = nested_owner_previous
                if not context_guard_transferred and context is not None:
                    _pop_bootstrap_context(context)

        cls.bot_registry = None  # type: ignore[attr-defined]
        cls.data_bot = None  # type: ignore[attr-defined]
        cls.manager = manager_instance  # type: ignore[attr-defined]
        cls._self_coding_manual_mode = True
        cls._self_coding_pending_manager = False  # type: ignore[attr-defined]

        def _finalize_self_coding_bootstrap(
            real_manager: Any,
            *,
            registry: Any | None = None,
            data_bot: Any | None = None,
        ) -> None:
            """Promote sentinel helpers to ``real_manager`` once available."""

            nonlocal manager_instance, resolved_registry, resolved_data_bot
            nonlocal bootstrap_done, bootstrap_error, bootstrap_in_progress
            nonlocal register_deferred, deferred_registration

            if real_manager is None:
                return

            registry_obj = registry
            if registry_obj is None:
                registry_obj = resolved_registry or getattr(real_manager, "bot_registry", None)
            data_bot_obj = data_bot
            if data_bot_obj is None:
                data_bot_obj = resolved_data_bot or getattr(real_manager, "data_bot", None)

            promotion_kwargs: dict[str, Any] | None = None
            should_promote_registry = False
            pending_registration: dict[str, Any] | None = None

            try:
                with bootstrap_lock:
                    manager_instance = real_manager
                    try:
                        cls.manager = real_manager  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - defensive
                        logger.debug(
                            "%s: failed to assign promoted manager", cls, exc_info=True
                        )
                    if registry_obj is not None:
                        resolved_registry = registry_obj
                        try:
                            cls.bot_registry = registry_obj  # type: ignore[attr-defined]
                        except Exception:  # pragma: no cover - defensive
                            logger.debug(
                                "%s: failed to assign promoted registry", cls, exc_info=True
                            )
                    if data_bot_obj is not None:
                        resolved_data_bot = data_bot_obj
                        try:
                            cls.data_bot = data_bot_obj  # type: ignore[attr-defined]
                        except Exception:  # pragma: no cover - defensive
                            logger.debug(
                                "%s: failed to assign promoted data bot", cls, exc_info=True
                            )
                    pending_registration = deferred_registration
                    deferred_registration = None
                    was_deferred = register_deferred
                    register_deferred = False
                    if (
                        (register_as_coding or was_deferred)
                        and registry_obj is not None
                    ):
                        should_promote_registry = True
                        promotion_kwargs = dict(update_kwargs or {})
                        if (
                            decision is not None
                            and getattr(decision, "provenance", None) is not None
                            and promotion_kwargs is not None
                            and "provenance" not in promotion_kwargs
                        ):
                            promotion_kwargs["provenance"] = decision.provenance
                    cls._self_coding_manual_mode = False
                    try:
                        cls._self_coding_pending_manager = False  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - defensive
                        logger.debug(
                            "%s: failed to reset pending manager flag", cls, exc_info=True
                        )
                    bootstrap_done = True
                    bootstrap_error = None
                    bootstrap_in_progress = False
                    bootstrap_event.set()
                if should_promote_registry:
                    helper = getattr(registry_obj, "promote_self_coding_manager", None)
                    if callable(helper):
                        try:
                            helper(
                                name,
                                real_manager,
                                data_bot_obj,
                                **(promotion_kwargs or {}),
                            )
                        except Exception:  # pragma: no cover - best effort
                            logger.debug(
                                "%s: registry promotion failed for %s", cls, name, exc_info=True
                            )
                if pending_registration is not None and registry_obj is not None:
                    try:
                        registry_obj.register_bot(
                            name,
                            module_path,
                            manager=real_manager,
                            roi_threshold=pending_registration.get("roi_threshold"),
                            error_threshold=pending_registration.get("error_threshold"),
                            **pending_registration.get("register_kwargs", {}),
                        )
                    except Exception:  # pragma: no cover - best effort
                        logger.exception(
                            "bot registration failed during deferred promotion for %s",
                            name,
                        )
                    if (
                        pending_registration.get("should_update")
                        and pending_registration.get("update_kwargs")
                    ):
                        try:
                            registry_obj.update_bot(
                                name,
                                module_path,
                                **pending_registration["update_kwargs"],
                            )
                        except Exception:  # pragma: no cover - best effort
                            logger.exception(
                                "bot update failed during deferred promotion for %s", name
                            )
            except Exception:  # pragma: no cover - promotion should be best-effort
                logger.debug(
                    "%s: manager promotion finalisation failed", cls, exc_info=True
                )

        def _subscribe_to_deferred_promotion(
            sentinel: Any | None,
        ) -> None:
            if sentinel is None:
                return
            key = id(sentinel)
            if key in _DEFERRED_SENTINEL_CALLBACKS:
                return
            _DEFERRED_SENTINEL_CALLBACKS.add(key)

            def _trigger(real_manager: Any) -> None:
                pipeline = getattr(real_manager, "pipeline", None)
                _promote_pipeline_manager(pipeline, real_manager, sentinel)

            subscribe_callback = getattr(sentinel, "add_delegate_callback", None)
            if not callable(subscribe_callback):
                subscribe_callback = getattr(sentinel, "add_promotion_callback", None)
            if not callable(subscribe_callback):
                return
            subscribe_callback(_trigger)

        cls._finalize_self_coding_bootstrap = staticmethod(  # type: ignore[attr-defined]
            _finalize_self_coding_bootstrap
        )
        _finalize_self_coding_bootstrap._subscribe_to_deferred_promotion = (  # type: ignore[attr-defined]
            _subscribe_to_deferred_promotion
        )

        @wraps(orig_init)
        def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
            registry_obj: BotRegistry | None = None
            data_bot_obj: DataBot | None = None
            pipeline_ref: Any | None = self if is_pipeline_cls else None
            context_guard: _BootstrapContextGuard | None = None
            try:
                if (not bootstrap_done) or resolved_registry is None or resolved_data_bot is None:
                    registry_obj, data_bot_obj, context_guard = _bootstrap_helpers(
                        pipeline=pipeline_ref,
                        override_manager=_current_helper_manager_override(),
                    )
                else:
                    registry_obj = resolved_registry
                    data_bot_obj = resolved_data_bot

                context = _current_bootstrap_context()
                if (
                    context is not None
                    and context.registry is not None
                    and context.data_bot is not None
                ):
                    registry_obj = context.registry
                    data_bot_obj = context.data_bot
                    manager_default = (
                        context.manager
                        if context.manager is not None
                        else manager_instance
                    )
                else:
                    manager_default = manager_instance

                if registry_obj is None or data_bot_obj is None:
                    if context_guard is not None:
                        context_guard.release()
                        context_guard = None
                    registry_obj, data_bot_obj, context_guard = _bootstrap_helpers(
                        pipeline=pipeline_ref,
                        override_manager=_current_helper_manager_override(),
                    )
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
            finally:
                if context_guard is not None:
                    context_guard.release()

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
            pending_manager_flag = bool(
                getattr(cls, "_self_coding_pending_manager", False)
            )
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

            if pending_manager_flag:
                try:
                    self._self_coding_pending_manager = True  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - best effort marker
                    logger.debug(
                        "%s: failed to flag pending manager on %s", name_local, self, exc_info=True
                    )

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
                    builder, builder_is_stub = getattr(
                        _EO, "resolve_context_builder", lambda _=None: (create_context_builder(), False)
                    )(logger)
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
                    if builder_is_stub:
                        try:
                            orchestrator.context_builder_degraded = True
                        except Exception:
                            logger.debug("failed to flag degraded context builder", exc_info=True)
                except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: EvolutionOrchestrator dependencies unavailable: %s",
                        cls.__name__,
                        exc,
                    )
                    orchestrator_boot_failed = True
                    orchestrator = None
                except Exception as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: EvolutionOrchestrator is required but could not be instantiated; disabling orchestration",
                        cls.__name__,
                        exc_info=exc,
                    )
                    orchestrator_boot_failed = True
                    orchestrator = None

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
