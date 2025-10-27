from __future__ import annotations

"""Graph based registry capturing bot interactions.

The registry persists bot connections to a database and allows bots to be
hot swapped at runtime. Updating a bot's backing module via ``update_bot``
broadcasts a ``bot:updated`` event so other components can react to the
change.
"""

from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from collections.abc import MutableMapping
from pathlib import Path
import time
import importlib
import importlib.util
import sys
import subprocess
import json
import os
import re
import threading
from contextlib import contextmanager
import errno
from dataclasses import asdict, dataclass, field, is_dataclass
from types import SimpleNamespace

from shared.provenance_state import (
    UNSIGNED_PROVENANCE_WARNING_CACHE as _UNSIGNED_PROVENANCE_WARNING_CACHE,
    UNSIGNED_PROVENANCE_WARNING_LAST_TS as _UNSIGNED_PROVENANCE_WARNING_LAST_TS,
    UNSIGNED_PROVENANCE_WARNING_LOCK as _UNSIGNED_PROVENANCE_WARNING_LOCK,
)

try:
    from .shared.self_coding_import_guard import is_self_coding_import_active
except ImportError:  # pragma: no cover - support legacy layout
    from shared.self_coding_import_guard import (  # type: ignore
        is_self_coding_import_active,
    )

try:
    from .databases import MenaceDB
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore
try:
    from .neuroplasticity import PathwayDB
except Exception:  # pragma: no cover - optional dependency
    PathwayDB = None  # type: ignore

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - allow execution without networkx
    from collections.abc import Iterator, MutableMapping

    class _NodeView(MutableMapping):
        """Lightweight mapping that mirrors ``networkx`` node access semantics."""

        def __init__(self, graph: "_DiGraph") -> None:
            self._graph = graph

        def __getitem__(self, key: str) -> dict:
            return self._graph._ensure_node(key)

        def __setitem__(self, key: str, value: dict) -> None:
            self._graph._nodes[key] = dict(value)
            self._graph._adj.setdefault(key, {})

        def __delitem__(self, key: str) -> None:
            self._graph._nodes.pop(key, None)
            self._graph._adj.pop(key, None)
            for nbrs in self._graph._adj.values():
                nbrs.pop(key, None)

        def __contains__(self, key: object) -> bool:
            return key in self._graph._nodes

        def __iter__(self) -> Iterator[str]:
            return iter(self._graph._nodes)

        def __len__(self) -> int:
            return len(self._graph._nodes)

        def get(self, key: str, default: Optional[dict] = None) -> Optional[dict]:
            return self._graph._nodes.get(key, default)

    class _EdgeView:
        """Subset of the ``networkx`` edge view interface used in the registry."""

        def __init__(self, graph: "_DiGraph") -> None:
            self._graph = graph

        def _iter_edges(self, data: bool = False, default: Optional[Any] = None):
            for u, nbrs in self._graph._adj.items():
                for v, attrs in nbrs.items():
                    if data:
                        yield (u, v, dict(attrs))
                    else:
                        yield (u, v)

        def __iter__(self):  # pragma: no cover - exercised indirectly
            return self._iter_edges()

        def __call__(self, data: bool = False, default: Optional[Any] = None):
            return self._iter_edges(data=data, default=default)

        def data(self, data: bool = True, default: Optional[Any] = None):
            return self._iter_edges(data=data, default=default)

    class _DiGraph:
        """Minimal ``DiGraph`` implementation covering the registry use cases."""

        def __init__(self) -> None:
            self._nodes: Dict[str, dict] = {}
            self._adj: Dict[str, Dict[str, dict]] = {}
            self.nodes = _NodeView(self)

        def _ensure_node(self, node: str) -> dict:
            data = self._nodes.setdefault(node, {})
            self._adj.setdefault(node, {})
            return data

        def add_node(self, node: str, **attrs: Any) -> None:
            data = self._ensure_node(node)
            if attrs:
                data.update(attrs)

        def add_edge(self, u: str, v: str, **attrs: Any) -> None:
            self.add_node(u)
            self.add_node(v)
            edge = self._adj[u].setdefault(v, {})
            edge.update(attrs)

        def has_edge(self, u: str, v: str) -> bool:
            return v in self._adj.get(u, {})

        def successors(self, node: str) -> Iterator[str]:
            return iter(self._adj.get(node, {}))

        @property
        def edges(self) -> _EdgeView:  # pragma: no cover - exercised indirectly
            return _EdgeView(self)

        def clear(self) -> None:
            self._nodes.clear()
            self._adj.clear()

        def __contains__(self, node: object) -> bool:
            return node in self._nodes

        def __getitem__(self, node: str) -> Dict[str, dict]:
            self._ensure_node(node)
            return self._adj[node]

    class _NetworkXShim:
        DiGraph = _DiGraph

    nx = _NetworkXShim()  # type: ignore[assignment]

import logging
import traceback
from datetime import datetime, timezone

try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover
    class UnifiedEventBus:  # type: ignore[override]
        pass
try:  # pragma: no cover - allow flat imports
    from .shared_event_bus import event_bus as _SHARED_EVENT_BUS
except Exception:  # pragma: no cover - flat layout fallback
    from shared_event_bus import event_bus as _SHARED_EVENT_BUS  # type: ignore
import db_router
from db_router import DBRouter, init_db_router

try:  # pragma: no cover - allow flat imports
    from .sandbox_settings import SandboxSettings, normalize_workflow_tests
except Exception:  # pragma: no cover - fallback for flat layout
    from sandbox_settings import SandboxSettings, normalize_workflow_tests  # type: ignore

try:  # pragma: no cover - prefer package aware import helper when available
    from .import_compat import load_internal as _load_internal_module
except Exception:  # pragma: no cover - fallback for flat execution
    from import_compat import load_internal as _load_internal_module  # type: ignore

try:  # pragma: no cover - allow flat imports
    from .threshold_service import threshold_service
except Exception:  # pragma: no cover - fallback for flat layout
    from threshold_service import threshold_service  # type: ignore

try:  # pragma: no cover - allow flat imports
    from .self_coding_dependency_probe import ensure_self_coding_ready
except Exception:  # pragma: no cover - fallback for flat layout
    from self_coding_dependency_probe import ensure_self_coding_ready  # type: ignore

try:  # pragma: no cover - allow flat imports
    from .retry_utils import with_retry
except Exception:  # pragma: no cover - fallback for flat layout
    from retry_utils import with_retry  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .self_coding_manager import SelfCodingManager
    from .data_bot import DataBot
else:  # pragma: no cover - runtime placeholders
    SelfCodingManager = Any  # type: ignore
    DataBot = Any  # type: ignore

try:  # pragma: no cover - allow flat imports
    from .data_bot import persist_sc_thresholds
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from data_bot import persist_sc_thresholds  # type: ignore
    except Exception:  # pragma: no cover - last resort stub
        def persist_sc_thresholds(*_a, **_k):  # type: ignore
            return None

try:  # pragma: no cover - optional dependency
    from .rollback_manager import RollbackManager
except Exception:  # pragma: no cover - optional dependency
    RollbackManager = None  # type: ignore

logger = logging.getLogger(__name__)

_PATCH_PROVENANCE_UNSET = object()
_patch_provenance_service_cls: object = _PATCH_PROVENANCE_UNSET


def _get_patch_provenance_service_cls() -> type | None:
    """Return the :class:`PatchProvenanceService` implementation if available."""

    global _patch_provenance_service_cls
    if _patch_provenance_service_cls is _PATCH_PROVENANCE_UNSET:
        cls: type | None = None
        try:
            from .patch_provenance import PatchProvenanceService as rel_cls
        except ImportError as rel_exc:  # pragma: no cover - runtime guard
            try:
                module = importlib.import_module("patch_provenance")
            except ImportError as abs_exc:
                logger.debug(
                    "Unable to import patch_provenance module (relative error: %s, absolute error: %s)",
                    rel_exc,
                    abs_exc,
                )
                _patch_provenance_service_cls = None
                return None
            cls = getattr(module, "PatchProvenanceService", None)
            if cls is None:
                logger.debug(
                    "patch_provenance module is missing PatchProvenanceService"
                )
                _patch_provenance_service_cls = None
                return None
        else:
            cls = rel_cls
        _patch_provenance_service_cls = cls
    return (
        _patch_provenance_service_cls
        if _patch_provenance_service_cls is not None
        else None
    )
_UNSIGNED_COMMIT_PREFIX = "unsigned:"
_REGISTERED_BOTS: dict[str, dict[str, str]] = {
    "FutureSynergyProfitBot": {
        "module_path": "menace_sandbox.future_prediction_bots",
        "class_name": "FutureSynergyProfitBot",
    },
    "FutureSynergyMaintainabilityBot": {
        "module_path": "menace_sandbox.future_prediction_bots",
        "class_name": "FutureSynergyMaintainabilityBot",
    },
    "FutureSynergyCodeQualityBot": {
        "module_path": "menace_sandbox.future_prediction_bots",
        "class_name": "FutureSynergyCodeQualityBot",
    },
    "FutureSynergyNetworkLatencyBot": {
        "module_path": "menace_sandbox.future_prediction_bots",
        "class_name": "FutureSynergyNetworkLatencyBot",
    },
    "FutureSynergyThroughputBot": {
        "module_path": "menace_sandbox.future_prediction_bots",
        "class_name": "FutureSynergyThroughputBot",
    },
    "FutureSynergySecurityScoreBot": {
        "module_path": "menace_sandbox.bots.future_synergy_security_score_bot",
        "class_name": "FutureSynergySecurityScoreBot",
    },
    "FutureSynergyEfficiencyBot": {
        "module_path": "menace_sandbox.bots.future_synergy_efficiency_bot",
        "class_name": "FutureSynergyEfficiencyBot",
    },
    "FutureSynergyAntifragilityBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyAntifragilityBot",
    },
    "FutureSynergyResilienceBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyResilienceBot",
    },
    "FutureSynergyShannonEntropyBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyShannonEntropyBot",
    },
    "FutureSynergyFlexibilityBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyFlexibilityBot",
    },
    "FutureSynergyEnergyConsumptionBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyEnergyConsumptionBot",
    },
    "FutureSynergyAdaptabilityBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyAdaptabilityBot",
    },
    "FutureSynergySafetyRatingBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergySafetyRatingBot",
    },
    "FutureSynergyRiskIndexBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyRiskIndexBot",
    },
    "FutureSynergyRecoveryTimeBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyRecoveryTimeBot",
    },
    "FutureSynergyDiscrepancyCountBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyDiscrepancyCountBot",
    },
    "FutureSynergyGPUUsageBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyGPUUsageBot",
    },
    "FutureSynergyCPUUsageBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyCPUUsageBot",
    },
    "FutureSynergyMemoryUsageBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyMemoryUsageBot",
    },
    "FutureSynergyLongTermLucrativityBot": {
        "module_path": "menace_sandbox.prediction_manager_bot",
        "class_name": "FutureSynergyLongTermLucrativityBot",
    },
}

_HOT_SWAP_STATE = threading.local()


@contextmanager
def _hot_swap_guard(name: str):
    """Record that ``name`` is being hot swapped for the duration of the import."""

    stack = getattr(_HOT_SWAP_STATE, "stack", [])
    stack.append(name)
    _HOT_SWAP_STATE.stack = stack
    try:
        yield
    finally:
        try:
            stack.pop()
        finally:
            if stack:
                _HOT_SWAP_STATE.stack = stack
            else:
                try:
                    delattr(_HOT_SWAP_STATE, "stack")
                except AttributeError:  # pragma: no cover - defensive cleanup
                    pass


def _hot_swap_active() -> bool:
    """Return ``True`` when any hot swap import is currently active."""

    stack = getattr(_HOT_SWAP_STATE, "stack", None)
    return bool(stack)

_UNSIGNED_PROVENANCE_WARNING_INTERVAL_SECONDS = float(
    os.getenv("MENACE_UNSIGNED_PROVENANCE_WARNING_INTERVAL_SECONDS", "300")
)

_PATH_SEPARATORS: tuple[str, ...] = tuple(
    sorted({sep for sep in (os.sep, os.altsep, "/", "\\") if sep})
)
_QUOTED_TOKEN_RE = re.compile(r"([\'\"])(?P<token>[^\'\"\r\n]+)\1")
_HEX_ADDRESS_RE = re.compile(r"0x[0-9A-Fa-f]+")
_WINDOWS_ABS_PATH_RE = re.compile(
    r"(?<![\w.])(?:[A-Za-z]:\\|\\\\)(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]+"
)
_POSIX_ABS_PATH_RE = re.compile(r"(?<![\w.])/(?:[^\s\'\":]+/)*[^\s\'\":]+")


def _is_probable_filesystem_path(value: str | os.PathLike[str] | None) -> bool:
    """Heuristically determine whether ``value`` refers to a filesystem path.

    ``bot_registry`` persists module references exactly as they were supplied by
    upstream orchestrators.  On Unix-like systems this is typically a dotted
    module path such as ``"menace.task_validation_bot"`` or a relative file path
    with forward slashes.  Windows invocations, however, frequently pass fully
    qualified paths that include drive letters and backslashes.  Prior to this
    helper the registry only inspected the string for ``"/"`` and therefore
    mis-classified legitimate Windows paths as pure module names which caused
    :func:`importlib.import_module` to receive values like
    ``"C:\\menace\\task_validation_bot.py"``.  ``importlib`` rightfully raises a
    :class:`ModuleNotFoundError` for such values, stalling the sandbox during the
    internalisation loop.

    The heuristic intentionally prefers recall over precision: if the value
    *might* represent a path we delegate to ``importlib.util.spec_from_file_location``
    which safely handles non-existent files by raising an informative
    :class:`ImportError`.  The function therefore checks for several signals that
    only appear in filesystem paths – presence of native or alternative path
    separators, explicit drive specifications, relative prefixes and common
    source file extensions.  The resulting behaviour is deterministic across
    platforms which keeps the registry's hot swapping logic aligned between
    Linux CI and production Windows environments.
    """

    if value is None:
        return False

    if isinstance(value, os.PathLike):
        # ``os.fspath`` provides the canonical string representation even for
        # custom ``PathLike`` implementations.
        value = os.fspath(value)

    if not isinstance(value, str) or not value:
        return False

    lower = value.lower()
    if lower.endswith((".py", ".pyc", ".pyo", "__init__")):
        return True

    drive, _ = os.path.splitdrive(value)
    if drive:
        return True

    if value.startswith(("./", "../")):
        return True

    if any(sep in value for sep in _PATH_SEPARATORS):
        return True

    # ``Path`` normalisation is comparatively expensive but captures edge
    # cases such as ``C:module.py`` (drive without separator) or "virtual"
    # absolute prefixes returned by certain packaging workflows.  The
    # conversion deliberately avoids resolving the path to prevent I/O.
    try:
        parts = Path(value).parts
    except TypeError:
        return False
    if len(parts) > 1:
        return True

    return False


def _iter_module_search_roots() -> list[tuple[Path, str | None]]:
    """Return import roots and their associated package names.

    Windows deployments frequently execute the sandbox from unpacked archives
    where ``sys.path`` contains both the project root and namespace package
    directories.  Mapping filesystem paths back to their dotted module names
    therefore requires awareness of both the import system's search roots and
    already-imported packages.  The helper yields a deterministic list of
    ``(Path, package_name)`` tuples with the longest package-prefixed matches
    appearing first so callers can favour canonical package names like
    ``menace_sandbox`` over ad-hoc fallbacks.
    """

    roots: list[tuple[Path, str | None]] = []
    seen: set[Path] = set()

    def _add_root(candidate: Path, package: str | None) -> None:
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            resolved = candidate.absolute()
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append((resolved, package))

    for module in list(sys.modules.values()):
        pkg_path = getattr(module, "__path__", None)
        if not pkg_path:
            continue
        try:
            path_iter = iter(pkg_path)
        except TypeError:
            continue
        package_name = getattr(module, "__name__", None)
        for entry in path_iter:
            try:
                path_obj = Path(entry)
            except TypeError:
                continue
            _add_root(path_obj, package_name)

    for entry in sys.path:
        try:
            path_obj = Path(entry)
        except TypeError:
            continue
        _add_root(path_obj, None)

    # Sort roots to prefer longer (more specific) paths which ensures that a
    # package nested inside another namespace (``menace_sandbox.vector_service``)
    # wins over broader prefixes such as the workspace directory.
    roots.sort(key=lambda item: (-len(item[0].parts), item[1] or ""))
    return roots


def _module_name_from_path(path: Path) -> str:
    """Infer a stable module name for *path*.

    Registry persistence stores raw filesystem paths on Windows which breaks the
    traditional dotted-module import flow.  This helper reconstructs the
    canonical module name by combining two strategies:

    * Walk up the directory hierarchy while ``__init__`` modules are present to
      honour regular packages.
    * Fall back to already-imported package roots and ``sys.path`` entries to
      support namespace packages and editable installs.

    The returned name deliberately favours ``menace_sandbox`` prefixed modules so
    the sandbox behaves consistently between Windows and Linux environments.
    """

    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = path.absolute()

    suffixless = resolved.with_suffix("") if resolved.suffix else resolved
    leaf = suffixless.name

    package_parts: list[str] = []
    current = suffixless.parent
    while current and current != current.parent:
        init_py = current / "__init__.py"
        init_pyc = current / "__init__.pyc"
        if init_py.exists() or init_pyc.exists():
            package_parts.append(current.name)
            current = current.parent
            continue
        break

    if package_parts:
        package_parts.reverse()
        if leaf == "__init__":
            candidate = ".".join(package_parts)
        else:
            candidate = ".".join((*package_parts, leaf))
        if candidate:
            return candidate

    best_candidate: str | None = None
    best_score: tuple[int, int, int, int] = (-1, -1, -1, -1)
    for root, package_name in _iter_module_search_roots():
        try:
            relative = suffixless.relative_to(root)
        except ValueError:
            continue
        parts = list(relative.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts and not package_name:
            continue
        if package_name:
            candidate_parts = [package_name, *parts]
        else:
            candidate_parts = parts
        candidate = ".".join(candidate_parts)
        if not candidate:
            continue
        score = (
            2 if candidate.startswith("menace_sandbox") else 1 if candidate.startswith("menace") else 0,
            candidate.count("."),
            len(candidate_parts),
            len(candidate),
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate:
        return best_candidate

    fallback_parts = [part for part in suffixless.parts if part not in {"", os.sep, os.altsep}]
    if fallback_parts and fallback_parts[-1] == "__init__":
        fallback_parts = fallback_parts[:-1]
    if not fallback_parts:
        fallback_parts = [leaf or resolved.stem or resolved.name]
    fallback = ".".join(part for part in fallback_parts if part)
    return fallback.replace("/", ".").replace("\\", ".") or (leaf or resolved.stem or resolved.name)


def _git_status_for_path(path: Path) -> str | None:
    """Return ``git status --porcelain`` output for ``path`` when available.

    This helper keeps :meth:`BotRegistry.update_bot` resilient when the sandbox
    runs outside a git checkout (for example when unpacked from an archive).
    ``git`` commands exit with status 128 in that scenario which previously
    produced noisy error logs and interrupted hot swaps.  ``None`` is returned
    when the path is not part of a work tree or when git itself is missing so
    callers can silently skip manual change detection.
    """

    candidate = path if path.is_dir() else path.parent
    try:
        cwd = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        cwd = candidate.absolute()
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None

    if probe.stdout.strip().lower() not in {"true", "1"}:
        return None

    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(path)],
            cwd=str(cwd),
        ).decode()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None


def _module_spec_from_path(module_path: str) -> tuple[str, dict[str, object]]:
    """Return an importable module name and ``spec_from_file_location`` kwargs."""

    path_obj = Path(module_path)
    module_name = _module_name_from_path(path_obj)
    kwargs: dict[str, object] = {}
    if path_obj.name.startswith("__init__.py") or path_obj.name == "__init__":
        kwargs["submodule_search_locations"] = [str(path_obj.parent)]
    return module_name, kwargs


@dataclass(slots=True)
class _TransientErrorState:
    """Track repeated transient import failures for a single bot."""

    signature: tuple[str, str]
    count: int = 1
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    first_seen_wallclock: float = field(default_factory=time.time)
    last_seen_wallclock: float = field(default_factory=time.time)
    total_count: int = 1
    signature_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise bookkeeping so subsequent updates can rely on the aggregated
        # counters.  ``count`` always reflects the cumulative occurrences for the
        # current signature rather than only consecutive repeats which ensures
        # noisy error messages (for example those embedding file paths) are still
        # treated as the same failure mode.
        self.signature_counts[self.signature] = self.count

    def increment(self, signature: tuple[str, str]) -> "_TransientErrorState":
        self.last_seen = time.monotonic()
        self.last_seen_wallclock = time.time()
        self.total_count += 1
        aggregate = self.signature_counts.get(signature, 0) + 1
        self.signature_counts[signature] = aggregate
        if signature == self.signature:
            self.count = aggregate
        else:
            self.signature = signature
            self.count = aggregate
        return self

    @property
    def unique_signatures(self) -> int:
        return len(self.signature_counts)

    def window_seconds(self) -> float:
        return self.last_seen - self.first_seen

    def should_disable(
        self,
        *,
        max_signature_repeats: int,
        max_total_repeats: int,
        max_window: float,
    ) -> bool:
        if self.count >= max_signature_repeats:
            return True
        if self.total_count >= max_total_repeats:
            return True
        if max_window > 0 and self.window_seconds() >= max_window:
            return True
        return False


def _normalise_exception_message(exc: BaseException) -> str:
    """Return ``exc``'s message with volatile runtime details stripped."""

    message = str(exc)
    if not message:
        return ""

    def _strip_token(match: re.Match[str]) -> str:
        token = match.group("token")
        quote = match.group(1)
        if _HEX_ADDRESS_RE.fullmatch(token):
            return f"{quote}0x<addr>{quote}"
        if _is_probable_filesystem_path(token):
            return f"{quote}<path>{quote}"
        return match.group(0)

    # Normalise quoted tokens first so path heuristics see unescaped values.
    normalised = _QUOTED_TOKEN_RE.sub(_strip_token, message)

    def _replace_posix(match: re.Match[str]) -> str:
        token = match.group(0)
        return "<path>" if _is_probable_filesystem_path(token) else token

    # Replace explicit filesystem paths that were not quoted.
    normalised = _WINDOWS_ABS_PATH_RE.sub("<path>", normalised)
    normalised = _POSIX_ABS_PATH_RE.sub(_replace_posix, normalised)
    normalised = _HEX_ADDRESS_RE.sub("0x<addr>", normalised)

    return normalised


def _exception_signature(exc: BaseException) -> tuple[str, str]:
    """Return a stable signature describing *exc* for retry heuristics."""

    return (exc.__class__.__name__, _normalise_exception_message(exc))

def _default_thresholds() -> SimpleNamespace:
    """Return conservative default self-coding thresholds.

    The object is recreated on each call to prevent downstream mutation from
    affecting other callers.
    """

    return SimpleNamespace(
        roi_drop=-0.1,
        error_increase=1.0,
        test_failure_increase=0.0,
        patch_success_drop=-0.2,
    )


def _resolved_module_exists(module_name: str | None) -> bool:
    """Return ``True`` when ``module_name`` can be resolved locally."""

    if not module_name:
        return False
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):  # pragma: no cover - best effort
        spec = None
    if spec is not None:
        return True

    parts = module_name.replace(".", os.sep)
    base = Path(__file__).resolve().parent
    for candidate in (base / f"{parts}.py", base / parts / "__init__.py"):
        if candidate.exists():
            return True
    return False


def _missing_module_name(exc: ModuleNotFoundError) -> str | None:
    name = getattr(exc, "name", None)
    if name:
        return name
    if exc.args:
        msg = str(exc)
        prefix = "No module named "
        if msg.startswith(prefix):
            remainder = msg[len(prefix) :].strip("'\"")
            if remainder:
                return remainder
    return None


class SelfCodingUnavailableError(RuntimeError):
    """Raised when the self-coding stack cannot be initialised safely."""

    def __init__(
        self,
        message: str,
        *,
        missing: Iterable[str] | None = None,
        missing_resources: Iterable[str] | None = None,
    ) -> None:
        super().__init__(message)
        module_set = {
            m for m in (missing or ()) if isinstance(m, str) and m.strip()
        }
        resource_set = {
            r for r in (missing_resources or ()) if isinstance(r, str) and r.strip()
        }
        self.missing_modules: tuple[str, ...] = tuple(sorted(module_set))
        self.missing_resources: tuple[str, ...] = tuple(sorted(resource_set))

    @property
    def missing_dependencies(self) -> tuple[str, ...]:
        """Return the union of missing modules and runtime resources."""

        if not self.missing_modules and not self.missing_resources:
            return tuple()
        combined = set(self.missing_modules)
        combined.update(self.missing_resources)
        return tuple(sorted(combined))


class _SelfCodingComponents(NamedTuple):
    internalize_coding_bot: Any
    engine_cls: Any
    pipeline_cls: Any
    data_bot_cls: Any
    code_db_cls: Any
    memory_manager_cls: Any
    context_builder_factory: Any


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Yield *exc* and each exception in its cause/context chain safely."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
# Windows frequently reports missing optional dependencies via ``ImportError``
# messages that lack the ``name`` attribute.  Those errors typically follow the
# "DLL load failed while importing <module>" pattern which previously slipped
# through ``_collect_missing_modules`` and forced the registry into repeated
# transient retries.  Normalising the message here lets the caller surface a
# proper ``SelfCodingUnavailableError`` instead of looping indefinitely.
_DLL_LOAD_FAILED_RE = re.compile(r"while importing ([^:]+)")
_DLL_LOAD_FAILED_PREFIX_RE = re.compile(
    r"DLL load failed(?:[^:]*):\s*(?:The specified module could not be found\.?|The specified procedure could not be found\.?|%1 is not a valid Win32 application\.?|error code \d+)",
    re.IGNORECASE,
)
_CIRCULAR_IMPORT_RE = re.compile(
    r"cannot import name ['\"](?P<symbol>[^'\"]+)['\"] from"
    r"(?: partially initialized module)? ['\"](?P<module>[^'\"]+)['\"]",
    re.IGNORECASE,
)
_PARTIAL_MODULE_RE = re.compile(
    r"partially initialized module ['\"](?P<module>[^'\"]+)['\"]",
    re.IGNORECASE,
)
_CIRCULAR_HINT_RE = re.compile("circular import", re.IGNORECASE)
_WINDOWS_DLL_TOKEN_RE = re.compile(
    r"['\"](?P<token>[\w.-]+\.(?:dll|pyd))['\"]",
    re.IGNORECASE,
)
_WINDOWS_LIBRARY_HINT_RE = re.compile(
    r"(?:module|library) ['\"](?P<module>[^'\"\r\n]+)['\"] (?:could not|not) be found",
    re.IGNORECASE,
)
_WINDOWS_COULD_NOT_FIND_RE = re.compile(
    r"could not find module ['\"](?P<module>[^'\"\r\n]+)['\"]",
    re.IGNORECASE,
)
_WINDOWS_ERROR_LOADING_RE = re.compile(
    r"Error loading ['\"](?P<module>[^'\"]+)['\"]",
    re.IGNORECASE,
)
_WINDOWS_FATAL_DLL_HINTS: tuple[str, ...] = (
    "dll load failed",
    "the specified module could not be found",
    "the specified procedure could not be found",
    "winerror",
    "%1 is not a valid win32 application",
)
_MODULE_REQUIRED_RE = re.compile(
    r"(?P<module>[A-Za-z0-9_.-]+)\s+(?:module\s+)?is\s+required(?:\s+for|\s+by)?\s+(?P<context>[A-Za-z0-9_.-]+)?",
    re.IGNORECASE,
)
_PIP_INSTALL_RE = re.compile(
    r"pip\s+install\s+(?P<package>[A-Za-z0-9_.-]+(?:\[[^\]]+\])?)",
    re.IGNORECASE,
)
_MODULE_GRAPH_INIT_RE = re.compile(
    r"module graph still initialis(?:ing|zing)",
    re.IGNORECASE,
)
_TRACEBACK_IGNORED_PREFIXES: tuple[str, ...] = (
    "tests",
    "menace_sandbox.tests",
    "menace.tests",
)
_TRACEBACK_IGNORED_MODULES: frozenset[str] = frozenset(
    {
        "bot_registry",
        "menace.bot_registry",
        "menace_sandbox.bot_registry",
    }
)
_CANNOT_IMPORT_RE = re.compile(
    r"cannot import name ['\"](?P<symbol>[^'\"]+)['\"] from "
    r"(?:partially init(?:ialized|ialised) module |module )?['\"](?P<module>[^'\"]+)['\"]",
    re.IGNORECASE,
)
_IMPORT_HALTED_RE = re.compile(
    r"import of ['\"](?P<module>[^'\"]+)['\"] halted",
    re.IGNORECASE,
)
_RESOURCE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<token>[A-Za-z0-9_.-]+\.(?:db|json|ya?ml|pkl|bin|dat))",
    re.IGNORECASE,
)
_RESOURCE_LIST_PREFIXES: tuple[str, ...] = (
    "Missing required",
    "Context builder database paths are not files",
)
_RESOURCE_ERRNOS: tuple[int, ...] = (
    errno.ENOENT,
    errno.EACCES,
    errno.ENOTDIR,
    errno.EISDIR,
)

# High level ImportError messages occasionally hide the underlying module
# name which prevents ``_collect_missing_modules`` from surfacing actionable
# diagnostics.  ``QuickFixEngine`` is a common example on Windows where the
# optional quick-fix extension is missing; the resulting ImportError only
# references the class name or installation hint.  Mapping these textual
# signatures to their canonical modules keeps the dependency inference
# deterministic and avoids repeated transient retries during sandbox start-up.
_KNOWN_DEPENDENCY_HINTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\bquick[_\s-]?fix[_\s-]?engine\b", re.IGNORECASE),
        ("quick_fix_engine",),
    ),
    (
        re.compile(r"menace\[quickfix\]", re.IGNORECASE),
        ("quick_fix_engine",),
    ),
    (
        re.compile(r"\bcontext_builder_util\b", re.IGNORECASE),
        ("context_builder_util",),
    ),
)


def _normalise_module_aliases(module: str) -> set[str]:
    """Return a set of candidate module names for cache invalidation."""

    aliases = {module}
    if module.startswith("menace_sandbox."):
        aliases.add(module[len("menace_sandbox."):])
    if module.startswith("menace."):
        aliases.add(module[len("menace."):])
    if "." in module:
        parts = module.split(".")
        cumulative = []
        for part in parts:
            cumulative.append(part)
            aliases.add(".".join(cumulative))
    return {alias for alias in aliases if alias}


def _is_traceback_only_module(name: str) -> bool:
    """Return ``True`` when *name* only references test or bootstrap modules."""

    if not name:
        return False
    candidate = name.strip()
    if not candidate:
        return False
    if candidate in _TRACEBACK_IGNORED_MODULES:
        return True
    for prefix in _TRACEBACK_IGNORED_PREFIXES:
        if candidate == prefix or candidate.startswith(prefix + "."):
            return True
    return False


def _extract_partial_modules(exc: BaseException) -> set[str]:
    """Return modules referenced by partial import or circular import errors."""

    modules: set[str] = set()
    for item in _iter_exception_chain(exc):
        if not isinstance(item, ImportError):
            continue
        message = str(item)
        match = _PARTIAL_MODULE_RE.search(message)
        if match:
            modules.add(match.group("module"))
        match = _CIRCULAR_IMPORT_RE.search(message)
        if match:
            modules.add(match.group("module"))
    return modules


def _purge_partial_modules(exc: BaseException) -> list[str]:
    """Remove partially initialised modules from :mod:`sys.modules`.

    When Windows interleaves multiple concurrent imports the interpreter often
    leaves behind module placeholders whose ``__spec__`` is ``None``.  Subsequent
    imports then continue to observe the incomplete module and raise
    ``ImportError`` with "partially initialized" hints.  Clearing the affected
    entries gives the loader a clean slate on the next retry instead of
    repeatedly cycling through the half-imported module.
    """

    candidates = _extract_partial_modules(exc)
    removed: list[str] = []
    for module in candidates:
        for alias in _normalise_module_aliases(module):
            if alias in {"menace_sandbox", "menace"}:
                continue
            if alias in sys.modules:
                removed.append(alias)
                sys.modules.pop(alias, None)
    if removed:
        logger.debug(
            "purged partially initialised modules after import error: %s",
            ", ".join(sorted(set(removed))),
        )
    return removed

# Import failures originating from these modules indicate that the self-coding
# runtime itself is unavailable rather than a temporary race with the module
# graph.  Treating them as non-transient prevents the registry from retrying
# indefinitely on Windows when compiled extensions such as ``quick_fix_engine``
# are missing supporting DLLs.
_INTERNAL_SELF_CODING_MODULES: tuple[str, ...] = (
    "quick_fix_engine",
    "self_coding_manager",
    "model_automation_pipeline",
    "code_database",
    "gpt_memory",
    "self_coding_thresholds",
)


def _normalise_token(token: str | None) -> set[str]:
    """Return a normalised set of candidate module names for ``token``."""

    if not token:
        return set()

    cleaned = token.strip().strip("'\"")
    if not cleaned:
        return set()

    candidates: set[str] = set()
    base = cleaned.replace("\\", "/").strip()
    if not base:
        return set()

    # Handle filesystem paths or DLL names by using their stem in addition to the
    # raw representation.  ``Path.stem`` removes extensions such as ``.pyd``
    # while leaving dotted module names untouched.
    path = Path(base)
    stem = path.stem
    if stem and stem != base:
        candidates.add(stem)
    if base:
        candidates.add(base)
    if stem:
        candidates.add(stem.replace("-", "_"))
        if "." in stem:
            candidates.add(stem.split(".", 1)[0])
    if base:
        cleaned_base = base.replace("-", "_")
        if cleaned_base != base:
            candidates.add(cleaned_base)
        if "." in base:
            candidates.add(base.split(".", 1)[0])
    enriched: set[str] = set()
    for token in tuple(candidates):
        if token.lower().endswith((".dll", ".pyd", ".so", ".dylib")):
            enriched.add(Path(token).stem)
    if enriched:
        candidates.update(enriched)
    return {candidate for candidate in candidates if candidate}


def _collect_missing_modules(exc: BaseException) -> set[str]:
    """Return a best-effort set of import targets missing for *exc*."""

    missing: set[str] = set()
    module_graph_unstable = False
    for item in _iter_exception_chain(exc):
        message = str(item)
        if _MODULE_GRAPH_INIT_RE.search(message):
            module_graph_unstable = True
            continue
        cannot_import = _CANNOT_IMPORT_RE.search(message)
        if cannot_import:
            module_hint = cannot_import.group("module")
            symbol_hint = cannot_import.group("symbol")
            missing.update(_normalise_module_aliases(module_hint))
            missing.update(_normalise_module_aliases(symbol_hint))
        modules_attr = getattr(item, "missing_modules", None)
        resources_attr = getattr(item, "missing_resources", None)
        dependencies_attr = getattr(item, "missing_dependencies", None)
        if modules_attr:
            for module in modules_attr:
                if isinstance(module, str) and module.strip():
                    missing.add(module.strip())
        if resources_attr:
            for resource in resources_attr:
                if isinstance(resource, str) and resource.strip():
                    missing.add(resource.strip())
        if dependencies_attr:
            for dependency in dependencies_attr:
                if isinstance(dependency, str) and dependency.strip():
                    missing.add(dependency.strip())
        if isinstance(item, ModuleNotFoundError):
            name = _missing_module_name(item)
            if name:
                missing.add(name)
            # ``ModuleNotFoundError`` inherits from ``ImportError`` which means we
            # deliberately fall through instead of ``continue``-ing here.  Windows
            # frequently raises ``ModuleNotFoundError`` instances without the
            # ``name`` attribute populated when a transitive DLL import fails.
            # Allowing the execution to reach the ``ImportError`` branch lets us
            # analyse the error message and ``path`` attribute so that
            # ``_collect_missing_modules`` can still surface a deterministic
            # dependency hint.
        if isinstance(item, ImportError):
            if getattr(item, "name", None):
                missing.add(str(item.name))
            match = _MISSING_MODULE_RE.search(message)
            if match:
                missing.add(match.group(1))
            else:
                dll_match = _DLL_LOAD_FAILED_RE.search(message)
                if dll_match:
                    missing.add(dll_match.group(1))
            if "QuickFixEngine failed to initialise" in message:
                missing.update({"quick_fix_engine", "menace_sandbox.quick_fix_engine"})
            path_tokens: set[str] = set()
            path_hint = getattr(item, "path", None)
            if isinstance(path_hint, str) and _is_probable_filesystem_path(path_hint):
                path_tokens.update(_normalise_token(path_hint))
            filename_hint = getattr(item, "filename", None)
            if isinstance(filename_hint, str) and _is_probable_filesystem_path(
                filename_hint
            ):
                path_tokens.update(_normalise_token(filename_hint))
            for arg in getattr(item, "args", ()):  # pragma: no cover - defensive
                if isinstance(arg, str) and _is_probable_filesystem_path(arg):
                    path_tokens.update(_normalise_token(arg))
            if _DLL_LOAD_FAILED_PREFIX_RE.search(message):
                # ``ImportError`` instances raised by ``importlib`` on Windows
                # frequently omit the module name altogether when the failure
                # originates from a transitive DLL dependency.  Falling back to
                # the ``path`` attribute as well as any quoted DLL names in the
                # error message allows us to surface a deterministic
                # ``SelfCodingUnavailableError`` instead of looping on
                # transient retries.
                if not path_tokens:
                    path_tokens.update(_normalise_token(getattr(item, "path", None)))
                for match in _WINDOWS_DLL_TOKEN_RE.finditer(message):
                    path_tokens.update(_normalise_token(match.group("token")))
                lib_match = _WINDOWS_LIBRARY_HINT_RE.search(message)
                if lib_match:
                    path_tokens.update(_normalise_token(lib_match.group("module")))
                cause = getattr(item, "__cause__", None)
                if isinstance(cause, OSError):
                    path_tokens.update(_normalise_token(getattr(cause, "filename", None)))
            error_load_match = _WINDOWS_ERROR_LOADING_RE.search(message)
            if error_load_match:
                path_tokens.update(_normalise_token(error_load_match.group("module")))
            if path_tokens:
                missing.update(path_tokens)
            ctor_match = _WINDOWS_COULD_NOT_FIND_RE.search(message)
            if ctor_match:
                path_tokens.update(_normalise_token(ctor_match.group("module")))
                missing.update(path_tokens)
        for pattern, modules in _KNOWN_DEPENDENCY_HINTS:
            if pattern.search(message):
                missing.update(modules)
        for match in _MODULE_REQUIRED_RE.finditer(message):
            module_hint = match.group("module")
            context_hint = match.group("context")
            if module_hint:
                missing.update(_normalise_module_aliases(module_hint))
            if context_hint:
                missing.update(_normalise_module_aliases(context_hint))
        for match in _PIP_INSTALL_RE.finditer(message):
            package = match.group("package")
            if not package:
                continue
            canonical = package.split("[", 1)[0]
            missing.update(_normalise_module_aliases(canonical))
        if "QuickFixEngine is required" in message:
            missing.update({"quick_fix_engine", "menace_sandbox.quick_fix_engine"})
        circular_match = _CIRCULAR_IMPORT_RE.search(message)
        if circular_match:
            missing.add(circular_match.group("module"))
        partial_match = _PARTIAL_MODULE_RE.search(message)
        if partial_match:
            missing.add(partial_match.group("module"))
        halted_match = _IMPORT_HALTED_RE.search(message)
        if halted_match:
            missing.update(
                _normalise_module_aliases(halted_match.group("module"))
            )
    if not missing and not module_graph_unstable:
        missing.update(_infer_modules_from_traceback(exc))
    return missing


def _derive_import_error_hints(exc: BaseException) -> set[str]:
    """Return candidate module names implicated by an unresolved import error."""

    hints: set[str] = set()
    for item in _iter_exception_chain(exc):
        if not isinstance(item, ImportError):
            continue
        name_attr = getattr(item, "name", None)
        if isinstance(name_attr, str) and name_attr.strip():
            hints.update(_normalise_module_aliases(name_attr.strip()))
        path_attr = getattr(item, "path", None)
        if isinstance(path_attr, str) and _is_probable_filesystem_path(path_attr):
            hints.update(_normalise_token(path_attr))
        filename_attr = getattr(item, "filename", None)
        if isinstance(filename_attr, str) and _is_probable_filesystem_path(
            filename_attr
        ):
            hints.update(_normalise_token(filename_attr))
        for arg in getattr(item, "args", ()):  # pragma: no cover - defensive
            if not isinstance(arg, str):
                continue
            for regex in (
                _WINDOWS_DLL_TOKEN_RE,
                _WINDOWS_ERROR_LOADING_RE,
                _WINDOWS_COULD_NOT_FIND_RE,
            ):
                match = regex.search(arg)
                if not match:
                    continue
                token = match.groupdict().get("token") or match.groupdict().get(
                    "module"
                )
                if token:
                    hints.update(_normalise_token(token))
    hints.update(_infer_modules_from_traceback(exc))
    hints = {hint for hint in hints if not _is_traceback_only_module(hint)}
    if not hints or all(_is_traceback_only_module(hint) for hint in hints):
        hints.add("self_coding_runtime")
    return hints


def _infer_modules_from_traceback(exc: BaseException) -> set[str]:
    """Infer sandbox module names from traceback frames when messages are vague."""

    inferred: set[str] = set()
    repo_root = Path(__file__).resolve().parent

    for item in _iter_exception_chain(exc):
        tb = getattr(item, "__traceback__", None)
        while tb is not None:
            frame = tb.tb_frame
            module_name = frame.f_globals.get("__name__")
            if isinstance(module_name, str) and module_name.startswith(
                ("menace_sandbox.", "menace.")
            ):
                inferred.update(_normalise_module_aliases(module_name))

            filename = frame.f_code.co_filename
            if filename:
                try:
                    path = Path(filename)
                except (TypeError, ValueError):
                    path = None
                if path is not None:
                    try:
                        resolved = (
                            path if path.is_absolute() else (repo_root / path)
                        ).resolve(strict=False)
                    except (RuntimeError, OSError):
                        resolved = path
                    try:
                        relative = resolved.relative_to(repo_root)
                    except ValueError:
                        pass
                    else:
                        module_hint = ".".join(relative.with_suffix("").parts)
                        if module_hint.startswith(("menace_sandbox.", "menace.")):
                            inferred.update(_normalise_module_aliases(module_hint))
            tb = tb.tb_next

    inferred.discard("menace_sandbox")
    inferred.discard("menace")
    return inferred


def _collect_missing_resources(exc: BaseException) -> set[str]:
    """Return runtime resources inferred from filesystem related errors."""

    resources: set[str] = set()
    for item in _iter_exception_chain(exc):
        if isinstance(item, FileNotFoundError):
            resources.update(_normalise_token(getattr(item, "filename", None)))
            resources.update(_normalise_token(getattr(item, "filename2", None)))
        if isinstance(item, OSError):
            err_no = getattr(item, "errno", None)
            if err_no in _RESOURCE_ERRNOS:
                resources.update(_normalise_token(getattr(item, "filename", None)))
                resources.update(_normalise_token(getattr(item, "filename2", None)))
        message = str(item)
        for prefix in _RESOURCE_LIST_PREFIXES:
            if prefix in message:
                try:
                    _, remainder = message.split(":", 1)
                except ValueError:
                    remainder = ""
                for token in remainder.split(","):
                    resources.update(_normalise_token(token))
        for match in _RESOURCE_TOKEN_RE.finditer(message):
            resources.update(_normalise_token(match.group("token")))
    return {resource for resource in resources if resource}


def _is_transient_internalization_error(exc: Exception) -> bool:
    """Return ``True`` for import errors that are likely transient.

    ``@self_coding_managed`` decorators execute during module import which means
    dependencies may not yet be fully initialised.  When
    :func:`register_bot` attempts to internalise a bot in that window the eager
    imports can fail with ``ModuleNotFoundError`` or the more generic
    ``ImportError`` that reports a "partially initialized" module.  Those
    situations typically resolve once the module graph finishes loading, so we
    treat them as transient and defer the internalisation retry to the
    background scan instead of hard failing the import.
    """

    message = str(exc)
    lowered = message.lower()
    has_partial_hint = "partially init" in lowered or _MODULE_GRAPH_INIT_RE.search(lowered)

    if _CIRCULAR_HINT_RE.search(message):
        return False

    cannot_import = _CANNOT_IMPORT_RE.search(message)
    if cannot_import:
        if has_partial_hint:
            return True
        return False

    if isinstance(exc, ModuleNotFoundError):
        inferred_missing = _collect_missing_modules(exc)
        if inferred_missing:
            return False

        module_name = _missing_module_name(exc)
        if module_name is None:
            lowered = message.lower()
            if any(hint in lowered for hint in _WINDOWS_FATAL_DLL_HINTS):
                return False
            path_hint = getattr(exc, "path", None)
            if isinstance(path_hint, str) and _is_probable_filesystem_path(path_hint):
                return False
            filename_hint = getattr(exc, "filename", None)
            if isinstance(filename_hint, str) and _is_probable_filesystem_path(
                filename_hint
            ):
                return False
            for arg in getattr(exc, "args", ()):  # pragma: no cover - defensive
                if isinstance(arg, str):
                    if _WINDOWS_DLL_TOKEN_RE.search(arg) or _WINDOWS_ERROR_LOADING_RE.search(arg):
                        return False
                    if _is_probable_filesystem_path(arg):
                        return False
            return True
        root_name = module_name.split(".", 1)[0]
        if module_name.startswith(("menace_sandbox.", "menace.")) or root_name in _INTERNAL_SELF_CODING_MODULES:
            return False
        return _resolved_module_exists(module_name)
    if isinstance(exc, ImportError) and has_partial_hint:
        return True
    return False


def _load_self_coding_thresholds(name: str) -> Any:
    """Return self-coding thresholds while tolerating missing dependencies.

    ``self_coding_thresholds`` has an extensive dependency surface that often
    includes optional packages (for example :mod:`pydantic`).  When those
    packages are unavailable – a common scenario on fresh Windows
    installations – importing the module during bot internalisation raises a
    :class:`ModuleNotFoundError`.  Historically that exception bubbled up and
    was interpreted as a transient error which caused the registry to retry the
    import indefinitely, effectively stalling the sandbox start-up sequence.

    Instead we now attempt to import ``get_thresholds`` lazily and fall back to
    a conservative static configuration when either the import itself or the
    threshold lookup fails.  This keeps the sandbox responsive whilst clearly
    logging the degraded behaviour so operators can install the missing
    dependencies at their convenience.
    """

    try:
        from .self_coding_thresholds import get_thresholds
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        logger.warning(
            "self_coding_thresholds unavailable for %s; using built-in defaults", name
        )
        logger.debug("self_coding_thresholds import failure: %s", exc, exc_info=True)
        return _default_thresholds()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to import self_coding_thresholds for %s; using built-in defaults", name
        )
        logger.debug("self_coding_thresholds unexpected import error", exc_info=True)
        return _default_thresholds()

    try:
        return get_thresholds(name)
    except Exception as exc:  # pragma: no cover - configuration failure
        logger.warning(
            "Unable to load self-coding thresholds for %s; reverting to defaults: %s",
            name,
            exc,
        )
        logger.debug(
            "self_coding_thresholds.get_thresholds raised an exception",
            exc_info=True,
        )
        return _default_thresholds()

def _extend_workflow_tests(target: list[str], seen: set[str], value: Any) -> None:
    for item in normalize_workflow_tests(value):
        if item and item not in seen:
            target.append(item)
            seen.add(item)


def _collect_workflow_tests(
    bot_name: str,
    *,
    registry: "BotRegistry | None" = None,
    settings: SandboxSettings | None = None,
) -> list[str]:
    tests: list[str] = []
    seen: set[str] = set()

    if registry is not None:
        graph = getattr(registry, "graph", None)
        lock = getattr(registry, "_lock", None)
        if lock is not None:
            try:
                with lock:
                    if graph is not None and bot_name in graph:
                        node = graph.nodes[bot_name]
                        for key in (
                            "workflow_tests",
                            "workflow_pytest_args",
                            "pytest_args",
                        ):
                            _extend_workflow_tests(tests, seen, node.get(key))
            except Exception:
                logger.debug("workflow test lookup failed for bot %s", bot_name, exc_info=True)
        elif graph is not None and bot_name in graph:
            node = graph.nodes[bot_name]
            for key in ("workflow_tests", "workflow_pytest_args", "pytest_args"):
                _extend_workflow_tests(tests, seen, node.get(key))

    resolved_settings = settings
    if resolved_settings is None:
        try:
            resolved_settings = SandboxSettings()
        except Exception:
            resolved_settings = None

    if resolved_settings:
        thresholds = getattr(resolved_settings, "bot_thresholds", {})
        if isinstance(thresholds, dict):
            bot_cfg = thresholds.get(bot_name)
            if bot_cfg is not None:
                _extend_workflow_tests(tests, seen, getattr(bot_cfg, "workflow_tests", None))
            default_cfg = thresholds.get("default")
            if default_cfg is not None:
                _extend_workflow_tests(
                    tests,
                    seen,
                    getattr(default_cfg, "workflow_tests", None),
                )

    for key in (bot_name, None):
        try:
            cfg = threshold_service.load(key, resolved_settings)
        except Exception:
            cfg = None
        if cfg is not None:
            _extend_workflow_tests(
                tests,
                seen,
                getattr(cfg, "workflow_tests", None),
            )

    return tests


class BotRegistry:
    """Store connections between bots using a directed graph."""

    _BROKEN_PATCHES: ClassVar[dict[str, set[int]]] = {
        "TaskHandoffBot": {
            -2020412472,
            -3989464708,
        },
    }
    _PATCH_STATUS_KEY: ClassVar[str] = "provenance_patch_status"

    def __init__(
        self,
        *,
        persist: Optional[Path | str] = None,
        event_bus: Optional["UnifiedEventBus"] = None,
    ) -> None:
        self.graph = nx.DiGraph()
        self.modules: Dict[str, str] = {}
        self.persist_path = Path(persist) if persist else None
        # Default to the shared event bus so all registries participate in the
        # same publish/subscribe channel unless explicitly overridden.
        self.event_bus = event_bus or _SHARED_EVENT_BUS
        self.heartbeats: Dict[str, float] = {}
        self.interactions_meta: List[Dict[str, object]] = []
        self._lock = threading.RLock()
        self._internalization_retry_attempts: Dict[str, int] = {}
        self._internalization_retry_handles: Dict[str, threading.Timer] = {}
        self._transient_error_state: Dict[str, _TransientErrorState] = {}
        self._max_internalization_retries = 5
        self._max_transient_error_signature_repeats = 3
        # Import failures on Windows often alternate between several different
        # signatures (for example reporting individual DLL issues).  Track the
        # overall failure pressure so we can stop retrying after a reasonable
        # number of attempts even when the exact message fluctuates.
        self._max_transient_error_total_repeats = 12
        self._transient_error_grace_period = 45.0
        # Windows command prompt environments frequently import sandbox modules
        # noticeably slower than their POSIX counterparts which can lead to
        # eager internalisation attempts racing partially initialised modules.
        # Starting retries after a short, configurable delay gives the module
        # graph time to settle without blocking the caller thread.
        self._initial_internalization_delay = 0.75
        if self.persist_path and self.persist_path.exists():
            try:
                self.load(self.persist_path)
            except Exception as exc:
                logger.error(
                    "Failed to load bot registry from %s: %s", self.persist_path, exc
                )
        try:
            self.schedule_unmanaged_scan()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to schedule unmanaged bot scan")

    def hot_swap_active(self) -> bool:
        """Return ``True`` when this registry is performing a hot swap import."""

        return _hot_swap_active()

    @classmethod
    def _is_blacklisted_patch(cls, name: str, patch_id: int) -> bool:
        patches = cls._BROKEN_PATCHES.get(name)
        return bool(patches and patch_id in patches)

    def _get_patch_status_entry(
        self, node: MutableMapping[str, Any], patch_id: int
    ) -> Optional[Dict[str, Any]]:
        status_map = node.get(self._PATCH_STATUS_KEY)
        if not isinstance(status_map, dict):
            return None
        entry = status_map.get(str(patch_id))
        return entry if isinstance(entry, dict) else None

    def _commit_already_applied(
        self, node: MutableMapping[str, Any], commit: str | None
    ) -> bool:
        """Return ``True`` when ``commit`` is recorded as successfully applied."""

        if not commit:
            return False

        status_map = node.get(self._PATCH_STATUS_KEY)
        if not isinstance(status_map, dict):
            return False

        for entry in status_map.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != "applied":
                continue
            stored_commit = entry.get("commit")
            if isinstance(stored_commit, str) and stored_commit == commit:
                return True
        return False

    def _set_patch_status(
        self,
        target: MutableMapping[str, Any],
        patch_id: int,
        commit: str,
        status: str,
        *,
        reason: str | None = None,
        extra: Optional[Dict[str, Any]] = None,
        success: bool | None = None,
    ) -> None:
        status_map = target.get(self._PATCH_STATUS_KEY)
        if isinstance(status_map, dict):
            updated = dict(status_map)
        else:
            updated = {}
        entry: Dict[str, Any] = {
            "status": status,
            "commit": commit,
            "timestamp": time.time(),
        }
        if reason:
            entry["reason"] = reason
        if extra:
            entry.update(extra)
        if success is None:
            if status == "applied":
                success = True
            elif status in {"failed", "blocked"}:
                success = False
        if success is not None:
            entry["patch_success"] = success
        updated[str(patch_id)] = entry
        target[self._PATCH_STATUS_KEY] = updated

    def _cancel_internalization_retry(self, name: str) -> None:
        """Best-effort cancellation for a pending internalisation retry timer."""

        handle = self._internalization_retry_handles.pop(name, None)
        if handle is None:
            return
        try:
            handle.cancel()
        except Exception:  # pragma: no cover - timer cleanup best effort
            logger.debug(
                "failed to cancel internalization retry timer for %s", name, exc_info=True
            )

    def _schedule_internalization_retry(
        self,
        name: str,
        *,
        delay: float | None = None,
    ) -> None:
        """Schedule an asynchronous retry for ``internalize_coding_bot``."""

        with self._lock:
            existing = self._internalization_retry_handles.get(name)
            if existing is not None and getattr(existing, "is_alive", lambda: False)():
                return
            attempts = self._internalization_retry_attempts.setdefault(name, 0)
            retry_delay = delay if delay is not None else min(5.0, 0.5 * (attempts + 1))
            timer = threading.Timer(retry_delay, self._retry_internalization, args=(name,))
            timer.daemon = True
            self._internalization_retry_handles[name] = timer

        try:
            timer.start()
        except Exception:  # pragma: no cover - timer creation best effort
            logger.exception("failed to start internalization retry timer for %s", name)

    def _register_transient_failure(
        self, name: str, exc: Exception
    ) -> _TransientErrorState:
        """Record a transient failure and return the associated state."""

        signature = _exception_signature(exc)
        with self._lock:
            state = self._transient_error_state.get(name)
            if state is None:
                state = _TransientErrorState(signature=signature)
                self._transient_error_state[name] = state
            else:
                state.increment(signature)
        return state

    def _clear_transient_error_state(self, name: str) -> None:
        """Forget previously recorded transient failures for ``name``."""

        with self._lock:
            self._transient_error_state.pop(name, None)

    def _disable_self_coding_due_to_transient_errors(
        self, name: str, exc: Exception, state: _TransientErrorState
    ) -> None:
        """Disable self-coding after repeated transient import failures."""

        self._record_internalization_blocked(name, exc)

        node = self.graph.nodes.get(name)
        if node is None:
            self._clear_transient_error_state(name)
            return

        missing_modules = sorted(_collect_missing_modules(exc))
        missing_resources = sorted(_collect_missing_resources(exc))
        missing = sorted({*missing_modules, *missing_resources})
        timestamp = datetime.now(timezone.utc).isoformat()
        reason = (
            "self-coding disabled after repeated transient import failures"
            f" ({state.total_count} attempts across {state.unique_signatures}"
            f" signature(s)): {exc}"
        )
        signature_history = [
            {
                "type": sig[0],
                "message": sig[1],
                "count": count,
            }
            for sig, count in sorted(
                state.signature_counts.items(), key=lambda item: item[1], reverse=True
            )
        ]
        node["self_coding_disabled"] = {
            "reason": reason,
            "missing_dependencies": missing,
            "timestamp": timestamp,
            "transient_error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "repeat_count": state.count,
                "total_repeat_count": state.total_count,
                "unique_signatures": state.unique_signatures,
                "first_seen": datetime.fromtimestamp(
                    state.first_seen_wallclock, tz=timezone.utc
                ).isoformat(),
                "last_seen": datetime.fromtimestamp(
                    state.last_seen_wallclock, tz=timezone.utc
                ).isoformat(),
                "window_seconds": round(state.window_seconds(), 3),
                "signature_counts": signature_history,
            },
        }
        if missing_resources:
            node["self_coding_disabled"]["missing_resources"] = missing_resources
        self._clear_transient_error_state(name)

        logger.warning(
            (
                "self-coding disabled for %s after %s transient import failures"
                " across %s signature(s): %s"
            ),
            name,
            state.total_count,
            state.unique_signatures,
            exc,
        )

        if self.event_bus:
            payload = {
                "bot": name,
                "missing": missing,
                "reason": reason,
                "transient_error": node["self_coding_disabled"]["transient_error"],
            }
            if missing_resources:
                payload["resources"] = missing_resources
            try:
                self.event_bus.publish("bot:self_coding_disabled", payload)
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "failed to publish bot:self_coding_disabled event after transient errors",
                    exc_info=True,
                )

    def _record_internalization_blocked(self, name: str, exc: Exception) -> None:
        """Record a non-recoverable internalisation failure for ``name``."""

        node = self.graph.nodes.get(name)
        if node is None:
            return

        node["pending_internalization"] = False
        node["internalization_blocked"] = {
            "error": str(exc),
            "exception": exc.__class__.__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "traceback": "".join(
                traceback.format_exception(exc.__class__, exc, exc.__traceback__)
            ),
        }
        self._internalization_retry_attempts.pop(name, None)
        handle = self._internalization_retry_handles.pop(name, None)
        if handle is not None:
            try:
                handle.cancel()
            except Exception:  # pragma: no cover - timer cleanup best effort
                logger.debug(
                    "failed to cancel retry timer while recording blocked internalization",
                    exc_info=True,
                )

        disable_payload: dict[str, Any] | None = None
        # When we can reliably infer missing modules from the exception chain we
        # proactively disable self-coding.  Windows command prompt executions in
        # particular surface dependency gaps as ``ImportError`` instances during
        # the initial bootstrap which previously left bots stuck in a permanent
        # "internalization blocked" state.  Surfacing the missing dependencies
        # and emitting the ``bot:self_coding_disabled`` event keeps the sandbox
        # responsive while providing operators with actionable diagnostics.
        existing_disabled = node.get("self_coding_disabled")
        existing_missing: set[str] = set()
        existing_resources: set[str] = set()
        if isinstance(existing_disabled, dict):
            existing_missing.update(
                str(item).strip()
                for item in existing_disabled.get("missing_dependencies", ())
                if str(item).strip()
            )
            existing_resources.update(
                str(item).strip()
                for item in existing_disabled.get("missing_resources", ())
                if str(item).strip()
            )

        missing_modules = set(_collect_missing_modules(exc))
        missing_resources = set(_collect_missing_resources(exc))
        dependency_hints = set(_derive_import_error_hints(exc))

        combined_missing = sorted({*existing_missing, *missing_modules, *dependency_hints})
        combined_resources = sorted({*existing_resources, *missing_resources})

        disable_payload = {
            "reason": (
                "self-coding disabled after unrecoverable import failure: "
                f"{exc}"
            ),
            "missing_dependencies": combined_missing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "internalization_blocked",
        }
        if combined_resources:
            disable_payload["missing_resources"] = combined_resources

        node["self_coding_disabled"] = disable_payload

        logger.error(
            "internalization for %s blocked after non-recoverable error: %s",
            name,
            exc,
            exc_info=exc,
        )

        if self.event_bus:
            try:
                self.event_bus.publish(
                    "bot:internalization_blocked",
                    {"bot": name, "error": node["internalization_blocked"]},
                )
                payload = {
                    "bot": name,
                    "missing": disable_payload.get("missing_dependencies"),
                    "reason": disable_payload["reason"],
                    "source": "internalization_blocked",
                }
                if combined_resources:
                    payload["resources"] = combined_resources
                try:
                    self.event_bus.publish("bot:self_coding_disabled", payload)
                except Exception:  # pragma: no cover - best effort
                    logger.debug(
                        "failed to publish bot:self_coding_disabled event after blocked internalization",
                        exc_info=True,
                    )
            except Exception as pub_exc:  # pragma: no cover - best effort
                logger.error(
                    "Failed to publish bot:internalization_blocked event: %s",
                    pub_exc,
                )
        self._clear_transient_error_state(name)

    def _retry_internalization(self, name: str) -> None:
        """Attempt to internalise a bot that previously failed to import."""

        with self._lock:
            self._cancel_internalization_retry(name)

            node = self.graph.nodes.get(name)
            if node is None:
                self._internalization_retry_attempts.pop(name, None)
                return

            if not node.get("is_coding_bot", True):
                # Manual registrations or strict provenance policies can disable
                # self-coding after a retry was already scheduled.  Windows
                # command prompt executions in particular hit this race when the
                # provenance check downgrades a bot moments after the initial
                # internalisation failure.  Without the guard we would keep
                # invoking ``_internalize_missing_coding_bot`` even though the
                # bot is now intentionally operating in manual mode which
                # repeatedly surfaces the original import error and stalls the
                # sandbox start-up sequence.  Clearing the retry bookkeeping here
                # mirrors the behaviour of ``register_bot(..., is_coding_bot=False)``
                # and keeps the retry loop quiescent.
                self._internalization_retry_attempts.pop(name, None)
                node.pop("pending_internalization", None)
                node.pop("internalization_errors", None)
                node.pop("internalization_blocked", None)
                self._clear_transient_error_state(name)
                logger.debug(
                    "skipping internalization retry for %s because self-coding is disabled",
                    name,
                )
                return

            attempts = self._internalization_retry_attempts.get(name, 0) + 1
            self._internalization_retry_attempts[name] = attempts

            mgr = node.get("selfcoding_manager") or node.get("manager")
            db = node.get("data_bot")

            try:
                self._internalize_missing_coding_bot(
                    name,
                    manager=mgr,
                    data_bot=db,
                )
            except SelfCodingUnavailableError as exc:
                node["pending_internalization"] = False
                missing = list(exc.missing_dependencies)
                disabled_payload: dict[str, Any] = {
                    "reason": str(exc),
                    "missing_dependencies": missing,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if exc.missing_resources:
                    disabled_payload["missing_resources"] = list(
                        exc.missing_resources
                    )
                node["self_coding_disabled"] = disabled_payload
                self._internalization_retry_attempts.pop(name, None)
                self._clear_transient_error_state(name)
                logger.warning("self-coding disabled for %s: %s", name, exc)
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:self_coding_disabled",
                            {
                                "bot": name,
                                "missing": missing,
                                "resources": list(exc.missing_resources)
                                if exc.missing_resources
                                else None,
                                "reason": str(exc),
                            },
                        )
                    except Exception:  # pragma: no cover - best effort
                        logger.debug(
                            "failed to publish bot:self_coding_disabled event",
                            exc_info=True,
                        )
                return
            except Exception as exc:
                node.setdefault("internalization_errors", []).append(str(exc))
                missing_modules = _collect_missing_modules(exc)
                missing_resources = _collect_missing_resources(exc)
                _purge_partial_modules(exc)
                if missing_modules:
                    logger.debug(
                        "blocking internalization for %s due to missing dependencies: %s",
                        name,
                        ", ".join(sorted(missing_modules)),
                    )
                    self._record_internalization_blocked(name, exc)
                    return
                if missing_resources:
                    logger.debug(
                        "blocking internalization for %s due to missing resources: %s",
                        name,
                        ", ".join(sorted(missing_resources)),
                    )
                    self._record_internalization_blocked(name, exc)
                    return
                if _is_transient_internalization_error(exc) and (
                    attempts < self._max_internalization_retries
                ):
                    state = self._register_transient_failure(name, exc)
                    if state.should_disable(
                        max_signature_repeats=self._max_transient_error_signature_repeats,
                        max_total_repeats=self._max_transient_error_total_repeats,
                        max_window=self._transient_error_grace_period,
                    ):
                        self._disable_self_coding_due_to_transient_errors(
                            name, exc, state
                        )
                        return
                    logger.debug(
                        "deferring internalization for %s due to %s: %s",
                        name,
                        exc.__class__.__name__,
                        exc,
                    )
                    logger.debug(
                        "retrying internalization for %s after transient error (attempt %s)",
                        name,
                        attempts,
                    )
                    self._schedule_internalization_retry(name)
                    return

                self._record_internalization_blocked(name, exc)
                return

            node.pop("pending_internalization", None)
            self._internalization_retry_attempts.pop(name, None)
            self._clear_transient_error_state(name)
            logger.info(
                "internalization retry for %s succeeded after %s attempt(s)",
                name,
                attempts,
            )

    def _internalize_missing_coding_bot(
        self,
        name: str,
        *,
        manager: "SelfCodingManager | None",
        data_bot: "DataBot | None",
    ) -> None:
        node = self.graph.nodes[name]

        components = _load_self_coding_components()
        ctx = components.context_builder_factory()
        engine = components.engine_cls(
            components.code_db_cls(),
            components.memory_manager_cls(),
            context_builder=ctx,
        )
        pipeline = components.pipeline_cls(context_builder=ctx, bot_registry=self)
        db = data_bot or components.data_bot_cls(start_server=False)
        th = _load_self_coding_thresholds(name)
        try:
            components.internalize_coding_bot(
                name,
                engine,
                pipeline,
                data_bot=db,
                bot_registry=self,
                roi_threshold=getattr(th, "roi_drop", None),
                error_threshold=getattr(th, "error_increase", None),
                test_failure_threshold=getattr(th, "test_failure_increase", None),
            )
        except Exception as exc:
            missing_modules = _collect_missing_modules(exc)
            missing_resources = _collect_missing_resources(exc)
            if not missing_modules and not missing_resources:
                # When the import heuristics cannot determine the missing
                # component we fall back to the dependency probe.  This guards
                # against runtime errors raised from deep inside the self-coding
                # stack (for example ``quick_fix_engine`` complaining about
                # ``context_builder_util``) which otherwise surface as
                # transient failures and stall the sandbox on Windows.
                try:
                    ready, missing = ensure_self_coding_ready()
                except Exception:  # pragma: no cover - defensive best effort
                    ready, missing = (False, tuple())
                if not ready and missing:
                    missing_modules.update(missing)
            if not missing_modules and isinstance(exc, (ImportError, ModuleNotFoundError)):
                name_attr = getattr(exc, "name", None)
                if isinstance(name_attr, str) and name_attr.strip():
                    missing_modules.add(name_attr.strip())
            actionable_modules = {
                module
                for module in missing_modules
                if not _is_traceback_only_module(module)
            }
            missing_modules = actionable_modules
            if actionable_modules or missing_resources:
                if missing_modules and missing_resources:
                    reason = (
                        "self-coding bootstrap failed: missing dependencies and runtime resources"
                    )
                elif missing_modules:
                    reason = (
                        "self-coding bootstrap failed: circular import detected"
                        if _CIRCULAR_HINT_RE.search(str(exc))
                        else "self-coding bootstrap failed: missing runtime dependencies"
                    )
                else:
                    reason = "self-coding bootstrap failed: missing runtime resources"
                raise SelfCodingUnavailableError(
                    reason,
                    missing=actionable_modules or missing_modules,
                    missing_resources=missing_resources,
                ) from exc

            if isinstance(exc, ImportError):
                hinted: set[str] = set()
                name_attr = getattr(exc, "name", None)
                if name_attr and "DLL load failed" in str(exc):
                    hinted.add(str(name_attr))
                if "QuickFixEngine failed to initialise" in str(exc):
                    hinted.add("quick_fix_engine")
                if hinted:
                    raise SelfCodingUnavailableError(
                        "self-coding bootstrap failed: runtime component unavailable",
                        missing=hinted,
                    ) from exc

            if (
                not missing_modules
                and not missing_resources
                and isinstance(exc, ImportError)
                and not _is_transient_internalization_error(exc)
            ):
                signature_type, signature_msg = _exception_signature(exc)
                hints = _derive_import_error_hints(exc)
                reason = (
                    "self-coding bootstrap failed: unresolved import error "
                    f"({signature_type}: {signature_msg})"
                )
                raise SelfCodingUnavailableError(reason, missing=hints) from exc

            raise
        node.pop("pending_internalization", None)
        self._internalization_retry_attempts.pop(name, None)

        if self.event_bus:
            try:
                self.event_bus.publish("bot:internalized", {"bot": name})
            except Exception as exc:  # pragma: no cover - best effort
                logger.error(
                    "Failed to publish bot:internalized event: %s",
                    exc,
                )

    def register_bot(
        self,
        name: str,
        *,
        roi_threshold: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        manager: "SelfCodingManager" | None = None,
        data_bot: "DataBot" | None = None,
        module_path: str | os.PathLike[str] | None = None,
        is_coding_bot: bool = False,
    ) -> None:
        """Ensure *name* exists in the graph and persist metadata."""
        with self._lock:
            self.graph.add_node(name)
            node = self.graph.nodes[name]
            if module_path is None:
                fallback = _REGISTERED_BOTS.get(name)
                if isinstance(fallback, dict):
                    module_path = fallback.get("module_path") or fallback.get("module")
            resolved_path: str | None = None
            if module_path is not None:
                try:
                    resolved_path = os.fspath(module_path)
                except TypeError:
                    resolved_path = str(module_path)
                node["module"] = resolved_path
                self.modules[name] = resolved_path
            if not is_coding_bot:
                self._internalization_retry_attempts.pop(name, None)
                self._cancel_internalization_retry(name)
                self._clear_transient_error_state(name)
                node.pop("pending_internalization", None)
                node.pop("internalization_errors", None)
                node.pop("internalization_blocked", None)
                # Persist the manual override so future restarts do not attempt to
                # revive stale self-coding infrastructure.  Windows command prompt
                # executions frequently restart the sandbox without cleaning up
                # the registry which previously left ``selfcoding_manager`` and
                # ``data_bot`` references dangling.  That state triggered
                # background internalisation retries even though the bot was
                # intentionally operating in manual mode, effectively stalling
                # start-up while import errors looped indefinitely.  Clearing the
                # cached helpers ensures manual registrations remain passive.
                node.pop("selfcoding_manager", None)
                node.pop("manager", None)
                node.pop("data_bot", None)
                node["is_coding_bot"] = False
                previous_disabled = node.get("self_coding_disabled")
                timestamp = datetime.now(timezone.utc).isoformat()
                manual_reason = (
                    "self-coding disabled: manual registration without autonomous patching"
                )
                disabled_payload: dict[str, Any] = {
                    "reason": manual_reason,
                    "timestamp": timestamp,
                    "source": "manual_registration",
                    "manual_override": True,
                }
                if resolved_path:
                    disabled_payload["module"] = resolved_path
                if isinstance(previous_disabled, dict) and previous_disabled:
                    disabled_payload["previous_reason"] = previous_disabled.get("reason")
                    if previous_disabled.get("missing_dependencies"):
                        disabled_payload["missing_dependencies"] = list(
                            previous_disabled.get("missing_dependencies", ())
                        )
                node["self_coding_disabled"] = disabled_payload
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:self_coding_disabled",
                            {
                                "bot": name,
                                "reason": manual_reason,
                                "source": "manual_registration",
                            },
                        )
                    except Exception:  # pragma: no cover - best effort event
                        logger.debug(
                            "failed to publish bot:self_coding_disabled event for manual registration",
                            exc_info=True,
                        )
                return
            existing_mgr = node.get("selfcoding_manager") or node.get("manager")
            existing_data = node.get("data_bot")
            is_coding_bot = bool(is_coding_bot)
            if is_coding_bot:
                missing: list[str] = []
                mgr = manager or existing_mgr
                db = data_bot or existing_data
                if mgr is None:
                    missing.append("manager")
                if db is None:
                    missing.append("data_bot")
                if missing:
                    ready, missing_deps = ensure_self_coding_ready()
                    if not ready:
                        node["pending_internalization"] = False
                        node["self_coding_disabled"] = {
                            "reason": "self-coding dependencies unavailable",
                            "missing_dependencies": list(missing_deps),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        self._internalization_retry_attempts.pop(name, None)
                        self._cancel_internalization_retry(name)
                        self._clear_transient_error_state(name)
                        logger.warning(
                            "self-coding disabled for %s: missing dependencies: %s",
                            name,
                            ", ".join(missing_deps),
                        )
                        if self.event_bus:
                            try:
                                self.event_bus.publish(
                                    "bot:self_coding_disabled",
                                    {
                                        "bot": name,
                                        "missing": list(missing_deps),
                                        "reason": "self-coding dependencies unavailable",
                                    },
                                )
                            except Exception:  # pragma: no cover - best effort
                                logger.debug(
                                    "failed to publish bot:self_coding_disabled event",
                                    exc_info=True,
                                )
                        return

                    node["pending_internalization"] = True
                    self._internalization_retry_attempts.pop(name, None)
                    logger.debug(
                        "delaying self-coding bootstrap for %s; helpers missing: %s",
                        name,
                        ", ".join(sorted(missing)),
                    )
                    self._schedule_internalization_retry(
                        name, delay=self._initial_internalization_delay
                    )
                    return
            self._clear_transient_error_state(name)
            if roi_threshold is not None:
                node["roi_threshold"] = float(roi_threshold)
            if error_threshold is not None:
                node["error_threshold"] = float(error_threshold)
            if test_failure_threshold is not None:
                node["test_failure_threshold"] = float(test_failure_threshold)
            node.setdefault("patch_history", [])
            if manager is not None:
                node["selfcoding_manager"] = manager
            if data_bot is not None:
                node["data_bot"] = data_bot
                try:
                    data_bot.check_degradation(
                        name, roi=0.0, errors=0.0, test_failures=0.0
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "failed to initialise baseline for %s: %s", name, exc
                    )
                if manager is not None:
                    orchestrator = getattr(manager, "evolution_orchestrator", None)
                    if orchestrator is not None and hasattr(
                        orchestrator, "register_patch_cycle"
                    ):
                        bus = getattr(data_bot, "event_bus", None)
                        if bus:
                            handler = (
                                lambda _t, e: orchestrator.register_patch_cycle(e)
                            )

                            def _subscribe() -> None:
                                try:
                                    bus.subscribe("degradation:detected", handler)
                                except Exception as exc:  # pragma: no cover - best effort
                                    logger.error(
                                        "failed to subscribe degradation callback for %s: %s",
                                        name,
                                        exc,
                                    )
                                    try:
                                        bus.subscribe("bus:restarted", lambda *_: _subscribe())
                                    except Exception as sub_exc:  # pragma: no cover - best effort
                                        logger.error(
                                            "failed to schedule resubscription for %s: %s",
                                            name,
                                            sub_exc,
                                        )

                            _subscribe()
                        else:
                            try:
                                data_bot.subscribe_degradation(
                                    orchestrator.register_patch_cycle
                                )
                            except Exception as exc:  # pragma: no cover - best effort
                                logger.error(
                                    "failed to subscribe degradation callback for %s: %s",
                                    name,
                                    exc,
                            )
                    else:
                        def _on_degraded(event: dict, _bot=name, _mgr=manager):
                            if str(event.get("bot")) != _bot:
                                return
                            try:
                                desc = f"auto_patch_due_to_degradation:{_bot}"
                                token = getattr(
                                    getattr(_mgr, "evolution_orchestrator", None),
                                    "provenance_token",
                                    None,
                                )
                                result_vals = _mgr.register_patch_cycle(
                                    desc, event, provenance_token=token
                                )
                                if isinstance(result_vals, tuple):
                                    patch_id, commit = result_vals
                                else:
                                    patch_id, commit = (None, None)
                                module = self.graph.nodes[_bot].get("module")
                                result = None
                                post_validation: Dict[str, Any] | None = None
                                post_validation_error: str | None = None
                                module_path = Path(module) if module else None
                                if module_path and hasattr(_mgr, "generate_and_patch"):
                                    result, new_commit = _mgr.generate_and_patch(
                                        module_path,
                                        desc,
                                        context_meta=event,
                                        context_builder=_mgr.refresh_quick_fix_context(),
                                        provenance_token=token,
                                    )
                                    commit = commit or new_commit
                                    if (
                                        commit
                                        and hasattr(_mgr, "run_post_patch_cycle")
                                    ):
                                        try:
                                            post_validation = _mgr.run_post_patch_cycle(
                                                module_path,
                                                desc,
                                                provenance_token=token,
                                                context_meta=event,
                                            )
                                        except Exception as exc:  # pragma: no cover - best effort
                                            post_validation_error = str(exc)
                                            logger.error(
                                                "post patch self-test failed for %s: %s",
                                                _bot,
                                                exc,
                                            )
                                try:
                                    ph = self.graph.nodes[_bot].setdefault(
                                        "patch_history", []
                                    )
                                    ph.append(
                                        {
                                            "patch_id": patch_id,
                                            "commit": commit,
                                            "ts": time.time(),
                                        }
                                    )
                                except Exception:
                                    logger.exception(
                                        "failed to record patch history for %s",
                                        _bot,
                                    )
                                if self.event_bus:
                                    try:
                                        converted_result: Any | None = None
                                        if result is not None:
                                            converted_result = (
                                                asdict(result)
                                                if is_dataclass(result)
                                                else getattr(result, "__dict__", result)
                                            )
                                        payload: Dict[str, Any] = {
                                            "bot": _bot,
                                            "patch_id": patch_id,
                                            "commit": commit,
                                        }
                                        if converted_result is not None:
                                            payload["result"] = converted_result
                                        if post_validation is not None:
                                            payload["post_validation"] = post_validation
                                        if post_validation_error is not None:
                                            payload["post_validation_error"] = (
                                                post_validation_error
                                            )
                                            self.event_bus.publish(
                                                "bot:patch_failed", payload
                                            )
                                        elif converted_result is not None:
                                            self.event_bus.publish(
                                                "bot:patch_applied", payload
                                            )
                                    except Exception as exc:
                                        logger.error(
                                            "Failed to publish bot patch event: %s",
                                            exc,
                                        )
                            except Exception as exc:  # pragma: no cover - best effort
                                logger.error(
                                    "degradation callback failed for %s: %s",
                                    _bot,
                                    exc,
                                )

                        bus = getattr(data_bot, "event_bus", None)

                        def _subscribe() -> None:
                            if bus:
                                bus.subscribe(
                                    "degradation:detected",
                                    lambda _t, e: _on_degraded(e),
                                )
                            else:
                                data_bot.subscribe_degradation(_on_degraded)

                        try:
                            with_retry(_subscribe, attempts=3, delay=1.0, logger=logger)
                        except Exception as exc:
                            logger.error(
                                "failed to subscribe degradation callback for %s: %s",
                                name,
                                exc,
                            )
                            if self.event_bus:
                                try:
                                    self.event_bus.publish(
                                        "bot:subscription_failed",
                                        {"bot": name, "error": str(exc)},
                                    )
                                except Exception as exc2:  # pragma: no cover - best effort
                                    logger.error(
                                        "Failed to publish bot:subscription_failed event: %s",
                                        exc2,
                                    )
                            raise
            if (
                roi_threshold is not None
                or error_threshold is not None
                or test_failure_threshold is not None
            ):
                try:
                    threshold_service.update(
                        name,
                        roi_drop=roi_threshold,
                        error_threshold=error_threshold,
                        test_failure_threshold=test_failure_threshold,
                    )
                    persist_sc_thresholds(
                        name,
                        roi_drop=roi_threshold,
                        error_increase=error_threshold,
                        test_failure_increase=test_failure_threshold,
                        event_bus=self.event_bus,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "failed to persist thresholds for %s: %s", name, exc
                    )
            if self.event_bus:
                try:
                    self.event_bus.publish("bot:new", {"name": name})
                except Exception as exc:
                    logger.error("Failed to publish bot:new event: %s", exc)
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as exc:
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, exc
                    )

    # ------------------------------------------------------------------
    def get_workflow_tests(
        self,
        bot_name: str,
        *,
        settings: SandboxSettings | None = None,
    ) -> list[str]:
        """Return configured workflow tests for ``bot_name``."""

        return _collect_workflow_tests(bot_name, registry=self, settings=settings)

    def schedule_unmanaged_scan(self, interval: float = 3600.0) -> None:
        """Periodically scan for unmanaged coding bots and register them."""

        root = Path(__file__).resolve().parent
        script = root / "tools" / "find_unmanaged_bots.py"
        if not script.exists():
            return

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    result = subprocess.run(
                        [sys.executable, str(script), str(root)],
                        capture_output=True,
                        text=True,
                    )
                    for line in result.stdout.splitlines():
                        if "unmanaged bot classes" in line:
                            cls_part = line.split("unmanaged bot classes:", 1)[1]
                            for bot in [c.strip() for c in cls_part.split(",") if c.strip()]:
                                try:
                                    self.register_bot(bot, is_coding_bot=True)
                                except Exception:  # pragma: no cover - best effort
                                    logger.exception(
                                        "auto-registration failed for %s", bot
                                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("scheduled unmanaged bot scan failed")

        threading.Thread(target=_loop, daemon=True).start()

    def _verify_signed_provenance(self, patch_id: int, commit: str) -> bool:
        """Return ``True`` if a signed provenance file confirms the update."""

        prov_file = os.environ.get("PATCH_PROVENANCE_FILE")
        pubkey = os.environ.get("PATCH_PROVENANCE_PUBKEY") or os.environ.get(
            "PATCH_PROVENANCE_PUBLIC_KEY"
        )
        if not prov_file or not pubkey:
            raise RuntimeError("signed provenance required")
        try:
            with open(prov_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            data = payload.get("data") or {}
            signature = payload.get("signature")
            if not signature:
                raise RuntimeError("missing signature")
            if str(data.get("patch_id")) != str(patch_id) or str(
                data.get("commit")
            ) != str(commit):
                raise RuntimeError("provenance mismatch")
            from .override_validator import verify_signature

            if not verify_signature(data, signature, pubkey):
                raise RuntimeError("invalid signature")
            if logger.disabled:
                logger.disabled = False
            logger.info(
                "verified signed provenance for patch_id=%s commit=%s",
                patch_id,
                commit,
            )
            return True
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - best effort
            raise RuntimeError(f"provenance verification failed: {exc}") from exc

    def update_bot(
        self,
        name: str,
        module_path: str,
        *,
        patch_id: int | None = None,
        commit: str | None = None,
    ) -> None:
        """Update stored module path for ``name`` and emit ``bot:updated``.

        ``patch_id`` and ``commit`` are expected from the ``SelfCodingManager``
        so changes can be traced back to their origin.  If either piece of
        metadata is missing this method attempts to retrieve it from
        :mod:`patch_provenance` and retries once.  If the metadata still cannot
        be determined a :class:`RuntimeError` is raised.
        """

        is_unsigned = isinstance(commit, str) and commit.startswith(
            _UNSIGNED_COMMIT_PREFIX
        )

        if patch_id is None or commit is None:
            logger.warning(
                "update_bot called without provenance for %s (patch_id=%s commit=%s)",
                name,
                patch_id,
                commit,
            )
            if patch_id is not None:
                try:
                    service_cls = _get_patch_provenance_service_cls()
                    if service_cls is None:
                        raise ImportError("patch_provenance unavailable")
                    service = service_cls()
                    rec = service.db.get(patch_id)
                    if rec and getattr(rec, "summary", None):
                        try:
                            commit = json.loads(rec.summary).get("commit")
                        except Exception:  # pragma: no cover - best effort
                            commit = None
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to fetch patch provenance for %s: %s", patch_id, exc
                    )
            if patch_id is None or commit is None:
                raise RuntimeError("patch provenance required")

        with self._lock:
            self.register_bot(name, is_coding_bot=False)
            node = self.graph.nodes[name]
            unsigned_meta: dict[str, Any] | None = None
            path_obj = Path(module_path)

            if self._commit_already_applied(node, commit):
                if (
                    patch_id is not None
                    and not self._get_patch_status_entry(node, patch_id)
                ):
                    try:
                        self._set_patch_status(node, patch_id, commit, "applied")
                    except Exception:  # pragma: no cover - defensive best effort
                        logger.debug(
                            "failed to alias applied commit %s to patch %s for %s",
                            commit,
                            patch_id,
                            name,
                            exc_info=True,
                        )
                logger.info(
                    "Skipping update for %s because commit %s is already active",
                    name,
                    commit,
                )
                return False

            if patch_id is not None:
                status_entry = self._get_patch_status_entry(node, patch_id)
                if status_entry and status_entry.get("commit") == commit:
                    status = str(status_entry.get("status", ""))
                    if status == "applied":
                        logger.info(
                            "Skipping update for %s because patch %s is already applied",
                            name,
                            patch_id,
                        )
                        return False
                    if status in {"failed", "blocked"}:
                        reason = status_entry.get("reason")
                        logger.warning(
                            "Skipping update for %s because patch %s previously %s (%s)",
                            name,
                            patch_id,
                            status,
                            reason or "no reason recorded",
                        )
                        return False
                if self._is_blacklisted_patch(name, patch_id):
                    logger.warning(
                        "Blocking known broken patch %s for %s", patch_id, name
                    )
                    self._set_patch_status(
                        node,
                        patch_id,
                        commit,
                        "blocked",
                        reason="blacklisted_patch",
                    )
                    node["update_blocked"] = True
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "bot:update_blocked",
                                {
                                    "name": name,
                                    "module": module_path,
                                    "patch_id": patch_id,
                                    "commit": commit,
                                    "reason": "blacklisted_patch",
                                    "patch_success": False,
                                },
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed to publish bot:update_blocked for %s: %s",
                                name,
                                exc,
                            )
                    if self.persist_path:
                        try:
                            self.save(self.persist_path)
                        except Exception as exc:
                            logger.error(
                                "Failed to save bot registry to %s: %s",
                                self.persist_path,
                                exc,
                            )
                    return False

            if not is_unsigned:
                try:
                    self._verify_signed_provenance(patch_id, commit)
                except RuntimeError as exc:
                    logger.error("Signed provenance verification failed: %s", exc)
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "bot:update_blocked",
                                {
                                    "name": name,
                                    "module": module_path,
                                    "patch_id": patch_id,
                                    "commit": commit,
                                    "reason": "unverified_provenance",
                                    "error": str(exc),
                                    "patch_success": False,
                                },
                            )
                        except Exception as exc2:
                            logger.error(
                                "Failed to publish bot:update_blocked event: %s", exc2
                            )
                    node["update_blocked"] = True
                    if self.persist_path:
                        try:
                            self.save(self.persist_path)
                        except Exception as exc2:  # pragma: no cover - best effort
                            logger.error(
                                "Failed to save bot registry to %s: %s",
                                self.persist_path,
                                exc2,
                            )
                    raise RuntimeError(
                        "update blocked: provenance verification failed"
                    ) from exc
            else:
                timestamp = time.time()
                unsigned_meta = {
                    "commit": commit,
                    "patch_id": patch_id,
                    "timestamp": timestamp,
                }
                node.pop("update_blocked", None)
                cache_key = (name, commit)
                with _UNSIGNED_PROVENANCE_WARNING_LOCK:
                    cache_miss = (
                        cache_key not in _UNSIGNED_PROVENANCE_WARNING_CACHE
                    )
                    if cache_miss:
                        _UNSIGNED_PROVENANCE_WARNING_CACHE.add(cache_key)
                    last_warn_ts = _UNSIGNED_PROVENANCE_WARNING_LAST_TS.get(name)
                    should_warn = cache_miss and (
                        last_warn_ts is None
                        or timestamp - last_warn_ts
                        >= _UNSIGNED_PROVENANCE_WARNING_INTERVAL_SECONDS
                    )
                    if should_warn:
                        _UNSIGNED_PROVENANCE_WARNING_LAST_TS[name] = timestamp
                if should_warn:
                    logger.warning(
                        "Applying unsigned provenance update for %s (patch_id=%s)",
                        name,
                        patch_id,
                    )
                elif cache_miss:
                    logger.debug(
                        "Rate limiting unsigned provenance warning for %s (patch_id=%s)",
                        name,
                        patch_id,
                    )
                else:
                    logger.debug(
                        "Suppressing duplicate unsigned provenance warning for %s (patch_id=%s)",
                        name,
                        patch_id,
                    )

            prev_state = dict(node)
            prev_module_entry = self.modules.get(name)
            node["module"] = module_path
            node["version"] = int(node.get("version", 0)) + 1
            node["patch_id"] = patch_id
            node["commit"] = commit
            if unsigned_meta is not None:
                node["unsigned_provenance"] = unsigned_meta
            try:
                ph = node.setdefault("patch_history", [])
                ph.append({"patch_id": patch_id, "commit": commit, "ts": time.time()})
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to record patch history for %s", name)
            self.modules[name] = module_path

            if self.event_bus:
                try:
                    payload = {
                        "name": name,
                        "module": module_path,
                        "version": node["version"],
                        "patch_id": patch_id,
                        "commit": commit,
                    }
                    if unsigned_meta is not None:
                        payload["unsigned"] = True
                    if patch_id is not None:
                        payload["patch_success"] = True
                    self.event_bus.publish("bot:updated", payload)
                except Exception as exc:
                    logger.error("Failed to publish bot:updated event: %s", exc)
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as exc:
                    logger.error(
                        "Failed to save bot registry to %s: %s",
                        self.persist_path,
                        exc,
                    )
            update_ok = False
            try:
                self.hot_swap_bot(name)
                self.health_check_bot(name, prev_state)
                update_ok = True
                if patch_id is not None:
                    self._set_patch_status(node, patch_id, commit, "applied")
                if self.event_bus and unsigned_meta is not None:
                    try:
                        self.event_bus.publish(
                            "bot:unsigned_update",
                            {
                                "name": name,
                                "module": module_path,
                                "patch_id": patch_id,
                                "commit": commit,
                                "timestamp": unsigned_meta["timestamp"],
                            },
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error(
                            "Failed to publish bot:unsigned_update event: %s", exc
                        )
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "thresholds:refresh", {"bot": name}
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error(
                            "Failed to publish thresholds:refresh event for %s: %s",
                            name,
                            exc,
                        )
            except Exception as exc:
                if patch_id is not None:
                    self._set_patch_status(
                        prev_state,
                        patch_id,
                        commit,
                        "failed",
                        reason="exception",
                        extra={"error": str(exc)},
                    )
                if prev_module_entry is None:
                    self.modules.pop(name, None)
                else:
                    self.modules[name] = prev_module_entry
                node.clear()
                node.update(prev_state)
                if self.persist_path:
                    try:
                        self.save(self.persist_path)
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error(
                            "Failed to save bot registry to %s: %s",
                            self.persist_path,
                            exc,
                        )
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:update_rolled_back",
                            {
                                "name": name,
                                "module": module_path,
                                "patch_id": patch_id,
                                "commit": commit,
                                "error": str(exc),
                                "patch_success": False,
                            },
                        )
                    except Exception as pub_exc:
                        logger.error(
                            "Failed to publish bot:update_rolled_back event: %s",
                            pub_exc,
                        )
                if RollbackManager is not None:
                    try:
                        RollbackManager().rollback(
                            str(patch_id), requesting_bot=name
                        )
                    except Exception as rb_exc:  # pragma: no cover - best effort
                        logger.error(
                            "RollbackManager rollback failed for %s: %s",
                            name,
                            rb_exc,
                        )
                raise
        return update_ok

    def hot_swap(self, name: str, module_path: str) -> None:
        """Update ``module_path`` for ``name`` and reload the bot.

        This helper ensures the registry records the new module before
        delegating to :meth:`hot_swap_bot` which performs the actual import
        and validation.  The bot entry must already contain provenance
        metadata (commit hash and patch id) which is typically provided by
        :meth:`update_bot`.
        """

        with self._lock:
            self.register_bot(name, is_coding_bot=False)
            node = self.graph.nodes[name]
            node["module"] = module_path
            self.modules[name] = module_path
        self.hot_swap_bot(name)

    def hot_swap_bot(self, name: str) -> None:
        """Import or reload the module backing ``name`` and refresh references."""

        node = self.graph.nodes.get(name)
        if node is None:
            raise KeyError(f"bot {name!r} has no module path")
        module_path = node.get("module")
        if not module_path:
            module_path = self.modules.get(name)
            if module_path:
                node["module"] = module_path
        if not module_path:
            fallback = _REGISTERED_BOTS.get(name)
            if isinstance(fallback, dict):
                module_path = fallback.get("module_path") or fallback.get("module")
                if module_path:
                    node["module"] = module_path
                    self.modules[name] = module_path
        if not module_path:
            raise KeyError(f"bot {name!r} has no module path")
        commit = node.get("commit")
        patch_id = node.get("patch_id")
        prev_module = node.get("last_good_module")
        prev_version = node.get("last_good_version")
        prev_commit = node.get("last_good_commit")
        prev_patch = node.get("last_good_patch_id")
        if not commit or patch_id is None:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "missing_provenance"},
                    )
                    self.event_bus.publish(
                        "bot:update_blocked",
                        {
                            "name": name,
                            "module": module_path,
                            "reason": "missing_provenance",
                            "patch_success": False,
                        },
                    )
                except Exception as exc:
                    logger.error("Failed to publish bot:update_blocked event: %s", exc)
            node["update_blocked"] = True
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise RuntimeError("update blocked: missing provenance metadata")

        stored_commit: str | None = None
        try:
            service_cls = _get_patch_provenance_service_cls()
            if service_cls is None:
                raise ImportError("patch_provenance unavailable")
            service = service_cls()
            rec = service.db.get(patch_id)
            if rec and getattr(rec, "summary", None):
                try:
                    stored_commit = json.loads(rec.summary).get("commit")
                except Exception:  # pragma: no cover - best effort
                    stored_commit = None
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to fetch patch provenance for %s: %s", patch_id, exc)
        # When the provenance service cannot supply a stored commit we
        # conservatively allow the hot swap to proceed.  The provenance
        # metadata provided by :meth:`update_bot` is still present in the
        # registry entry so we are not weakening the guard rails; however, we
        # must not treat ``None`` (or other falsey values) as a mismatched
        # commit.  Doing so caused the runtime error observed when the
        # provenance database had not yet recorded a commit for ``patch_id``.
        if stored_commit is not None and stored_commit != commit:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "provenance_mismatch"},
                    )
                    self.event_bus.publish(
                        "bot:update_blocked",
                        {
                            "name": name,
                            "module": module_path,
                            "reason": "provenance_mismatch",
                            "expected": stored_commit,
                            "actual": commit,
                            "patch_success": False,
                        },
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to publish bot:update_blocked event: %s", exc
                    )
            manager = node.get("selfcoding_manager") or node.get("manager")
            if manager and hasattr(manager, "register_patch_cycle"):
                try:
                    token = getattr(
                        getattr(manager, "evolution_orchestrator", None),
                        "provenance_token",
                        None,
                    )
                    manager.register_patch_cycle(
                        f"manual change detected for {name}",
                        {
                            "reason": "provenance_mismatch",
                            "module": module_path,
                        },
                        provenance_token=token,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to notify SelfCodingManager for %s: %s", name, exc
                    )
            node["update_blocked"] = True
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise RuntimeError("update blocked: provenance mismatch")

        try:
            status = _git_status_for_path(path_obj)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to check manual changes for %s: %s", module_path, exc)
        else:
            if status and status.strip() and self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "uncommitted_changes"},
                    )
                except Exception as exc:
                    logger.error("Failed to publish bot:manual_change event: %s", exc)
        try:
            with _hot_swap_guard(name):
                if _is_probable_filesystem_path(module_path):
                    module_name, spec_kwargs = _module_spec_from_path(module_path)
                    spec = importlib.util.spec_from_file_location(
                        module_name, module_path, **spec_kwargs
                    )
                    if not spec or not spec.loader:
                        raise ImportError(f"cannot load module from {module_path}")
                    importlib.invalidate_caches()
                    if module_name in sys.modules:
                        module = sys.modules[module_name]
                        module.__file__ = module_path
                        module.__spec__ = spec
                        spec.loader.exec_module(module)
                    else:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                    for alias in _normalise_module_aliases(module_name):
                        sys.modules.setdefault(alias, module)
                else:
                    if module_path in sys.modules:
                        importlib.reload(sys.modules[module_path])
                    else:
                        importlib.import_module(module_path)
                node["last_good_module"] = module_path
                node["last_good_version"] = node.get("version")
                node["last_good_commit"] = commit
                node["last_good_patch_id"] = patch_id
                if self.persist_path:
                    try:
                        self.save(self.persist_path)
                    except Exception as save_exc:  # pragma: no cover - best effort
                        logger.error(
                            "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                        )
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(
                "Failed to hot swap bot %s from %s: %s", name, module_path, exc
            )
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:hot_swap_failed",
                        {"name": name, "module": module_path, "error": str(exc)},
                    )
                except Exception as pub_exc:
                    logger.error(
                        "Failed to publish bot:hot_swap_failed event: %s", pub_exc
                    )
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise

    def health_check_bot(self, name: str, prev_state: Optional[Dict[str, object]] = None) -> None:
        """Import the bot module and record a heartbeat to verify health."""

        node = self.graph.nodes.get(name)
        if not node:
            raise KeyError(f"bot {name!r} has no module path")
        module_path = node.get("module")
        if not module_path:
            fallback = _REGISTERED_BOTS.get(name)
            if isinstance(fallback, dict):
                module_path = fallback.get("module_path") or fallback.get("module")
                if module_path:
                    node["module"] = module_path
                    self.modules[name] = module_path
        if not module_path:
            raise KeyError(f"bot {name!r} has no module path")
        if is_self_coding_import_active(module_path):
            logger.debug(
                "Skipping health check for %s while module import is in progress",
                name,
            )
            self.record_heartbeat(name)
            return
        try:
            with _hot_swap_guard(name):
                path_obj = Path(module_path)
                if _is_probable_filesystem_path(module_path):
                    module_name, spec_kwargs = _module_spec_from_path(module_path)
                    spec = importlib.util.spec_from_file_location(
                        module_name, module_path, **spec_kwargs
                    )
                    if not spec or not spec.loader:
                        raise ImportError(f"cannot load module from {module_path}")
                    importlib.invalidate_caches()
                    if module_name in sys.modules:
                        module = sys.modules[module_name]
                        module.__file__ = module_path
                        module.__spec__ = spec
                        spec.loader.exec_module(module)
                    else:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                    for alias in _normalise_module_aliases(module_name):
                        sys.modules.setdefault(alias, module)
                else:
                    module_name = module_path
                    importlib.invalidate_caches()
                    importlib.import_module(module_name)
                self.record_heartbeat(name)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Health check failed for bot %s: %s", name, exc)
            if prev_state is not None:
                node.clear()
                node.update(prev_state)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:hot_swap_failed",
                        {"name": name, "module": module_path, "error": str(exc)},
                    )
                except Exception as pub_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to publish bot:hot_swap_failed event: %s", pub_exc
                    )
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s",
                        self.persist_path,
                        save_exc,
                    )
            raise

    def register_interaction(self, from_bot: str, to_bot: str, weight: float = 1.0) -> None:
        """Record that *from_bot* interacted with *to_bot*."""
        self.register_bot(from_bot, is_coding_bot=False)
        self.register_bot(to_bot, is_coding_bot=False)
        if self.graph.has_edge(from_bot, to_bot):
            self.graph[from_bot][to_bot]["weight"] += weight
        else:
            self.graph.add_edge(from_bot, to_bot, weight=weight)
        self.interactions_meta.append(
            {"from": from_bot, "to": to_bot, "weight": weight, "ts": time.time()}
        )
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "bot:interaction", {"from": from_bot, "to": to_bot, "weight": weight}
                )
            except Exception as exc:
                logger.error("Failed to publish bot:interaction event: %s", exc)
        if self.persist_path:
            try:
                self.save(self.persist_path)
            except Exception as exc:
                logger.error(
                    "Failed to save bot registry to %s: %s", self.persist_path, exc
                )

    def connections(self, bot: str, depth: int = 1) -> List[Tuple[str, float]]:
        """Return outgoing connections up to *depth* hops."""
        results: List[Tuple[str, float]] = []
        if bot not in self.graph:
            return results
        for nbr in self.graph.successors(bot):
            w = float(self.graph[bot][nbr].get("weight", 1.0))
            results.append((nbr, w))
            if depth > 1:
                results.extend(self.connections(nbr, depth - 1))
        return results

    # ------------------------------------------------------------------
    def record_heartbeat(self, name: str) -> None:
        """Update last seen timestamp for *name*."""
        self.heartbeats[name] = time.time()
        if self.event_bus:
            try:
                self.event_bus.publish("bot:heartbeat", {"name": name})
            except Exception as exc:
                logger.error("Failed to publish bot:heartbeat event: %s", exc)
                try:
                    self.event_bus.publish(
                        "bot:heartbeat_error", {"name": name, "error": str(exc)}
                    )
                except Exception:
                    logger.exception("Failed publishing heartbeat error")

    def record_validation(self, bot: str, module: str, passed: bool) -> None:
        """Record patch validation outcome for ``bot``."""
        self.interactions_meta.append(
            {"bot": bot, "module": module, "passed": bool(passed), "ts": time.time()}
        )
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "bot:patch_validation",
                    {"bot": bot, "module": module, "passed": bool(passed)},
                )
            except Exception as exc:
                logger.error("Failed to publish bot:patch_validation event: %s", exc)

    def active_bots(self, timeout: float = 60.0) -> Dict[str, float]:
        """Return bots seen within ``timeout`` seconds."""
        now = time.time()
        return {n: ts for n, ts in self.heartbeats.items() if now - ts <= timeout}

    def record_interaction_metadata(
        self,
        from_bot: str,
        to_bot: str,
        *,
        duration: float,
        success: bool,
        resources: str = "",
    ) -> None:
        """Store detailed metadata for an interaction."""
        self.interactions_meta.append(
            {
                "from": from_bot,
                "to": to_bot,
                "duration": duration,
                "success": success,
                "resources": resources,
                "ts": time.time(),
            }
        )

    def aggregate_statistics(self) -> Dict[str, float]:
        """Return simple aggregate metrics about interactions."""
        if not self.interactions_meta:
            return {"count": 0, "success_rate": 0.0, "avg_duration": 0.0}
        count = len(self.interactions_meta)
        successes = sum(1 for rec in self.interactions_meta if rec.get("success"))
        total_dur = sum(float(rec.get("duration", 0.0)) for rec in self.interactions_meta)
        return {
            "count": count,
            "success_rate": successes / count,
            "avg_duration": total_dur / count,
        }

    # ------------------------------------------------------------------
    def save(
        self, dest: Union[Path, str, "MenaceDB", "PathwayDB", DBRouter]
    ) -> None:
        """Persist the current graph to a SQLite-backed database."""
        if isinstance(dest, (str, Path)):
            path = Path(dest)
            router = db_router.GLOBAL_ROUTER or init_db_router(
                "bot_registry", str(path), str(path)
            )
            conn = router.get_connection("bots")
            close_conn = False
        elif isinstance(dest, DBRouter):
            conn = dest.get_connection("bots")
            close_conn = False
        elif MenaceDB is not None and isinstance(dest, MenaceDB):
            conn = dest.engine.raw_connection()
            close_conn = False
        elif PathwayDB is not None and isinstance(dest, PathwayDB):
            conn = dest.conn
            close_conn = False
        else:  # pragma: no cover - invalid type
            raise TypeError("Unsupported destination for save")

        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS bot_nodes(" "name TEXT PRIMARY KEY, "
            "module TEXT, "
            "version INTEGER, "
            "last_good_module TEXT, "
            "last_good_version INTEGER)"
        )
        # Ensure columns exist for databases created before they were introduced.
        try:  # pragma: no cover - only executed on legacy schemas
            cols = [r[1] for r in cur.execute("PRAGMA table_info(bot_nodes)").fetchall()]
            if "module" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN module TEXT")
            if "version" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN version INTEGER")
            if "last_good_module" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN last_good_module TEXT")
            if "last_good_version" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN last_good_version INTEGER")
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_edges(
                from_bot TEXT,
                to_bot TEXT,
                weight REAL,
                PRIMARY KEY(from_bot, to_bot)
            )
            """
        )
        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            module = data.get("module")
            version = data.get("version")
            last_mod = data.get("last_good_module")
            last_ver = data.get("last_good_version")
            cur.execute(
                """
                INSERT OR REPLACE INTO bot_nodes(
                    name, module, version, last_good_module, last_good_version
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (node, module, version, last_mod, last_ver),
            )
        for u, v, data in self.graph.edges.data():
            cur.execute(
                "REPLACE INTO bot_edges(from_bot,to_bot,weight) VALUES(?,?,?)",
                (u, v, float(data.get("weight", 1.0))),
            )
        conn.commit()
        if close_conn:
            conn.close()

    # ------------------------------------------------------------------
    def load(
        self, src: Union[Path, str, "MenaceDB", "PathwayDB", DBRouter]
    ) -> None:
        """Populate ``self.graph`` from ``src`` tables."""
        if isinstance(src, (str, Path)):
            path = Path(src)
            router = db_router.GLOBAL_ROUTER or init_db_router(
                "bot_registry", str(path), str(path)
            )
            conn = router.get_connection("bots")
            close_conn = False
        elif isinstance(src, DBRouter):
            conn = src.get_connection("bots")
            close_conn = False
        elif MenaceDB is not None and isinstance(src, MenaceDB):
            conn = src.engine.raw_connection()
            close_conn = False
        elif PathwayDB is not None and isinstance(src, PathwayDB):  # pragma: no cover - rarely used
            conn = src.conn
            close_conn = False
        else:  # pragma: no cover - invalid type
            raise TypeError("Unsupported source for load")

        self.graph.clear()
        cur = conn.cursor()
        try:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(bot_nodes)").fetchall()]
        except Exception:
            cols = []

        module_col = "module" in cols
        version_col = "version" in cols
        last_mod_col = "last_good_module" in cols
        last_ver_col = "last_good_version" in cols
        select_cols = [
            c for c in ["module", "version", "last_good_module", "last_good_version"] if c in cols
        ]
        col_sql = ", ".join(select_cols)
        try:
            if col_sql:
                node_rows = cur.execute(
                    f"SELECT name, {col_sql} FROM bot_nodes"
                ).fetchall()
            else:
                node_rows = cur.execute("SELECT name FROM bot_nodes").fetchall()
        except Exception:  # pragma: no cover - corrupted table
            node_rows = []

        for row in node_rows:
            name = row[0]
            self.graph.add_node(name)
            idx = 1
            if module_col:
                module = row[idx]
                idx += 1
                if module is not None:
                    self.graph.nodes[name]["module"] = module
            if version_col:
                version = row[idx]
                idx += 1
                if version is not None:
                    self.graph.nodes[name]["version"] = int(version)
            if last_mod_col:
                last_mod = row[idx]
                idx += 1
                if last_mod is not None:
                    self.graph.nodes[name]["last_good_module"] = last_mod
            if last_ver_col:
                last_ver = row[idx]
                idx += 1
                if last_ver is not None:
                    self.graph.nodes[name]["last_good_version"] = int(last_ver)

        try:
            edge_rows = cur.execute(
                "SELECT from_bot, to_bot, weight FROM bot_edges"
            ).fetchall()
        except Exception:
            edge_rows = []
        for u, v, w in edge_rows:
            self.graph.add_edge(u, v, weight=float(w))

        if close_conn:
            conn.close()


def get_bot_workflow_tests(
    bot_name: str,
    *,
    registry: "BotRegistry | None" = None,
    settings: SandboxSettings | None = None,
) -> list[str]:
    """Return workflow tests from registry, settings or threshold config."""

    return _collect_workflow_tests(bot_name, registry=registry, settings=settings)

def get_all_registered_bots():
    return list(_REGISTERED_BOTS.keys())


__all__ = ["BotRegistry", "get_bot_workflow_tests"]


def _load_self_coding_components() -> _SelfCodingComponents:
    """Return the core classes required for self-coding integration."""

    ready, missing = ensure_self_coding_ready()
    if not ready:
        raise SelfCodingUnavailableError(
            "self-coding dependencies unavailable",
            missing=missing,
        )

    def _load(name: str) -> Any:
        try:
            return _load_internal_module(name)
        except (ImportError, ModuleNotFoundError) as exc:
            modules = set(_collect_missing_modules(exc))
            if name not in modules and f"menace_sandbox.{name}" not in modules:
                modules.add(name)
            raise SelfCodingUnavailableError(
                "self-coding bootstrap failed: missing runtime dependencies",
                missing=modules,
            ) from exc

    try:
        manager_mod = _load("self_coding_manager")
        engine_mod = _load("self_coding_engine")
        pipeline_mod = _load("model_automation_pipeline")
        data_bot_mod = _load("data_bot")
        code_db_mod = _load("code_database")
        memory_mod = _load("gpt_memory")
        ctx_util_mod = _load("context_builder_util")
    except SelfCodingUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        modules = _collect_missing_modules(exc)
        raise SelfCodingUnavailableError(
            "self-coding bootstrap failed: unexpected runtime error",
            missing=modules or ("self_coding_runtime",),
        ) from exc

    missing_attrs: list[str] = []
    components = {
        "internalize_coding_bot": getattr(manager_mod, "internalize_coding_bot", None),
        "engine_cls": getattr(engine_mod, "SelfCodingEngine", None),
        "pipeline_cls": getattr(pipeline_mod, "ModelAutomationPipeline", None),
        "data_bot_cls": getattr(data_bot_mod, "DataBot", None),
        "code_db_cls": getattr(code_db_mod, "CodeDB", None),
        "memory_manager_cls": getattr(memory_mod, "GPTMemoryManager", None),
        "context_builder_factory": getattr(ctx_util_mod, "create_context_builder", None),
    }

    for attr, value in components.items():
        if value is None:
            missing_attrs.append(attr)

    if missing_attrs:
        raise SelfCodingUnavailableError(
            "self-coding bootstrap failed: missing runtime dependencies",
            missing=tuple(sorted(missing_attrs)),
        )

    return _SelfCodingComponents(
        components["internalize_coding_bot"],
        components["engine_cls"],
        components["pipeline_cls"],
        components["data_bot_cls"],
        components["code_db_cls"],
        components["memory_manager_cls"],
        components["context_builder_factory"],
    )

