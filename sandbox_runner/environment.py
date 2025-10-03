# flake8: noqa
"""Runtime helpers for sandbox execution.

The sandbox repository location is resolved via :mod:`sandbox_runner.config`,
which consults the ``SANDBOX_REPO_URL`` and ``SANDBOX_REPO_PATH`` environment
variables or a ``SandboxSettings`` instance.  When unset, defaults point to the
current repository checkout.
"""

from __future__ import annotations

import sys

_this_module = sys.modules.setdefault(__name__, sys.modules.get(__name__))
for _alias in (
    "sandbox_runner.environment",
    "menace.sandbox_runner.environment",
    "menace_sandbox.sandbox_runner.environment",
):
    if _alias != __name__:
        sys.modules[_alias] = _this_module
del _alias, _this_module

import ast
import asyncio
import json
import math
import os
import yaml
from vector_service.context_builder import ContextBuilder

if os.getenv("SANDBOX_CENTRAL_LOGGING") == "1":
    from logging_utils import setup_logging

    setup_logging()

from logging_utils import get_logger, log_record
from metrics_exporter import sandbox_crashes_total
from alert_dispatcher import dispatch_alert
import re
from dynamic_path_router import resolve_path, repo_root, path_for_prompt

from .orphan_integration import integrate_and_graph_orphans
from .scoring import record_run as _score_record_run, load_summary as _load_run_summary

try:
    import resource
except (ImportError, OSError):  # pragma: no cover - not available on some platforms
    resource = None  # type: ignore
import shutil
import socket
import subprocess
import tempfile
import textwrap
import logging
import multiprocessing
import time
import inspect
import random
import threading
import queue
import signal
import weakref
import traceback
import statistics
import copy
from datetime import datetime
from pathlib import Path
import importlib
import atexit
from typing import (
    Any,
    Dict,
    List,
    Callable,
    Iterable,
    Sequence,
    Mapping,
    Iterator,
    get_origin,
    get_args,
    TYPE_CHECKING,
    TypeVar,
)
from contextlib import asynccontextmanager, contextmanager, suppress, nullcontext
from lock_utils import SandboxLock as FileLock, Timeout
from dataclasses import dataclass, asdict
from sandbox_settings import SandboxSettings
from collections import deque, defaultdict

T = TypeVar("T")

from .workflow_sandbox_runner import WorkflowSandboxRunner
from metrics_exporter import Gauge, environment_failure_total

environment_failure_severity_total = Gauge(
    "environment_failure_severity_total",
    "Total number of sandbox environment failures by severity",
    labelnames=["severity"],
)

# Name of the sandbox snippet file.  Configurable to avoid hard-coded paths.
SNIPPET_NAME = os.getenv("SANDBOX_SNIPPET_NAME", "snippet" + ".py")

DiagnosticManager = None  # type: ignore[assignment]
ResolutionRecord = None  # type: ignore[assignment]

try:  # optional dependency
    from .meta_logger import _SandboxMetaLogger
except Exception:  # pragma: no cover - best effort
    _SandboxMetaLogger = None  # type: ignore

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:  # pragma: no cover - coverage is optional
    import coverage  # type: ignore
except Exception:  # pragma: no cover - coverage unavailable
    coverage = None  # type: ignore


def generate_edge_cases() -> dict[str, Any]:
    """Expose edge case payloads for test injection."""
    from .edge_case_generator import generate_edge_cases as _gen

    return _gen()

# ---------------------------------------------------------------------------
# Reusable edge case stubs

from .edge_case_generator import (
    malformed_json as _ec_malformed_json,
    timeout_sentinel as _ec_timeout,
    null_or_empty as _ec_null_or_empty,
    invalid_format as _ec_invalid_format,
)

_EDGE_CASE_ENV = {
    "malformed_json": "SANDBOX_EC_MALFORMED_JSON",
    "timeouts": "SANDBOX_EC_TIMEOUTS",
    "nulls": "SANDBOX_EC_NULLS",
    "empty_strings": "SANDBOX_EC_EMPTY_STRINGS",
    "invalid_formats": "SANDBOX_EC_INVALID_FORMATS",
}


def _edge_case_enabled(name: str) -> bool:
    """Return True if the given edge case category is enabled."""
    flag = os.getenv(_EDGE_CASE_ENV[name], "1")
    return flag not in {"0", "false", "False"}


def get_edge_case_profiles() -> list[dict[str, Any]]:
    """Return enabled edge case profiles."""
    none_val, empty_val = _ec_null_or_empty()
    profiles: list[dict[str, Any]] = []
    if _edge_case_enabled("malformed_json"):
        val = _ec_malformed_json()
        profiles.append({"malformed.json": val})
        profiles.append({"http://edge-case.test/malformed": val})
    if _edge_case_enabled("timeouts"):
        val = _ec_timeout()
        profiles.append({"timeout": val})
        profiles.append({"http://edge-case.test/timeout": val})
    if _edge_case_enabled("nulls"):
        profiles.append({"null.txt": none_val})
        profiles.append({"http://edge-case.test/null": none_val})
    if _edge_case_enabled("empty_strings"):
        profiles.append({"empty.txt": empty_val})
        profiles.append({"http://edge-case.test/empty": empty_val})
    if _edge_case_enabled("invalid_formats"):
        val = _ec_invalid_format()
        profiles.append({"invalid.bin": val})
        profiles.append({"http://edge-case.test/invalid": val})
    return profiles


def get_edge_case_stubs() -> dict[str, Any]:
    """Return merged edge case stubs respecting configuration flags."""
    merged: dict[str, Any] = {}
    for prof in get_edge_case_profiles():
        merged.update(prof)
    return merged

_USE_MODULE_SYNERGY = os.getenv("SANDBOX_USE_MODULE_SYNERGY") == "1"
try:  # pragma: no cover - optional dependency
    from module_synergy_grapher import get_synergy_cluster
except ImportError:  # pragma: no cover - optional dependency
    def get_synergy_cluster(*_args: object, **_kwargs: object) -> set[str]:  # type: ignore
        return set()

if TYPE_CHECKING:  # pragma: no cover
    from self_improvement.baseline_tracker import BaselineTracker

_BASELINE_TRACKER: "BaselineTracker" | None = None


def _get_baseline_tracker() -> "BaselineTracker":
    """Return the shared :class:`BaselineTracker` instance lazily."""

    global _BASELINE_TRACKER
    if _BASELINE_TRACKER is None:
        from self_improvement.baseline_tracker import TRACKER as _GLOBAL_TRACKER  # type: ignore

        tracker: "BaselineTracker" = _GLOBAL_TRACKER
        for _metric in ("side_effects", "intent_similarity", "synergy"):
            tracker._history.setdefault(_metric, deque(maxlen=tracker.window))
        _BASELINE_TRACKER = tracker
    return _BASELINE_TRACKER

if TYPE_CHECKING:  # pragma: no cover
    from foresight_tracker import ForesightTracker
    from db_router import DBRouter
    from intent_clusterer import IntentClusterer

# Relevancy radar integration -------------------------------------------------
_ENABLE_RELEVANCY_RADAR = os.getenv("SANDBOX_ENABLE_RELEVANCY_RADAR") == "1"
_RADAR_WARNING_EMITTED = False
try:  # pragma: no cover - optional dependency
    from relevancy_radar import track_usage as _radar_track_module_usage
    RELEVANCY_RADAR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RELEVANCY_RADAR_AVAILABLE = False

    def _radar_track_module_usage(_module: str) -> None:  # type: ignore
        return None


class _RadarWorker:
    """Context manager managing the background radar tracking worker."""

    def __init__(self) -> None:
        self.queue: queue.Queue[str] | None = None
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None

    def __enter__(self) -> "_RadarWorker":
        if self.thread is not None and self.thread.is_alive():
            return self
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=_radar_worker, args=(self.queue, self.stop_event),
            name="radar-track", daemon=True,
        )
        try:
            self.thread.start()
            logger.info("radar worker started")
        except RuntimeError as exc:  # pragma: no cover - targeted
            sandbox_crashes_total.inc()
            logger.exception(
                "radar worker failed to start",
                exc_info=exc,
                extra=log_record(component="radar_worker"),
            )
            self.thread = None
            self.stop_event = None
            self.queue = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.queue is None:
            return
        if self.stop_event is not None:
            self.stop_event.set()
        try:
            self.queue.put(None)
        except queue.Full as exc:  # pragma: no cover - targeted
            sandbox_crashes_total.inc()
            logger.exception(
                "radar worker queue full on shutdown",
                exc_info=exc,
                extra=log_record(component="radar_worker"),
            )
            raise
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=1.0)
            except RuntimeError as exc:  # pragma: no cover - targeted
                sandbox_crashes_total.inc()
                logger.exception(
                    "failed joining radar worker",
                    exc_info=exc,
                    extra=log_record(component="radar_worker"),
                )
                raise
            if self.thread.is_alive():
                logger.warning("radar worker did not terminate before timeout")
            else:
                logger.info("radar worker terminated")

    def track(self, module: str) -> None:
        if self.queue is None:
            raise RuntimeError("radar worker not running")
        try:
            self.queue.put_nowait(module)
        except queue.Full as exc:
            sandbox_crashes_total.inc()
            logger.exception(
                "radar worker queue full",
                exc_info=exc,
                extra=log_record(module=module, component="radar_worker"),
            )
            raise RuntimeError("radar worker queue full") from exc


def _radar_worker(q: "queue.Queue[str]", stop: threading.Event) -> None:
    while not stop.is_set():
        module = q.get()
        if module is None:
            break
        try:
            _radar_track_module_usage(module)
        except Exception as exc:  # pragma: no cover - best effort
            if _ERROR_CONTEXT_BUILDER is not None:
                record_error(exc, context_builder=_ERROR_CONTEXT_BUILDER)
            else:
                logger.exception(
                    "relevancy radar tracking failed", exc_info=exc
                )


_RADAR_MANAGER = _RadarWorker()


def _async_radar_track(module: str) -> None:
    """Record ``module`` usage without blocking."""

    if not _ENABLE_RELEVANCY_RADAR:
        return
    if not RELEVANCY_RADAR_AVAILABLE:
        global _RADAR_WARNING_EMITTED
        if not _RADAR_WARNING_EMITTED:
            logger.warning(
                "relevancy_radar dependency missing; tracking disabled",
            )
            _RADAR_WARNING_EMITTED = True
        return
    try:
        _RADAR_MANAGER.track(module)
    except RuntimeError as exc:  # pragma: no cover - best effort
        if _ERROR_CONTEXT_BUILDER is not None:
            record_error(exc, context_builder=_ERROR_CONTEXT_BUILDER)
        else:
            logger.exception(
                "relevancy radar queue failure", exc_info=exc
            )

import builtins

_original_import = builtins.__import__


def _tracking_import(
    name, globals=None, locals=None, fromlist=(), level=0
):  # pragma: no cover - thin wrapper
    mod = _original_import(name, globals, locals, fromlist, level)
    record_fn = globals().get("record_module_usage")
    module_file = getattr(mod, "__file__", "")
    if record_fn and module_file:
        root = repo_root()
        path = Path(module_file).resolve()
        try:
            path = path.relative_to(root)
        except ValueError:
            logger.debug("module %s outside root %s", module_file, root)
        record_fn(path.as_posix())
    return mod


@contextmanager
def _patched_imports() -> Iterable[None]:
    """Temporarily install import tracker."""

    previous = builtins.__import__
    builtins.__import__ = _tracking_import
    try:
        with _RADAR_MANAGER:
            yield
    finally:
        builtins.__import__ = previous

logger = get_logger(__name__)

# Snapshot initial environment for restoration between runs
_BASE_ENV = os.environ.copy()


@contextmanager
def preserve_sandbox_env() -> Iterator[None]:
    """Restore ``SANDBOX_*`` variables to previous values after the block."""
    snapshot = {k: os.environ.get(k) for k in os.environ if k.startswith("SANDBOX_")}
    try:
        yield
    finally:
        current = {k for k in os.environ if k.startswith("SANDBOX_")}
        for k in current - snapshot.keys():
            os.environ.pop(k, None)
        for k, v in snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _reset_runtime_state() -> None:
    """Restore environment variables and clear temporary state."""
    os.environ.clear()
    os.environ.update(_BASE_ENV)

    # Clear temporary directory contents
    tmp_root = Path(tempfile.gettempdir())
    for child in tmp_root.iterdir():
        if not child.name.startswith(("tmp", "sandbox")):
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("temp cleanup failed for %s", child, exc_info=True)

    # Reset tempfile module cache and import caches
    tempfile.tempdir = None
    importlib.invalidate_caches()
    cleanup_artifacts()


def cleanup_artifacts(extra_paths: Iterable[Path] | None = None) -> None:
    """Remove temporary files, coverage data and Docker artefacts.

    Parameters
    ----------
    extra_paths:
        Additional paths to purge.  Missing paths are ignored.
    """

    leftovers: list[str] = []
    for path in extra_paths or []:
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            leftovers.append(str(path))
            logger.debug(
                "artifact cleanup failed for %s",
                path_for_prompt(path),
                exc_info=True,
            )

    cov_files: list[Path] = []
    for env_name, default in [
        ("SANDBOX_COVERAGE_FILE", "sandbox_data/coverage.json"),
        ("SANDBOX_COVERAGE_SUMMARY", "sandbox_data/coverage_summary.json"),
    ]:
        try:
            cov_files.append(_env_path(env_name, default))
        except FileNotFoundError:
            cov_files.append(_env_path(env_name, default))
    cov_files.extend([resolve_path(".coverage"), resolve_path("cov.json")])
    for path in cov_files:
        try:
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            leftovers.append(str(path))
            logger.debug(
                "coverage cleanup failed for %s",
                path_for_prompt(path),
                exc_info=True,
            )

    if shutil.which("docker"):
        try:
            try:
                _run_subprocess_with_progress(
                    ["docker", "container", "prune", "-f"],
                    capture_output=True,
                    text=True,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
                logger.warning(
                    "docker container prune timed out (%.1fs)",
                    duration,
                )
                leftovers.append("docker:container-prune")
                _record_failed_cleanup(
                    "docker:container-prune",
                    reason="docker container prune timeout",
                )
                return
            try:
                _run_subprocess_with_progress(
                    ["docker", "volume", "prune", "-f"],
                    capture_output=True,
                    text=True,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
                logger.warning(
                    "docker volume prune timed out (%.1fs)",
                    duration,
                )
                leftovers.append("docker:volume-prune")
                _record_failed_cleanup(
                    "docker:volume-prune",
                    reason="docker volume prune timeout",
                )
                return
            try:
                proc = _run_subprocess_with_progress(
                    ["docker", "ps", "-aq"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
                logger.warning(
                    "docker ps timed out during artifact cleanup (%.1fs)",
                    duration,
                )
                leftovers.append("docker:container-list")
                _record_failed_cleanup(
                    "docker:container-list",
                    reason="docker ps timeout",
                )
                return
            containers = proc.stdout.strip().splitlines()
            try:
                proc = _run_subprocess_with_progress(
                    ["docker", "volume", "ls", "-q"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
                logger.warning(
                    "docker volume ls timed out during artifact cleanup (%.1fs)",
                    duration,
                )
                leftovers.append("docker:volume-list")
                _record_failed_cleanup(
                    "docker:volume-list",
                    reason="docker volume ls timeout",
                )
                return
            volumes = proc.stdout.strip().splitlines()
            leftovers.extend(f"container:{c}" for c in containers)
            leftovers.extend(f"volume:{v}" for v in volumes)
        except Exception:
            leftovers.append("docker")
            logger.debug("docker cleanup failed", exc_info=True)

    if leftovers:
        logger.warning("residual artifacts after cleanup: %s", leftovers)
    else:
        logger.info("runtime cleanup completed successfully")


def _fallback_logger() -> logging.Logger:
    """Return a minimal logger when the main logger is unavailable."""
    fb = logging.getLogger("sandbox_runner.environment.fallback")
    if not fb.handlers:
        handler = logging.StreamHandler()
        fb.addHandler(handler)
        fb.setLevel(logging.INFO)
    return fb

if os.name == "nt" and "fcntl" not in sys.modules:
    try:
        import fcntl_compat as _fcntl
        sys.modules["fcntl"] = _fcntl
    except (ImportError, OSError) as exc:  # pragma: no cover - best effort
        logger.error("fcntl compatibility shim is required on Windows", exc_info=exc)
        raise RuntimeError("fcntl support unavailable on Windows") from exc

_SOCKET_SUPPORTS_AF_UNIX = hasattr(socket, "AF_UNIX")

try:  # pragma: no cover - optional dependency
    if os.name == "nt" and not _SOCKET_SUPPORTS_AF_UNIX:
        raise RuntimeError(
            "pyroute2 requires socket.AF_UNIX which is unavailable on Windows"
        )
    from pyroute2 import IPRoute, NSPopen, netns
except (ImportError, OSError, RuntimeError) as exc:
    IPRoute = None  # type: ignore
    NSPopen = None  # type: ignore
    netns = None  # type: ignore
    if os.name == "nt" and not _SOCKET_SUPPORTS_AF_UNIX:
        logger.info(
            "pyroute2 support disabled: %s. network emulation features will be skipped",
            exc,
        )
    else:
        logger.warning("pyroute2 import failed: %s", exc)

try:
    from faker import Faker  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Faker = None  # type: ignore

try:
    from hypothesis import strategies as _hyp_strats  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _hyp_strats = None  # type: ignore

_FAKER = Faker() if Faker is not None else None

from . import config as sandbox_config
from .input_history_db import InputHistoryDB, router as history_router
from collections import Counter
try:
    from error_logger import ErrorLogger
except ImportError as exc:  # pragma: no cover - required dependency
    logger.debug("error_logger unavailable", exc_info=exc)
    raise RuntimeError(
        "error_logger is required for sandbox operations. Install the package"
        " providing it (e.g. `pip install menace-sandbox`) or ensure the"
        " module is available on the PYTHONPATH."
    ) from exc
from knowledge_graph import KnowledgeGraph

from db_router import GLOBAL_ROUTER, init_db_router


def _env_path(name: str, default: str, *, create: bool = False) -> Path:
    """Resolve *name* from the environment to an absolute :class:`Path`.

    The value of ``name`` is looked up in the environment, falling back to
    ``default`` if unset. The resulting string is passed directly to
    :func:`resolve_path` which handles resolution relative to the repository
    root.
    """

    value = os.getenv(name, default)
    try:
        return resolve_path(value)
    except FileNotFoundError:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repo_root() / candidate

        if create:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.touch(exist_ok=True)
            return candidate.resolve()

        raise

# Persistent module usage tracking -------------------------------------------
MODULE_USAGE_PATH = _env_path(
    "SANDBOX_MODULE_USAGE_PATH",
    "sandbox_data/module_usage.json",
    create=True,
)
_MODULE_USAGE_LOCK = FileLock(str(MODULE_USAGE_PATH) + ".lock")


def record_module_usage(module_name: str) -> None:
    """Record *module_name* execution with a timestamp.

    Entries are stored in :data:`MODULE_USAGE_PATH` as a mapping of module
    paths to dictionaries of timestamp -> count. This lightweight log allows
    external runners to track which modules contribute to sandbox runs.
    """

    ts = datetime.utcnow().isoformat()
    with _RADAR_MANAGER:
        _async_radar_track(module_name)
    with _MODULE_USAGE_LOCK:
        try:
            if MODULE_USAGE_PATH.exists():
                data = json.loads(MODULE_USAGE_PATH.read_text())
            else:
                data = {}
            module_map = data.setdefault(module_name, {})
            module_map[ts] = module_map.get(ts, 0) + 1
            MODULE_USAGE_PATH.write_text(json.dumps(data, indent=2))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception(
                "module usage logging failed", extra={"module": module_name}, exc_info=exc
            )

# path to cleanup log file
_CLEANUP_LOG_PATH = _env_path("SANDBOX_CLEANUP_LOG", "sandbox_data/cleanup.log", create=True)
_CLEANUP_LOG_LOCK = threading.Lock()
POOL_LOCK_FILE = _env_path(
    "SANDBOX_POOL_LOCK",
    "sandbox_data/pool.lock",
    create=True,
)

_INPUT_HISTORY_DB: InputHistoryDB | None = None

# Shared error logger and category counters for sandbox runs.  The logger is
# initialised lazily so a ``ContextBuilder`` can be supplied by higher level
# orchestration code instead of being constructed at import time.
KNOWLEDGE_GRAPH = KnowledgeGraph()
ERROR_LOGGER: ErrorLogger | None = None
_ERROR_CONTEXT_BUILDER: ContextBuilder | None = None


def get_error_logger(context_builder: ContextBuilder) -> ErrorLogger:
    """Return a shared :class:`ErrorLogger` instance using ``context_builder``.

    The ``context_builder`` argument is required and must not be ``None``.
    """

    if context_builder is None:
        raise ValueError("context_builder must not be None")

    global ERROR_LOGGER, _ERROR_CONTEXT_BUILDER
    _ERROR_CONTEXT_BUILDER = context_builder
    if ERROR_LOGGER is None:
        ERROR_LOGGER = ErrorLogger(
            knowledge_graph=KNOWLEDGE_GRAPH,
            context_builder=context_builder,
        )
    return ERROR_LOGGER


ERROR_CATEGORY_COUNTS: Counter[str] = Counter()

# Track how many times each module/scenario combination was executed
COVERAGE_TRACKER: Dict[str, Dict[str, int]] = {}
COVERAGE_FILES: Dict[str, set[str]] = {}
COVERAGE_FUNCTIONS: Dict[str, set[str]] = {}
FUNCTION_COVERAGE_TRACKER: Dict[str, Dict[str, Dict[str, int]]] = {}


def record_module_coverage(
    module: str, files: Iterable[str], functions: Iterable[str]
) -> None:
    """Record executed ``files`` and ``functions`` for ``module``."""

    if files:
        COVERAGE_FILES.setdefault(module, set()).update(files)
    if functions:
        prefixed = []
        for fn in functions:
            name = fn.split(":", 1)[-1]
            prefixed.append(f"{module}:{name}")
        COVERAGE_FUNCTIONS.setdefault(module, set()).update(prefixed)


def load_coverage_report(report: Mapping[str, Any] | str | Path) -> list[str]:
    """Load a coverage JSON ``report`` and aggregate by module.

    Returns a flat list of executed ``"path:function"`` entries for the run.
    """

    if isinstance(report, (str, Path)):
        try:
            data = json.loads(Path(report).read_text())
        except Exception:
            return
    else:
        data = report

    files = data.get("files", {}) if isinstance(data, Mapping) else {}
    root = repo_root()
    executed_functions: list[str] = []
    for fpath, info in files.items():
        path_obj = Path(fpath)
        try:
            rel = path_obj.resolve().relative_to(root).as_posix()
        except Exception:
            rel = path_obj.as_posix()
        module = rel[:-3].replace("/", ".")
        executed = set(info.get("executed_lines", []))
        funcs: list[str] = []
        try:
            source = path_obj.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    end = getattr(node, "end_lineno", node.lineno)
                    if any(l in executed for l in range(node.lineno, end + 1)):
                        func_id = f"{rel}:{node.name}"
                        funcs.append(func_id)
                        executed_functions.append(func_id)
        except Exception:
            pass
        record_module_coverage(module, [rel], funcs)
    return executed_functions


def _functions_for_module(
    cov_map: Mapping[str, List[str]], module: str
) -> List[str]:
    """Return executed function names for ``module`` from ``cov_map``."""
    try:
        mod_obj = importlib.import_module(module)
        file = inspect.getsourcefile(mod_obj) or inspect.getfile(mod_obj)  # type: ignore[arg-type]
        root = repo_root()
        rel = Path(file).resolve().relative_to(root).as_posix()
    except Exception:
        return []
    return cov_map.get(rel, [])


@contextmanager
def create_ephemeral_repo(
    settings: SandboxSettings | None = None,
) -> Iterable[tuple[Path, Callable[..., subprocess.CompletedProcess]]]:
    """Clone the current repository into an ephemeral location.

    Depending on ``settings.sandbox_backend`` the repository is cloned into a
    temporary directory (``venv``) or mounted into a lightweight Docker
    container (``docker``). The context manager yields the cloned repository
    path and a ``run`` helper that executes commands within that repository.
    """

    settings = settings or SandboxSettings()
    backend = getattr(settings, "sandbox_backend", "venv").lower()
    repo_src = settings.sandbox_repo_path or "."

    use_docker = backend == "docker" and shutil.which("docker")

    with tempfile.TemporaryDirectory(prefix="repo_clone_") as td:
        repo_path = Path(td) / "repo"
        subprocess.check_call(["git", "clone", "--depth", "1", repo_src, str(repo_path)])

        if use_docker:
            image = getattr(settings, "sandbox_docker_image", "python:3.11-slim")

            def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, **kw: Any):
                env = env or {}
                env_args: list[str] = []
                for k, v in env.items():
                    env_args.extend(["-e", f"{k}={v}"])
                docker_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{repo_path}:/repo",
                    "-w",
                    "/repo",
                    *env_args,
                    image,
                    *cmd,
                ]
                return subprocess.run(docker_cmd, **kw)

        else:

            def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, **kw: Any):
                env_local = os.environ.copy()
                if env:
                    env_local.update(env)
                return subprocess.run(cmd, cwd=str(repo_path), env=env_local, **kw)

        yield repo_path, _run


@contextmanager
def create_ephemeral_env(
    workdir: Path,
    *,
    context_builder: ContextBuilder,
) -> Iterable[tuple[Path, Callable[..., subprocess.CompletedProcess]]]:
    """Prepare an isolated repo copy with dependencies installed.

    The repository at ``workdir`` is cloned into a temporary location and a
    Python environment (``venv`` by default, Docker when available and
    ``SANDBOX_BACKEND=docker``) is initialised with ``requirements.txt``
    installed. The context manager yields the cloned repository path and a
    ``run`` helper for executing commands inside the isolated environment.

    The provided ``context_builder`` is mandatory and no fallback builder is
    created. Environment startup time and installation failures are logged via
    :mod:`logging_utils` so metrics can be exported by callers.
    """

    if context_builder is None:
        raise ValueError("context_builder must not be None")

    get_error_logger(context_builder)
    backend = os.getenv("SANDBOX_BACKEND", "venv").lower()
    use_docker = backend == "docker" and shutil.which("docker")

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="env_") as td:
        repo_path = Path(td) / "repo"
        subprocess.check_call(["git", "clone", "--depth", "1", str(workdir), str(repo_path)])

        if use_docker:
            image = os.getenv("SANDBOX_DOCKER_IMAGE", "python:3.11-slim")

            def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, **kw: Any):
                env = env or {}
                env_args: list[str] = []
                for k, v in env.items():
                    env_args.extend(["-e", f"{k}={v}"])
                docker_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{repo_path}:/repo",
                    "-w",
                    "/repo",
                    *env_args,
                    image,
                    *cmd,
                ]
                return subprocess.run(docker_cmd, **kw)

            req = repo_path / "requirements.txt"
            if req.exists():
                try:
                    _run(
                        ["pip", "install", "-r", str(req)],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort
                    logger.error(
                        "dependency install failed",
                        extra=log_record(rc=exc.returncode, output=exc.stderr),
                    )
                    record_error(exc, context_builder=context_builder)
                    raise
        else:
            venv_dir = Path(td) / "venv"
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
            python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
            req = repo_path / "requirements.txt"
            if req.exists():
                try:
                    subprocess.run(
                        [str(python), "-m", "pip", "install", "-r", str(req)],
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=str(repo_path),
                    )
                except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort
                    logger.error(
                        "dependency install failed",
                        extra=log_record(rc=exc.returncode, output=exc.stderr),
                    )
                    record_error(exc, context_builder=context_builder)
                    raise

            def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, **kw: Any):
                env_local = os.environ.copy()
                env_local["PATH"] = f"{python.parent}{os.pathsep}{env_local.get('PATH', '')}"
                env_local["VIRTUAL_ENV"] = str(venv_dir)
                if env:
                    env_local.update(env)
                return subprocess.run(cmd, cwd=str(repo_path), env=env_local, **kw)

        elapsed = (time.perf_counter() - start) * 1000.0
        logger.info("ephemeral env ready", extra=log_record(startup_ms=int(elapsed)))
        yield repo_path, _run


def _update_coverage(
    module: str, scenario: str, functions: Iterable[str] | None = None
) -> None:
    """Increment coverage counter for ``module`` and ``functions`` under ``scenario``.

    Scenario names are normalised to their canonical form using the alias map
    from :mod:`environment_generator` so that coverage for e.g. ``schema_mismatch``
    is recorded under ``schema_drift``.  This keeps the coverage tracker in
    sync with :data:`CANONICAL_PROFILES` which now includes ``schema_drift`` and
    ``flaky_upstream``.
    """

    try:  # pragma: no cover - environment generator optional
        from menace.environment_generator import _PROFILE_ALIASES
    except ImportError as exc:  # pragma: no cover - fallback when generator unavailable
        logger.debug("environment generator unavailable", exc_info=exc)
        _PROFILE_ALIASES = {}

    canonical = _PROFILE_ALIASES.get(scenario, scenario)
    mod_map = COVERAGE_TRACKER.setdefault(module, {})
    mod_map[canonical] = mod_map.get(canonical, 0) + 1

    if functions:
        COVERAGE_FUNCTIONS.setdefault(module, set()).update(
            f"{module}:{fn.split(':',1)[-1]}" for fn in functions
        )
        func_map = FUNCTION_COVERAGE_TRACKER.setdefault(module, {})
        for fn in functions:
            name = fn.split(":", 1)[-1]
            scen_map = func_map.setdefault(name, {})
            scen_map[canonical] = scen_map.get(canonical, 0) + 1


def coverage_summary() -> Dict[str, Dict[str, Any]]:
    """Return coverage counts and executed files/functions per module."""
    try:
        from menace.environment_generator import CANONICAL_PROFILES
    except ImportError as exc:  # pragma: no cover - environment generator optional
        logger.debug("environment generator unavailable", exc_info=exc)
        profiles: List[str] = []
    else:
        profiles = list(CANONICAL_PROFILES)
    summary: Dict[str, Dict[str, Any]] = {}
    modules = (
        set(COVERAGE_TRACKER)
        | set(COVERAGE_FILES)
        | set(COVERAGE_FUNCTIONS)
        | set(FUNCTION_COVERAGE_TRACKER)
    )
    for mod in modules:
        scen_map = COVERAGE_TRACKER.get(mod, {})
        missing = [p for p in profiles if p not in scen_map]
        info: Dict[str, Any] = {"counts": dict(scen_map), "missing": missing}
        files = COVERAGE_FILES.get(mod)
        if files:
            info["files"] = sorted(files)
        funcs = COVERAGE_FUNCTIONS.get(mod)
        if funcs:
            info["functions"] = sorted(funcs)
        f_counts = FUNCTION_COVERAGE_TRACKER.get(mod)
        if f_counts:
            info["function_counts"] = {
                f"{mod}:{fn}": dict(counts) for fn, counts in f_counts.items()
            }
        summary[mod] = info
    return summary


def verify_scenario_coverage(
    *, raise_on_missing: bool = False
) -> Dict[str, List[str]]:
    """Verify that each module covers all canonical scenarios.

    Scenario names stored in :data:`COVERAGE_TRACKER` may include legacy
    aliases. This helper normalises those aliases and recomputes missing
    scenarios against the canonical list so that new profiles such as
    ``schema_drift`` and ``flaky_upstream`` are validated correctly.

    Parameters
    ----------
    raise_on_missing:
        When ``True`` a :class:`RuntimeError` is raised if coverage gaps are
        detected. Otherwise warnings are logged.

    Returns
    -------
    Mapping of modules to lists of missing scenario names.
    """

    summary = coverage_summary()
    try:  # pragma: no cover - environment generator optional
        from menace.environment_generator import (
            CANONICAL_PROFILES,
            _PROFILE_ALIASES,
        )
    except ImportError as exc:  # pragma: no cover - graceful fallback
        logger.debug("environment generator unavailable", exc_info=exc)
        CANONICAL_PROFILES = []
        _PROFILE_ALIASES = {}

    profiles = set(CANONICAL_PROFILES)
    missing: Dict[str, List[str]] = {}
    for mod, info in summary.items():
        counts = dict(info.get("counts", {}))
        for alias, canonical in _PROFILE_ALIASES.items():
            if alias in counts:
                counts[canonical] = counts.get(canonical, 0) + counts.pop(alias)
        absent = [p for p in profiles if p not in counts]
        if absent:
            missing[mod] = absent
            msg = f"module {mod} missing scenarios: {', '.join(absent)}"
            if raise_on_missing:
                logger.error(msg)
            else:
                logger.warning(msg)
        func_counts = info.get("function_counts", {})
        for fn, fc in func_counts.items():
            fcounts = dict(fc)
            for alias, canonical in _PROFILE_ALIASES.items():
                if alias in fcounts:
                    fcounts[canonical] = fcounts.get(canonical, 0) + fcounts.pop(alias)
            fabsent = [p for p in profiles if p not in fcounts]
            if fabsent:
                missing[fn] = fabsent
                msg = f"function {fn} missing scenarios: {', '.join(fabsent)}"
                if raise_on_missing:
                    logger.error(msg)
                else:
                    logger.warning(msg)
    if raise_on_missing and missing:
        raise RuntimeError("scenario coverage incomplete")
    return missing


def save_coverage_data() -> None:
    """Persist :data:`COVERAGE_TRACKER` and its summary to ``sandbox_data`` files."""

    path = _env_path("SANDBOX_COVERAGE_FILE", "sandbox_data/coverage.json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "modules": COVERAGE_TRACKER,
            "functions": FUNCTION_COVERAGE_TRACKER,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    except OSError as exc:  # pragma: no cover - best effort only
        logger.exception(
            "failed to save coverage data",
            extra=log_record(path=str(path)),
            exc_info=exc,
        )

    summary_path = _env_path(
        "SANDBOX_COVERAGE_SUMMARY", "sandbox_data/coverage_summary.json"
    )
    summary = coverage_summary()
    try:  # pragma: no cover - best effort only
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
    except OSError as exc:
        logger.exception(
            "failed to save coverage summary",
            extra=log_record(path=str(summary_path)),
            exc_info=exc,
        )


def _scenario_summary_path() -> Path:
    data_dir = _env_path("SANDBOX_DATA_DIR", "sandbox_data")
    default = str(data_dir / "scenario_summary.json")
    return _env_path("SANDBOX_SCENARIO_SUMMARY", default)


def save_scenario_summary(
    synergy_data: Dict[str, Dict[str, list]],
    roi_deltas: Dict[str, float] | None = None,
    worst_scenario: str | None = None,
) -> Dict[str, Any]:
    """Persist aggregated ROI and success metrics per scenario."""

    summary: Dict[str, Dict[str, float]] = {}
    for scen, data in synergy_data.items():
        roi_total = sum(float(r) for r in data.get("roi", []))
        metrics = data.get("metrics", [])
        failure_runs = 0
        for m in metrics:
            fails = float(m.get("hostile_failures", 0.0)) + float(
                m.get("misuse_failures", 0.0)
            )
            if fails > 0:
                failure_runs += 1
        summary[scen] = {
            "roi": roi_total,
            "successes": max(0, len(metrics) - failure_runs),
            "failures": failure_runs,
        }
    if roi_deltas:
        for scen, delta in roi_deltas.items():
            summary.setdefault(scen, {})["roi_delta"] = float(delta)
    result = {"scenarios": summary, "worst_scenario": worst_scenario}
    path = _scenario_summary_path()
    try:  # pragma: no cover - best effort only
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
    except (OSError, TypeError, ValueError) as exc:
        logger.exception(
            "failed to save scenario summary",
            extra={"path": path_for_prompt(path)},
            exc_info=exc,
        )
    return result


def load_scenario_summary() -> Dict[str, Any]:
    """Return the saved scenario summary if available."""
    path = _scenario_summary_path()
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "failed to load scenario summary",
            extra={"path": path_for_prompt(path)},
            exc_info=exc,
        )
        return {}


def record_error(
    exc: Exception,
    *,
    fatal: bool = False,
    context_builder: ContextBuilder,
) -> None:
    """Log *exc* via :class:`ErrorLogger` and track its category and severity.

    The ``context_builder`` argument is mandatory and must not be ``None``.
    """

    if context_builder is None:
        raise ValueError("context_builder must not be None")

    logger_obj = get_error_logger(context_builder)
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _, category, _ = logger_obj.classifier.classify_details(exc, stack)
    logger_obj.log(exc, None, None)
    ERROR_CATEGORY_COUNTS[category.value] += 1
    severity = "fatal" if fatal else "recoverable"
    try:
        environment_failure_total.labels(reason=category.value).inc()
        environment_failure_severity_total.labels(severity=severity).inc()
    except (ValueError, RuntimeError) as metrics_exc:  # pragma: no cover - metrics best effort
        logger.exception(
            "failed to update environment failure metrics", exc_info=metrics_exc
        )


def _get_history_db() -> InputHistoryDB:
    """Return cached :class:`InputHistoryDB` instance."""
    global _INPUT_HISTORY_DB
    if _INPUT_HISTORY_DB is None:
        path = str(_env_path("SANDBOX_INPUT_HISTORY", "sandbox_data/input_history.db"))
        _INPUT_HISTORY_DB = InputHistoryDB(path)
    return _INPUT_HISTORY_DB


def _load_diagnostic_manager() -> bool:
    """Attempt to import the optional diagnostic manager lazily."""

    global DiagnosticManager, ResolutionRecord
    if DiagnosticManager is not None and ResolutionRecord is not None:
        return True
    try:  # pragma: no cover - optional dependency
        from menace.diagnostic_manager import (
            DiagnosticManager as _DiagnosticManager,
            ResolutionRecord as _ResolutionRecord,
        )
    except ImportError as exc:
        get_logger(__name__).debug("diagnostic manager unavailable", exc_info=exc)
        DiagnosticManager = None  # type: ignore[assignment]
        ResolutionRecord = None  # type: ignore[assignment]
        return False
    DiagnosticManager = _DiagnosticManager  # type: ignore[assignment]
    ResolutionRecord = _ResolutionRecord  # type: ignore[assignment]
    return True


_DIAGNOSTIC: DiagnosticManager | None = None


def init_diagnostic_manager(
    context_builder: ContextBuilder,
) -> None:
    """Initialise the optional ``DiagnosticManager`` if available.

    The ``context_builder`` argument is required and must not be ``None``.
    """

    if context_builder is None:
        raise ValueError("context_builder must not be None")

    global _DIAGNOSTIC
    if _DIAGNOSTIC is not None:
        return
    if not _load_diagnostic_manager():
        return
    try:
        _DIAGNOSTIC = DiagnosticManager(context_builder=context_builder)
    except (OSError, RuntimeError) as exc:  # pragma: no cover - diagnostics optional
        logger.warning(
            "diagnostic manager unavailable",
            extra=log_record(module=__name__),
            exc_info=exc,
        )
        _DIAGNOSTIC = None


def _log_diagnostic(issue: str, success: bool) -> None:
    """Record a resolution attempt with ``DiagnosticManager`` if available."""
    if _DIAGNOSTIC is None or ResolutionRecord is None:
        return
    try:
        _DIAGNOSTIC.log.add(ResolutionRecord(issue, "retry", success))
        if not success:
            _DIAGNOSTIC.error_bot.handle_error(issue)
    except (OSError, AttributeError) as exc:
        logger.exception(
            "diagnostic logging failed",
            extra=log_record(issue=issue, success=success),
            exc_info=exc,
        )


# ----------------------------------------------------------------------
class _DangerVisitor(ast.NodeVisitor):
    """Collect suspicious patterns without executing code."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.files_written: List[str] = []
        self.flags: List[str] = []
        self.imports: List[str] = []
        self.attributes: List[str] = []

    def visit_Call(self, node: ast.Call) -> Any:
        name = self._name(node.func)
        if name:
            self.calls.append(name)
        lowered = name.lower() if name else ""
        if lowered in {"eval", "exec"}:
            self.flags.append(f"dangerous call {name}")
        if lowered.startswith("subprocess") or lowered.startswith("os.system"):
            self.flags.append(f"process call {name}")
        if lowered == "open":
            mode = "r"
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if any(m in mode for m in ("w", "a", "+")):
                path = self._literal_arg(node.args[0]) if node.args else "?"
                self.files_written.append(str(path))
                self.flags.append("file write")
        if lowered.startswith(("requests", "socket")):
            self.flags.append(f"network call {name}")
        if lowered in {
            "os.setuid",
            "os.setgid",
            "os.seteuid",
            "os.setegid",
        }:
            self.flags.append(f"privilege escalation {name}")
        if lowered.startswith("subprocess") or lowered.startswith("os.system"):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and "sudo" in arg.value
                ):
                    self.flags.append("privilege escalation sudo")
                    break
        if "reward" in lowered:
            self.flags.append(f"reward manipulation {name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        name = self._name(node)
        self.attributes.append(name)
        if name in {
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "requests.get",
            "requests.post",
        }:
            self.flags.append(f"risky attribute {name}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            mod = alias.name.split(".")[0]
            self.imports.append(alias.name)
            if mod in {"socket", "requests", "subprocess", "ctypes"}:
                self.flags.append(f"import dangerous module {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = node.module or ""
        self.imports.append(mod)
        if mod.split(".")[0] in {"socket", "requests", "subprocess", "ctypes"}:
            self.flags.append(f"import dangerous module {mod}")
        self.generic_visit(node)

    def _name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            value = self._name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return ""

    def _literal_arg(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        return ""


# ----------------------------------------------------------------------
def static_behavior_analysis(code_str: str) -> Dict[str, Any]:
    """Return dictionary describing risky constructs in ``code_str``."""
    logger.debug("starting static analysis of code (%d chars)", len(code_str))
    result: Dict[str, Any] = {
        "calls": [],
        "files_written": [],
        "flags": [],
        "imports": [],
        "attributes": [],
    }
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        result["flags"].append(f"syntax error: {exc}")
        return result
    visitor = _DangerVisitor()
    visitor.visit(tree)
    result["calls"] = visitor.calls
    result["files_written"] = visitor.files_written
    result["flags"] = visitor.flags
    result["imports"] = visitor.imports
    result["attributes"] = visitor.attributes

    patterns = [r"\bsubprocess\b", r"\bos\.system\b", r"eval\(", r"exec\("]
    if any(re.search(p, code_str) for p in patterns):
        result.setdefault("regex_flags", []).append("raw_dangerous_pattern")

    logger.debug(
        "static analysis result: %s",
        {k: v for k, v in result.items() if v},
    )
    return result


# ----------------------------------------------------------------------
# Docker container pooling support
try:  # pragma: no cover - optional dependency
    import docker  # type: ignore
    from docker.errors import DockerException, APIError
except Exception as exc:  # pragma: no cover - docker may be unavailable
    logger.info(
        "Docker SDK unavailable (%s); container pooling features will be skipped",
        exc,
    )
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore
    APIError = Exception  # type: ignore
    _DOCKER_CLIENT = None
else:
    _DOCKER_CLIENT = None


_DOCKER_CLIENT_TIMEOUT = float(os.getenv("SANDBOX_DOCKER_CLIENT_TIMEOUT", "5"))
_DOCKER_PING_TIMEOUT = float(
    os.getenv("SANDBOX_DOCKER_PING_TIMEOUT", str(max(1.0, _DOCKER_CLIENT_TIMEOUT)))
)


def _close_docker_client(client: Any | None) -> None:
    """Best effort close for Docker client instances."""

    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.debug("failed to close docker client", exc_info=True)


def _configure_docker_client(client: Any) -> None:
    """Tune client level timeouts for more responsive failures."""

    try:
        api = getattr(client, "api", None)
        if api is not None:
            if hasattr(api, "timeout"):
                api.timeout = max(float(getattr(api, "timeout", 0.0) or 0.0), _DOCKER_CLIENT_TIMEOUT)
            http_client = getattr(api, "client", None)
            if http_client is not None and hasattr(http_client, "timeout"):
                http_client.timeout = _DOCKER_CLIENT_TIMEOUT
    except Exception:
        logger.debug("failed to configure docker client timeouts", exc_info=True)


def _ping_docker(
    client: Any,
    *,
    timeout: float | None = None,
) -> tuple[bool, Exception | None]:
    """Return ``(True, None)`` when ``client`` responds to ``ping`` within ``timeout``."""

    if client is None:
        return False, None

    methods: list[Callable[..., object]] = []
    ping_fn = getattr(client, "ping", None)
    if callable(ping_fn):
        methods.append(ping_fn)
    api = getattr(client, "api", None)
    if api is not None:
        ping_fn = getattr(api, "ping", None)
        if callable(ping_fn):
            methods.append(ping_fn)

    last_exc: Exception | None = None
    for method in methods:
        try:
            if timeout is not None:
                try:
                    method(timeout=timeout)
                except TypeError:
                    method()
            else:
                method()
            return True, None
        except DockerException as exc:  # pragma: no cover - depends on docker availability
            last_exc = exc
        except Exception as exc:  # pragma: no cover - defensive guard
            last_exc = exc
    return False, last_exc


def _create_docker_client() -> tuple[Any | None, Exception | None]:
    """Return a connected Docker client or ``(None, exc)`` when unavailable."""

    if docker is None:
        return None, None
    try:
        client = docker.from_env(timeout=_DOCKER_CLIENT_TIMEOUT)
    except DockerException as exc:  # pragma: no cover - docker may be unavailable
        return None, exc
    except Exception as exc:  # pragma: no cover - defensive guard
        return None, exc

    _configure_docker_client(client)
    healthy, exc = _ping_docker(client, timeout=_DOCKER_PING_TIMEOUT)
    if healthy:
        return client, None

    _close_docker_client(client)
    return None, exc


if docker is not None:
    _DOCKER_CLIENT, _DOCKER_STARTUP_ERROR = _create_docker_client()
    if _DOCKER_CLIENT is None and _DOCKER_STARTUP_ERROR is not None:
        logger.info(
            "Docker client unavailable (%s); container pooling features disabled",
            _DOCKER_STARTUP_ERROR,
        )

_CONTAINER_POOL_SIZE = int(os.getenv("SANDBOX_CONTAINER_POOL_SIZE", "2"))
_CONTAINER_IDLE_TIMEOUT = float(os.getenv("SANDBOX_CONTAINER_IDLE_TIMEOUT", "300"))
_POOL_CLEANUP_INTERVAL = float(os.getenv("SANDBOX_POOL_CLEANUP_INTERVAL", "60"))
_WORKER_CHECK_INTERVAL = float(os.getenv("SANDBOX_WORKER_CHECK_INTERVAL", "30"))
_CLEANUP_SUBPROCESS_TIMEOUT = float(
    os.getenv("SANDBOX_CLEANUP_SUBPROCESS_TIMEOUT", "30")
)
_CLEANUP_WATCHDOG_MARGIN = float(
    os.getenv("SANDBOX_CLEANUP_WATCHDOG_MARGIN", "30")
)


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Return ``True`` when *name* is set to a truthy value."""

    raw = os.getenv(name)
    if raw is None:
        return default
    candidate = str(raw).strip().lower()
    if not candidate:
        return default
    if candidate in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if candidate in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


_SANDBOX_DISABLE_CLEANUP = _env_flag("SANDBOX_DISABLE_CLEANUP")
_CLEANUP_DISABLE_NOTICE_LOCK = threading.Lock()
_CLEANUP_DISABLE_NOTICES: set[str] = set()


def _log_cleanup_disabled(context: str | None = None) -> None:
    """Log that cleanup automation has been disabled, once per context."""

    if not _SANDBOX_DISABLE_CLEANUP:
        return

    key = context or "<default>"
    with _CLEANUP_DISABLE_NOTICE_LOCK:
        if key in _CLEANUP_DISABLE_NOTICES:
            return
        message = "sandbox cleanup automation disabled via SANDBOX_DISABLE_CLEANUP"
        if context:
            message = f"{message}; skipped {context}"
        try:
            logger.info(message)
        except Exception:
            # Logging should never be fatal for cleanup disable notices.
            pass
        _CLEANUP_DISABLE_NOTICES.add(key)

_HEARTBEAT_GUARD_INTERVAL = float(
    os.getenv("SANDBOX_HEARTBEAT_GUARD_INTERVAL", "1.0")
)
_DEFAULT_HEARTBEAT_GUARD_MAX = 480.0 if os.name == "nt" else 300.0
_HEARTBEAT_GUARD_MAX_DURATION = float(
    os.getenv(
        "SANDBOX_HEARTBEAT_GUARD_MAX_DURATION",
        str(_DEFAULT_HEARTBEAT_GUARD_MAX),
    )
)


def _coerce_positive_float(value: str | None, default: float) -> float:
    """Return ``value`` parsed as a positive float or ``default`` on failure."""

    if not value:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0.0:
        return default
    return parsed


def _coerce_positive_int(value: str | None, default: int) -> int:
    """Return ``value`` parsed as a positive integer or ``default`` on failure."""

    if not value:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed
def _is_windows_platform() -> bool:
    """Return ``True`` when running on Windows or Windows Subsystem for Linux."""

    if os.name == "nt":
        return True
    # Heuristic: WSL exports ``WSL_DISTRO_NAME``/``WSL_INTEROP`` variables.
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    return False


def _docker_host_uses_named_pipe() -> bool:
    """Return ``True`` when Docker communicates over a Windows named pipe."""

    host = (os.environ.get("DOCKER_HOST") or "").lower()
    if not host:
        return False
    if host.startswith("npipe://") or host.startswith("npipe:"):
        return True
    return "\\\pipe\\" in host or "//./pipe/" in host


_WINDOWS_DOCKER_CONTEXT = _is_windows_platform() or _docker_host_uses_named_pipe()
_WINDOWS_WATCHDOG_FACTOR = 1.0
if _WINDOWS_DOCKER_CONTEXT:
    _WINDOWS_WATCHDOG_FACTOR = max(
        1.0,
        _coerce_positive_float(
            os.getenv("SANDBOX_WINDOWS_WATCHDOG_FACTOR"),
            3.0,
        ),
    )


class _AdaptiveWatchdogBudget:
    """Adaptive limits for the cleanup watchdog to avoid false positives."""

    def __init__(
        self,
        *,
        history_size: int = 40,
        percentile: float = 0.95,
        base_backoff: float = 15.0,
        max_backoff: float = 240.0,
        windows_bias: float = 1.0,
    ) -> None:
        self._history: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, history_size))
        )
        self._restarts: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._percentile = percentile if 0.0 < percentile <= 1.0 else 0.95
        self._base_backoff = max(0.0, base_backoff)
        self._max_backoff = max(self._base_backoff, max_backoff)
        self._windows_bias = max(1.0, windows_bias)

    def record(self, worker: str, duration: float) -> None:
        """Track ``duration`` for ``worker`` and reset restart counters."""

        if duration <= 0.0:
            return
        with self._lock:
            self._history[worker].append(float(duration))
            self._restarts.pop(worker, None)

    def effective_limit(self, worker: str, baseline: float) -> float:
        """Return a watchdog limit adjusted for observed runtime variance."""

        limit = max(0.0, float(baseline))
        with self._lock:
            history = list(self._history.get(worker, ()))
            restarts = self._restarts.get(worker, 0)

        if history and self._percentile > 0.0:
            ordered = sorted(history)
            idx = max(0, int(math.ceil(len(ordered) * self._percentile)) - 1)
            percentile_value = ordered[idx]
            limit = max(limit, float(percentile_value) + _CLEANUP_WATCHDOG_MARGIN)

        if restarts and self._base_backoff > 0.0:
            penalty = min(
                self._base_backoff * (2 ** max(0, restarts - 1)),
                self._max_backoff,
            )
            limit = max(limit, float(baseline) + penalty)

        if self._windows_bias > 1.0:
            limit = max(limit, float(baseline) * self._windows_bias)

        return limit

    def note_restart(self, worker: str) -> None:
        """Increase the restart counter for ``worker`` to widen future limits."""

        with self._lock:
            self._restarts[worker] = self._restarts.get(worker, 0) + 1

    def reset(self, worker: str) -> None:
        """Clear restart counters when the worker stabilises."""

        with self._lock:
            self._restarts.pop(worker, None)


_WATCHDOG_HISTORY_SIZE = _coerce_positive_int(
    os.getenv("SANDBOX_WATCHDOG_HISTORY"),
    40,
)
_WATCHDOG_PERCENTILE = min(
    max(
        _coerce_positive_float(os.getenv("SANDBOX_WATCHDOG_PERCENTILE"), 0.95),
        0.01,
    ),
    1.0,
)
_WATCHDOG_BACKOFF = _coerce_positive_float(
    os.getenv("SANDBOX_WATCHDOG_BACKOFF"),
    15.0,
)
_WATCHDOG_BACKOFF_MAX = _coerce_positive_float(
    os.getenv("SANDBOX_WATCHDOG_BACKOFF_MAX"),
    240.0,
)

_WATCHDOG_BUDGET = _AdaptiveWatchdogBudget(
    history_size=_WATCHDOG_HISTORY_SIZE,
    percentile=_WATCHDOG_PERCENTILE,
    base_backoff=_WATCHDOG_BACKOFF,
    max_backoff=_WATCHDOG_BACKOFF_MAX,
    windows_bias=_WINDOWS_WATCHDOG_FACTOR,
)
_CONTAINER_MAX_LIFETIME = float(os.getenv("SANDBOX_CONTAINER_MAX_LIFETIME", "3600"))
_CONTAINER_DISK_LIMIT_STR = os.getenv("SANDBOX_CONTAINER_DISK_LIMIT", "0")
_CONTAINER_DISK_LIMIT = 0
_CONTAINER_USER = os.getenv("SANDBOX_CONTAINER_USER")
_MAX_CONTAINER_COUNT = int(os.getenv("SANDBOX_MAX_CONTAINER_COUNT", "10"))
_MAX_OVERLAY_COUNT = int(os.getenv("SANDBOX_MAX_OVERLAY_COUNT", "0"))

# label applied to pooled containers; overridable via SANDBOX_POOL_LABEL
_POOL_LABEL = os.getenv("SANDBOX_POOL_LABEL", "menace_sandbox_pool")

_CONTAINER_POOLS: Dict[str, List[Any]] = {}
_CONTAINER_DIRS: Dict[str, str] = {}
_CONTAINER_LAST_USED: Dict[str, float] = {}
_CONTAINER_CREATED: Dict[str, float] = {}
_CONTAINER_FINALIZERS: Dict[str, weakref.finalize] = {}
_POOL_LOCK = threading.Lock()
_WARMUP_TASKS: Dict[str, Any] = {}
_CLEANUP_TASK: Any | None = None
_REAPER_TASK: Any | None = None
_WORKER_CHECK_TIMER: threading.Timer | None = None

_EVENT_THREAD: threading.Thread | None = None
_EVENT_STOP: threading.Event | None = None

_BACKGROUND_LOOP: asyncio.AbstractEventLoop | None = None
_BACKGROUND_THREAD: threading.Thread | None = None


def _get_event_loop() -> asyncio.AbstractEventLoop | None:
    """Return a running event loop or ``None`` if unavailable."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return _BACKGROUND_LOOP


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Start a background event loop if not already running."""
    global _BACKGROUND_LOOP, _BACKGROUND_THREAD
    if _BACKGROUND_LOOP is None:
        _BACKGROUND_LOOP = asyncio.new_event_loop()

        def _runner(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        _BACKGROUND_THREAD = threading.Thread(
            target=_runner,
            args=(_BACKGROUND_LOOP,),
            daemon=True,
            name="sandbox-bg-loop",
        )
        _BACKGROUND_THREAD.start()
    return _BACKGROUND_LOOP


def stop_background_loop() -> None:
    """Stop ``_BACKGROUND_LOOP`` and join ``_BACKGROUND_THREAD``."""
    global _BACKGROUND_LOOP, _BACKGROUND_THREAD
    loop = _BACKGROUND_LOOP
    thread = _BACKGROUND_THREAD
    if loop is not None:
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception as exc:
            logger.warning("failed stopping background loop: %s", exc)
    if thread is not None:
        try:
            thread.join(timeout=1.0)
        except Exception as exc:
            logger.warning("failed joining background thread: %s", exc)
    if loop is not None:
        try:
            loop.close()
        except Exception as exc:
            logger.warning("failed closing background loop: %s", exc)
    _BACKGROUND_LOOP = None
    _BACKGROUND_THREAD = None


def _schedule_coroutine(coro: asyncio.coroutines.Coroutine[Any, Any, Any]) -> Any:
    """Schedule ``coro`` on an available event loop."""
    loop = _get_event_loop()
    if loop is not None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if loop is running:
                return loop.create_task(coro)
            return asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
            logger.debug("failed to schedule coroutine on existing loop", exc_info=True)
            coro.close()
            return None
    try:
        loop = _ensure_background_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception:
        logger.debug("failed to schedule coroutine on background loop", exc_info=True)
        coro.close()
        return None


_CREATE_FAILURES: Counter[str] = Counter()
_CONSECUTIVE_CREATE_FAILURES: Counter[str] = Counter()
_CREATE_BACKOFF_BASE = float(os.getenv("SANDBOX_CONTAINER_BACKOFF", "0.5"))
_CREATE_RETRY_LIMIT = int(os.getenv("SANDBOX_CONTAINER_RETRIES", "3"))
_POOL_METRICS_FILE = _env_path(
    "SANDBOX_POOL_METRICS_FILE", "sandbox_data/pool_failures.json", create=True
)


def _suspend_cleanup_workers(reason: str | None = None) -> None:
    """Cancel background cleanup workers when Docker is unavailable."""

    global _CLEANUP_TASK, _REAPER_TASK, _WORKER_CHECK_TIMER

    tasks = {
        "cleanup": _CLEANUP_TASK,
        "reaper": _REAPER_TASK,
    }
    for name, task in tasks.items():
        if task is None:
            continue
        cancel = getattr(task, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                logger.debug("failed to cancel %s worker", name, exc_info=True)
    _CLEANUP_TASK = None
    _REAPER_TASK = None
    _WORKER_ACTIVITY["cleanup"] = False
    _WORKER_ACTIVITY["reaper"] = False

    timer = _WORKER_CHECK_TIMER
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            logger.debug("failed to cancel cleanup watchdog timer", exc_info=True)
        _WORKER_CHECK_TIMER = None

    stopper = globals().get("stop_container_event_listener")
    if callable(stopper):
        try:
            stopper()
        except Exception:
            logger.debug(
                "failed to stop container event listener during docker suspension",
                exc_info=True,
            )

    if reason:
        try:
            logger.warning(reason)
        except Exception as exc:
            _fallback_logger().warning(
                "cleanup suspension notice failed: %s", exc,
                exc_info=True,
            )
_FAILURE_WARNING_THRESHOLD = int(os.getenv("SANDBOX_POOL_FAIL_THRESHOLD", "5"))
_CLEANUP_METRICS: Counter[str] = Counter()
_STALE_CONTAINERS_REMOVED = 0
_STALE_VMS_REMOVED = 0
_CLEANUP_FAILURES = 0
_FORCE_KILLS = 0
_RUNTIME_VMS_REMOVED = 0
_OVERLAY_CLEANUP_FAILURES = 0
_ACTIVE_CONTAINER_LIMIT_REACHED = 0
_ACTIVE_OVERLAY_LIMIT_REACHED = 0
_CLEANUP_RETRY_SUCCESSES = 0
_CLEANUP_RETRY_FAILURES = 0
_CONSECUTIVE_CLEANUP_FAILURES = 0
_CLEANUP_ALERT_THRESHOLD = int(os.getenv("SANDBOX_CLEANUP_ALERT_THRESHOLD", "3"))
_MAX_FAILURE_ATTEMPTS = int(os.getenv("SANDBOX_PRUNE_THRESHOLD", "5"))
_WATCHDOG_METRICS: Counter[str] = Counter()
_CLEANUP_DURATIONS = {"cleanup": 0.0, "reaper": 0.0}
_CLEANUP_CURRENT_RUNTIME = {"cleanup": 0.0, "reaper": 0.0}
_WORKER_ACTIVITY = {"cleanup": False, "reaper": False}
_LAST_CLEANUP_TS = time.monotonic()
_LAST_REAPER_TS = time.monotonic()

# Optional cleanup of sandbox Docker volumes and networks
_PRUNE_VOLUMES = str(os.getenv("SANDBOX_PRUNE_VOLUMES", "0")).lower() not in {
    "0",
    "false",
    "no",
    "",
}
_PRUNE_NETWORKS = str(os.getenv("SANDBOX_PRUNE_NETWORKS", "0")).lower() not in {
    "0",
    "false",
    "no",
    "",
}

_ACTIVE_CONTAINERS_FILE = _env_path(
    "SANDBOX_ACTIVE_CONTAINERS",
    "sandbox_data/active_containers.json",
    create=True,
)

_ACTIVE_OVERLAYS_FILE = _env_path(
    "SANDBOX_ACTIVE_OVERLAYS",
    "sandbox_data/active_overlays.json",
    create=True,
)

_FAILED_OVERLAYS_FILE = _env_path(
    "SANDBOX_FAILED_OVERLAYS",
    "sandbox_data/failed_overlays.json",
    create=True,
)

FAILED_CLEANUP_FILE = _env_path(
    "SANDBOX_FAILED_CLEANUP",
    "sandbox_data/failed_cleanup.json",
    create=True,
)

# timestamp of last automatic purge
_LAST_AUTOPURGE_FILE = _env_path(
    "SANDBOX_LAST_AUTOPURGE",
    "sandbox_data/last_autopurge",
    create=True,
)

# age threshold for automatic purge invocation
_SANDBOX_AUTOPURGE_THRESHOLD = 0.0
_LAST_AUTOPURGE_TS = 0.0

# file tracking persistent cleanup statistics
_CLEANUP_STATS_FILE = _env_path(
    "SANDBOX_CLEANUP_STATS", "sandbox_data/cleanup_stats.json", create=True
)

# duration after which stray overlay directories are purged
# defined later once _parse_timespan is available
_OVERLAY_MAX_AGE = 0.0

# threshold for logging persistent cleanup failures
# defined later once _parse_timespan is available
_FAILED_CLEANUP_ALERT_AGE = 0.0

_POOL_FILE_LOCK = FileLock(str(POOL_LOCK_FILE))
_PURGE_FILE_LOCK = FileLock(str(POOL_LOCK_FILE) + ".purge")

# timeout used when acquiring the pool file lock during cleanup to avoid
# crashing when another process already holds it.
_POOL_LOCK_ACQUIRE_TIMEOUT = 30.0

# locks protecting active container and overlay records
_ACTIVE_CONTAINERS_LOCK = FileLock(str(_ACTIVE_CONTAINERS_FILE) + ".lock")
_ACTIVE_OVERLAYS_LOCK = FileLock(str(_ACTIVE_OVERLAYS_FILE) + ".lock")


def _release_pool_lock() -> None:
    """Release the pool file lock if held."""
    try:
        if _POOL_FILE_LOCK.is_locked:
            _POOL_FILE_LOCK.release()
        try:
            os.remove(_POOL_FILE_LOCK.lock_file)
        except FileNotFoundError:
            logger.debug("pool lock file already removed")
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.warning("failed removing pool lock file: %s", exc)
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.warning("failed releasing pool lock: %s", exc)


async def _acquire_pool_lock() -> None:
    """Acquire the pool file lock asynchronously."""
    await asyncio.to_thread(_POOL_FILE_LOCK.acquire)


@asynccontextmanager
async def pool_lock() -> Any:
    """Async context manager acquiring the pool file lock."""
    await _acquire_pool_lock()
    try:
        yield
    finally:
        _release_pool_lock()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON ``data`` to ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as fh:
        json.dump(data, fh)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    os.replace(tmp, path)


T = TypeVar("T")


def _read_json_default(path: Path, default: T, *, log_context: str) -> T:
    """Return JSON data from ``path`` or ``default`` when unavailable."""

    try:
        if not path.exists():
            return default
        text = path.read_text()
    except FileNotFoundError:
        return default
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("%s %s: %s", log_context, path, exc)
        return default

    if not text.strip():
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("%s %s: %s", log_context, path, exc)
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.warning("%s %s: %s", log_context, path, exc)
    return default


def _read_active_containers() -> List[str]:
    """Return list of active container IDs from file."""
    with _ACTIVE_CONTAINERS_LOCK:
        return _read_active_containers_unlocked()


def _read_active_containers_unlocked() -> List[str]:
    """Read active container IDs without acquiring the lock."""
    data = _read_json_default(
        _ACTIVE_CONTAINERS_FILE,
        [],
        log_context="failed reading active containers",
    )
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _write_active_containers(ids: List[str]) -> None:
    """Persist ``ids`` to the active containers file."""
    with _ACTIVE_CONTAINERS_LOCK:
        _write_active_containers_unlocked(ids)


def _write_active_containers_unlocked(ids: List[str]) -> None:
    """Write active container IDs without acquiring the lock."""
    try:
        _atomic_write_json(_ACTIVE_CONTAINERS_FILE, ids)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning(
            "failed writing active containers %s: %s", _ACTIVE_CONTAINERS_FILE, exc
        )


def _record_active_container(cid: str) -> None:
    """Add ``cid`` to the active containers file."""
    with _ACTIVE_CONTAINERS_LOCK:
        ids = _read_active_containers_unlocked()
        if cid not in ids:
            ids.append(cid)
            _write_active_containers_unlocked(ids)



def _remove_active_container(cid: str) -> None:
    """Remove ``cid`` from the active containers file."""
    with _ACTIVE_CONTAINERS_LOCK:
        ids = _read_active_containers_unlocked()
        if cid in ids:
            ids.remove(cid)
            _write_active_containers_unlocked(ids)
    with _POOL_LOCK:
        fin = _CONTAINER_FINALIZERS.pop(cid, None)
    if fin is not None:
        fin.detach()


def _finalize_orphan(cid: str, dir_path: str | None) -> None:
    """Attempt to remove a leaked container and associated directory."""
    try:
        _run_subprocess_with_progress(
            ["docker", "rm", "-f", cid],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
        logger.warning(
            "orphan container remove timed out for %s (%.1fs)",
            cid,
            duration,
        )
        _record_failed_cleanup(
            f"container:{cid}",
            reason="docker rm timeout",
        )
        return
    except Exception:
        logger.exception("orphan container remove failed for %s", cid)
    _remove_active_container(cid)
    if dir_path:
        try:
            shutil.rmtree(dir_path)
        except Exception:
            logger.exception(
                "orphan directory removal failed for %s",
                path_for_prompt(dir_path),
            )


def _register_container_finalizer(container: Any) -> None:
    """Register a finalizer to clean up ``container`` if leaked."""
    cid = getattr(container, "id", None)
    if not cid:
        return
    with _POOL_LOCK:
        if cid in _CONTAINER_FINALIZERS:
            return
        dir_path = _CONTAINER_DIRS.get(cid)
        fin = weakref.finalize(container, _finalize_orphan, cid, dir_path)
        _CONTAINER_FINALIZERS[cid] = fin


def _read_active_overlays() -> List[str]:
    """Return list of active overlay directories from file."""
    with _ACTIVE_OVERLAYS_LOCK:
        return _read_active_overlays_unlocked()


def _read_active_overlays_unlocked() -> List[str]:
    """Read active overlays without acquiring the lock."""
    data = _read_json_default(
        _ACTIVE_OVERLAYS_FILE,
        [],
        log_context="failed reading active overlays",
    )
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _write_active_overlays(paths: List[str]) -> None:
    """Persist ``paths`` to the active overlays file."""
    with _ACTIVE_OVERLAYS_LOCK:
        _write_active_overlays_unlocked(paths)


def _write_active_overlays_unlocked(paths: List[str]) -> None:
    """Write overlay paths without acquiring the lock."""
    try:
        _atomic_write_json(_ACTIVE_OVERLAYS_FILE, paths)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning(
            "failed writing active overlays %s: %s", _ACTIVE_OVERLAYS_FILE, exc
        )


def _record_active_overlay(path: str) -> None:
    """Add overlay directory ``path`` to the active overlays file."""
    purge_paths: List[str] = []
    keep_paths: List[str]
    with _ACTIVE_OVERLAYS_LOCK:
        paths = _read_active_overlays_unlocked()
        if path not in paths:
            paths.append(path)
        if _MAX_OVERLAY_COUNT > 0 and len(paths) > _MAX_OVERLAY_COUNT:
            excess = len(paths) - _MAX_OVERLAY_COUNT
            purge_paths = paths[:excess]
            keep_paths = paths[excess:]
            _write_active_overlays_unlocked(purge_paths)
        else:
            keep_paths = paths
            _write_active_overlays_unlocked(keep_paths)

    if purge_paths:
        removed = _purge_stale_vms()
        try:
            logger.warning(
                "active overlay limit reached (%s), removed %d overlay(s)",
                _MAX_OVERLAY_COUNT,
                removed,
            )
        except Exception as exc:
            logger.debug("failed to log overlay limit warning: %s", exc)
        global _ACTIVE_OVERLAY_LIMIT_REACHED
        _ACTIVE_OVERLAY_LIMIT_REACHED += 1
        with _ACTIVE_OVERLAYS_LOCK:
            _write_active_overlays_unlocked(keep_paths)


def _remove_active_overlay(path: str) -> None:
    """Remove overlay directory ``path`` from the active overlays file."""
    with _ACTIVE_OVERLAYS_LOCK:
        paths = _read_active_overlays_unlocked()
        if path in paths:
            paths.remove(path)
            _write_active_overlays_unlocked(paths)


def _read_failed_overlays() -> List[str]:
    """Return list of overlay directories that failed to delete."""
    data = _read_json_default(
        _FAILED_OVERLAYS_FILE,
        [],
        log_context="failed reading failed overlays",
    )
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _write_failed_overlays(paths: List[str]) -> None:
    """Persist ``paths`` to the failed overlays file."""
    try:
        _atomic_write_json(_FAILED_OVERLAYS_FILE, paths)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning(
            "failed writing failed overlays %s: %s", _FAILED_OVERLAYS_FILE, exc
        )


def _record_failed_overlay(path: str) -> None:
    """Record overlay directory ``path`` as failed to delete."""
    global _OVERLAY_CLEANUP_FAILURES
    paths = _read_failed_overlays()
    if path not in paths:
        paths.append(path)
        _write_failed_overlays(paths)
    _record_failed_cleanup(path)
    _OVERLAY_CLEANUP_FAILURES += 1
    _increment_cleanup_stat("overlay_cleanup_failures")
    try:
        from . import metrics_exporter as _me
    except Exception:
        try:  # pragma: no cover - package may not be available
            import metrics_exporter as _me  # type: ignore
        except Exception:
            _me = None  # type: ignore
    gauge = getattr(_me, "overlay_cleanup_failures", None) if _me else None
    if gauge is not None:
        try:
            gauge.inc()
        except Exception:
            logger.exception("failed to increment overlay_cleanup_failures")


def _read_failed_cleanup() -> Dict[str, Dict[str, Any]]:
    """Return mapping of failed cleanup items and their metadata."""
    data = _read_json_default(
        FAILED_CLEANUP_FILE,
        {},
        log_context="failed reading failed cleanup",
    )
    if isinstance(data, dict):
        try:
            entries: Dict[str, Dict[str, Any]] = {}
            for key, value in data.items():
                ts: float
                reason = ""
                if isinstance(value, dict):
                    ts = float(value.get("ts", 0.0))
                    reason = str(value.get("reason", ""))
                else:
                    ts = float(value)
                entries[str(key)] = {"ts": ts, "reason": reason}
            return entries
        except Exception as exc:  # pragma: no cover - conversion errors
            logger.warning("failed reading failed cleanup %s: %s", FAILED_CLEANUP_FILE, exc)
    return {}


def _write_failed_cleanup(entries: Dict[str, Dict[str, Any]]) -> None:
    """Persist ``entries`` to the failed cleanup file."""
    try:
        _atomic_write_json(FAILED_CLEANUP_FILE, entries)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed writing failed cleanup %s: %s", FAILED_CLEANUP_FILE, exc)


def _read_cleanup_stats() -> Dict[str, int]:
    """Return persistent cleanup statistics."""
    data = _read_json_default(
        _CLEANUP_STATS_FILE,
        {},
        log_context="failed reading cleanup stats",
    )
    if isinstance(data, dict):
        try:
            return {str(k): int(v) for k, v in data.items()}
        except Exception as exc:  # pragma: no cover - conversion errors
            logger.warning("failed reading cleanup stats %s: %s", _CLEANUP_STATS_FILE, exc)
    return {}


def _write_cleanup_stats(stats: Dict[str, int]) -> None:
    """Persist cleanup ``stats`` to file."""
    try:
        _atomic_write_json(_CLEANUP_STATS_FILE, stats)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed writing cleanup stats %s: %s", _CLEANUP_STATS_FILE, exc)


def _increment_cleanup_stat(name: str, amount: int = 1) -> None:
    """Increment persistent cleanup stat ``name`` by ``amount``."""
    data = _read_cleanup_stats()
    data[name] = int(data.get(name, 0)) + amount
    _write_cleanup_stats(data)


def _read_last_autopurge() -> float:
    """Return timestamp of last automatic purge."""
    data = _read_json_default(
        _LAST_AUTOPURGE_FILE,
        0.0,
        log_context="failed reading last autopurge",
    )
    try:
        return float(data)
    except (TypeError, ValueError) as exc:
        logger.warning("failed reading last autopurge %s: %s", _LAST_AUTOPURGE_FILE, exc)
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.warning("failed reading last autopurge %s: %s", _LAST_AUTOPURGE_FILE, exc)
    return 0.0


def _write_last_autopurge(ts: float) -> None:
    """Persist ``ts`` as the last automatic purge time."""
    try:
        _LAST_AUTOPURGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_AUTOPURGE_FILE.write_text(str(ts))
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed writing last autopurge %s: %s", _LAST_AUTOPURGE_FILE, exc)


def _record_failed_cleanup(item: str, *, reason: str | None = None) -> None:
    """Record ``item`` as failed to clean up with current timestamp."""
    data = _read_failed_cleanup()
    entry = data.get(item, {"ts": 0.0, "reason": ""})
    entry["ts"] = time.time()
    if reason:
        entry["reason"] = reason
    data[item] = entry
    _write_failed_cleanup(data)


def _remove_failed_cleanup(item: str) -> None:
    """Remove ``item`` from the failed cleanup file."""
    data = _read_failed_cleanup()
    if item in data:
        del data[item]
        _write_failed_cleanup(data)


def _write_cleanup_log(entry: Dict[str, Any]) -> None:
    """Append a cleanup log ``entry`` to :data:`_CLEANUP_LOG_PATH`."""
    try:
        _CLEANUP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CLEANUP_LOG_LOCK, open(_CLEANUP_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:  # pragma: no cover - log failures shouldn't crash
        logger.warning("failed writing cleanup log: %s", exc)


def _log_cleanup_event(resource_id: str, reason: str, success: bool) -> None:
    """Record a container or VM cleanup attempt."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "resource_id": resource_id,
        "reason": reason,
        "success": success,
    }
    _write_cleanup_log(entry)


def _remove_failed_overlay(path: str) -> None:
    """Remove overlay directory ``path`` from the failed overlays file."""
    paths = _read_failed_overlays()
    if path in paths:
        paths.remove(path)
        _write_failed_overlays(paths)


def _rmtree_windows(path: str, attempts: int = 5, base: float = 0.2) -> bool:
    """Remove ``path`` in a helper process with exponential backoff."""
    script = textwrap.dedent(
        """
        import shutil, sys, time
        p = sys.argv[1]
        base = float(sys.argv[2])
        attempts = int(sys.argv[3])
        for i in range(attempts):
            try:
                shutil.rmtree(p)
                sys.exit(0)
            except Exception:
                time.sleep(base * (2 ** i))
        sys.exit(1)
        """
    )
    try:
        proc = _run_subprocess_with_progress(
            [sys.executable, "-c", script, path, str(base), str(attempts)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode == 0:
            return True
    except subprocess.TimeoutExpired as exc:
        duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
        logger.warning(
            "python rmtree helper timed out for %s (%.1fs)",
            path,
            duration,
        )
        _record_failed_cleanup(path, reason="python rmtree helper timeout")
        return False
    except Exception as exc:
        logger.debug("rmtree helper failed: %s", exc)

    try:
        proc = _run_subprocess_with_progress(
            ["cmd", "/c", "rmdir", "/s", "/q", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - rare
        duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
        logger.warning(
            "cmd rmdir timed out for %s (%.1fs)",
            path,
            duration,
        )
        _record_failed_cleanup(path, reason="cmd rmdir timeout")
        return False
    except Exception as exc:  # pragma: no cover - fallback errors rare
        logger.debug("rmdir fallback failed: %s", exc)
        return False


def _purge_stale_vms(*, record_runtime: bool = False) -> int:
    """Terminate leftover QEMU processes and remove overlay files."""
    global _STALE_VMS_REMOVED, _RUNTIME_VMS_REMOVED, _OVERLAY_CLEANUP_FAILURES
    removed_vms = 0
    recorded = _read_active_overlays()
    failed_overlays = _read_failed_overlays()
    to_cleanup = set(recorded + failed_overlays)
    if to_cleanup:
        for d in list(to_cleanup):
            try:
                logger.info("removing recorded overlay dir %s", d)
            except Exception as exc:
                logger.debug("failed to log overlay dir removal %s: %s", d, exc)
            success = True
            try:
                shutil.rmtree(d)
                removed_vms += 1
                _remove_failed_overlay(d)
                _remove_failed_cleanup(d)
            except Exception:
                if os.name == "nt" and _rmtree_windows(d):
                    removed_vms += 1
                    _remove_failed_overlay(d)
                    _remove_failed_cleanup(d)
                else:
                    logger.exception("temporary directory removal failed for %s", d)
                    _record_failed_overlay(d)
                    success = False
            _log_cleanup_event(str(d), "vm_overlay", success)
        _write_active_overlays([])
    if psutil is not None:
        try:
            tmp_dirs: set[str] = set()
            for p in psutil.process_iter(["name", "cmdline"]):
                name = p.info.get("name") or ""
                if name.startswith("qemu-system"):
                    try:
                        logger.info("terminating stale qemu process %s", p.pid)
                    except Exception as exc:
                        logger.debug("failed to log qemu termination %s: %s", p.pid, exc)
                    try:
                        for arg in p.info.get("cmdline") or []:
                            if "overlay.qcow2" in arg:
                                if arg.startswith("file="):
                                    arg = arg.split("=", 1)[1]
                                tmp_dirs.add(str(Path(arg).parent))
                        p.kill()
                        p.wait(timeout=5)
                        removed_vms += 1
                        _log_cleanup_event(str(p.pid), "vm_process", True)
                    except Exception:
                        logger.exception("failed to terminate qemu %s", p.pid)
                        _log_cleanup_event(str(p.pid), "vm_process", False)
            for d in tmp_dirs:
                success = True
                try:
                    shutil.rmtree(d)
                    _remove_failed_overlay(d)
                    _remove_failed_cleanup(d)
                except Exception:
                    if os.name == "nt" and _rmtree_windows(d):
                        _remove_failed_overlay(d)
                        _remove_failed_cleanup(d)
                    else:
                        logger.exception("temporary directory removal failed for %s", d)
                        _record_failed_overlay(d)
                        success = False
                _log_cleanup_event(str(d), "vm_overlay", success)
        except Exception as exc:
            logger.debug("qemu process cleanup failed: %s", exc)
    else:  # pragma: no cover - fallback path
        tmp_dirs: set[str] = set()
        try:
            proc = _run_subprocess_with_progress(
                ["pgrep", "-fa", "qemu-system"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
            logger.warning(
                "pgrep timed out during stale vm cleanup (%.1fs)",
                duration,
            )
            _record_failed_cleanup(
                "qemu-process-scan",
                reason="pgrep timeout",
            )
            return removed_vms
        except Exception as exc:
            logger.debug("qemu process cleanup failed: %s", exc)
        else:
            for line in proc.stdout.splitlines():
                parts = line.strip().split(maxsplit=1)
                if not parts:
                    continue
                pid = parts[0]
                cmdline = parts[1] if len(parts) > 1 else ""
                try:
                    logger.info("terminating stale qemu process %s", pid)
                except Exception as exc:
                    logger.debug("failed to log qemu termination %s: %s", pid, exc)
                for arg in cmdline.split():
                    if "overlay.qcow2" in arg:
                        if arg.startswith("file="):
                            arg = arg.split("=", 1)[1]
                        tmp_dirs.add(str(Path(arg).parent))
                try:
                    res = _run_subprocess_with_progress(
                        ["kill", "-9", pid],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                    )
                except subprocess.TimeoutExpired as exc:
                    duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
                    logger.warning(
                        "kill -9 timed out for qemu %s (%.1fs)",
                        pid,
                        duration,
                    )
                    _record_failed_cleanup(
                        f"qemu:{pid}",
                        reason="kill timeout",
                    )
                    _log_cleanup_event(str(pid), "vm_process", False)
                    continue
                removed_vms += 1
                _log_cleanup_event(str(pid), "vm_process", res.returncode == 0)
            for d in tmp_dirs:
                success = True
                try:
                    shutil.rmtree(d)
                    _remove_failed_overlay(d)
                    _remove_failed_cleanup(d)
                except Exception:
                    if os.name == "nt" and _rmtree_windows(d):
                        _remove_failed_overlay(d)
                        _remove_failed_cleanup(d)
                    else:
                        logger.exception("temporary directory removal failed for %s", d)
                        _record_failed_overlay(d)
                        success = False
                _log_cleanup_event(str(d), "vm_overlay", success)

    if os.name == "nt":  # pragma: no cover - windows process cleanup
        try:
            proc = _run_subprocess_with_progress(
                ["taskkill", "/F", "/T", "/IM", "qemu-system*"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            duration = float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT)
            logger.warning(
                "taskkill timed out during stale vm cleanup (%.1fs)",
                duration,
            )
            _record_failed_cleanup(
                "windows:qemu-taskkill",
                reason="taskkill timeout",
            )
            return removed_vms
        except Exception as exc:
            logger.debug("taskkill failed: %s", exc)
        else:
            for line in proc.stdout.splitlines():
                if "SUCCESS:" in line:
                    removed_vms += 1
                    _log_cleanup_event("windows_qemu", "vm_process", True)

    tmp_root = Path(tempfile.gettempdir())
    threshold = time.time() - _OVERLAY_MAX_AGE
    try:
        for overlay in tmp_root.rglob("overlay.qcow2"):
            try:
                mtime = overlay.parent.stat().st_mtime
            except Exception:
                try:
                    mtime = overlay.stat().st_mtime
                except Exception:
                    mtime = 0
            if _OVERLAY_MAX_AGE and mtime > threshold:
                continue
            try:
                logger.info("removing leftover overlay %s", overlay)
            except Exception as exc:
                logger.debug("failed to log overlay removal %s: %s", overlay, exc)
            success = True
            try:
                overlay.unlink()
                removed_vms += 1
                _remove_failed_cleanup(str(overlay.parent))
            except Exception:
                logger.exception("failed to remove overlay %s", overlay)
                _record_failed_overlay(str(overlay.parent))
                success = False
            try:
                shutil.rmtree(overlay.parent)
                _remove_failed_overlay(str(overlay.parent))
                _remove_failed_cleanup(str(overlay.parent))
            except Exception:
                if os.name == "nt" and _rmtree_windows(str(overlay.parent)):
                    _remove_failed_overlay(str(overlay.parent))
                    _remove_failed_cleanup(str(overlay.parent))
                else:
                    logger.exception(
                        "temporary directory removal failed for %s", overlay.parent
                    )
                    _record_failed_overlay(str(overlay.parent))
                    success = False
            _log_cleanup_event(str(overlay.parent), "vm_overlay", success)
    except Exception as exc:
        logger.debug("overlay cleanup failed: %s", exc)
        _OVERLAY_CLEANUP_FAILURES += 1
        _increment_cleanup_stat("overlay_cleanup_failures")
        try:
            from . import metrics_exporter as _me
        except Exception:
            try:  # pragma: no cover - package may not be available
                import metrics_exporter as _me  # type: ignore
            except Exception:
                _me = None  # type: ignore
        gauge = getattr(_me, "overlay_cleanup_failures", None) if _me else None
        if gauge is not None:
            try:
                gauge.inc()
            except Exception:
                logger.exception("failed to increment overlay_cleanup_failures")

    _STALE_VMS_REMOVED += removed_vms
    if record_runtime:
        _RUNTIME_VMS_REMOVED += removed_vms
    return removed_vms


def _prune_volumes(*, progress: Callable[[], None] | None = None) -> int:
    """Remove Docker volumes created by the sandbox."""
    if not _PRUNE_VOLUMES:
        return 0
    removed = 0
    labeled: set[str] = set()
    try:
        proc = _run_subprocess_with_progress(
            [
                "docker",
                "volume",
                "ls",
                "-q",
                "--filter",
                f"label={_POOL_LABEL}=1",
            ],
            progress=progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode == 0:
            for vol in proc.stdout.splitlines():
                vol = vol.strip()
                if not vol:
                    continue
                try:
                    logger.info("removing stale sandbox volume %s", vol)
                except Exception as exc:
                    logger.debug("failed to log volume removal %s: %s", vol, exc)
                try:
                    _run_subprocess_with_progress(
                        ["docker", "volume", "rm", "-f", vol],
                        progress=progress,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                    )
                except subprocess.TimeoutExpired as exc:
                    logger.warning(
                        "volume prune timed out for %s (%.1fs)",
                        vol,
                        float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                    )
                    _record_failed_cleanup(
                        f"volume:{vol}",
                        reason="docker volume rm timeout",
                    )
                    _notify_progress(progress)
                    return removed
                removed += 1
                labeled.add(vol)
                _CLEANUP_METRICS["volume"] += 1
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "volume listing timed out (%.1fs)",
            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
        )
        _notify_progress(progress)
        return removed
    except Exception as exc:
        logger.debug("leftover volume cleanup failed: %s", exc)

    threshold = time.time() - _CONTAINER_MAX_LIFETIME
    try:
        proc = _run_subprocess_with_progress(
            ["docker", "volume", "ls", "-q"],
            progress=progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode == 0:
            for vol in proc.stdout.splitlines():
                vol = vol.strip()
                if not vol or vol in labeled:
                    continue
                try:
                    info = _run_subprocess_with_progress(
                        ["docker", "volume", "inspect", vol],
                        progress=progress,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                    )
                except subprocess.TimeoutExpired as exc:
                    logger.warning(
                        "volume inspect timed out for %s (%.1fs)",
                        vol,
                        float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                    )
                    _record_failed_cleanup(
                        f"volume:{vol}",
                        reason="docker volume inspect timeout",
                    )
                    _notify_progress(progress)
                    return removed
                if info.returncode != 0:
                    continue
                try:
                    data = json.loads(info.stdout)[0]
                    created = data.get("CreatedAt") or data.get("Created")
                    labels = data.get("Labels") or {}
                    if labels.get(_POOL_LABEL) == "1":
                        continue
                    created_ts = datetime.fromisoformat(
                        str(created).replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    continue
                if created_ts <= threshold:
                    try:
                        logger.info("removing stale sandbox volume %s", vol)
                    except Exception as exc:
                        logger.debug("failed to log volume removal %s: %s", vol, exc)
                    try:
                        _run_subprocess_with_progress(
                            ["docker", "volume", "rm", "-f", vol],
                            progress=progress,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                        )
                    except subprocess.TimeoutExpired as exc:
                        logger.warning(
                            "volume prune timed out for %s (%.1fs)",
                            vol,
                            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                        )
                        _record_failed_cleanup(
                            f"volume:{vol}",
                            reason="docker volume rm timeout",
                        )
                        _notify_progress(progress)
                        return removed
                    removed += 1
                    _CLEANUP_METRICS["volume"] += 1
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "volume listing timed out (%.1fs)",
            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
        )
        _notify_progress(progress)
        return removed
    except Exception as exc:
        logger.debug("unlabeled volume cleanup failed: %s", exc)

    return removed


def _prune_networks(*, progress: Callable[[], None] | None = None) -> int:
    """Remove Docker networks created by the sandbox."""
    if not _PRUNE_NETWORKS:
        return 0
    removed = 0
    labeled: set[str] = set()
    try:
        proc = _run_subprocess_with_progress(
            [
                "docker",
                "network",
                "ls",
                "-q",
                "--filter",
                f"label={_POOL_LABEL}=1",
            ],
            progress=progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode == 0:
            for net in proc.stdout.splitlines():
                net = net.strip()
                if not net:
                    continue
                try:
                    logger.info("removing stale sandbox network %s", net)
                except Exception as exc:
                    logger.debug("failed to log network removal %s: %s", net, exc)
                try:
                    _run_subprocess_with_progress(
                        ["docker", "network", "rm", "-f", net],
                        progress=progress,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                    )
                except subprocess.TimeoutExpired as exc:
                    logger.warning(
                        "network prune timed out for %s (%.1fs)",
                        net,
                        float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                    )
                    _record_failed_cleanup(
                        f"network:{net}",
                        reason="docker network rm timeout",
                    )
                    _notify_progress(progress)
                    return removed
                removed += 1
                labeled.add(net)
                _CLEANUP_METRICS["network"] += 1
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "network listing timed out (%.1fs)",
            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
        )
        _notify_progress(progress)
        return removed
    except Exception as exc:
        logger.debug("leftover network cleanup failed: %s", exc)

    threshold = time.time() - _CONTAINER_MAX_LIFETIME
    try:
        proc = _run_subprocess_with_progress(
            ["docker", "network", "ls", "-q"],
            progress=progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode == 0:
            for net in proc.stdout.splitlines():
                net = net.strip()
                if not net or net in labeled:
                    continue
                try:
                    info = _run_subprocess_with_progress(
                        ["docker", "network", "inspect", net],
                        progress=progress,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                    )
                except subprocess.TimeoutExpired as exc:
                    logger.warning(
                        "network inspect timed out for %s (%.1fs)",
                        net,
                        float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                    )
                    _record_failed_cleanup(
                        f"network:{net}",
                        reason="docker network inspect timeout",
                    )
                    _notify_progress(progress)
                    return removed
                if info.returncode != 0:
                    continue
                try:
                    data = json.loads(info.stdout)[0]
                    created = data.get("Created") or data.get("CreatedAt")
                    labels = data.get("Labels") or {}
                    name = data.get("Name")
                    if name in {"bridge", "host", "none"}:
                        continue
                    if labels.get(_POOL_LABEL) == "1":
                        continue
                    created_ts = datetime.fromisoformat(
                        str(created).replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    continue
                if created_ts <= threshold:
                    try:
                        logger.info("removing stale sandbox network %s", net)
                    except Exception as exc:
                        logger.debug("failed to log network removal %s: %s", net, exc)
                    try:
                        _run_subprocess_with_progress(
                            ["docker", "network", "rm", "-f", net],
                            progress=progress,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                        )
                    except subprocess.TimeoutExpired as exc:
                        logger.warning(
                            "network prune timed out for %s (%.1fs)",
                            net,
                            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                        )
                        _record_failed_cleanup(
                            f"network:{net}",
                            reason="docker network rm timeout",
                        )
                        _notify_progress(progress)
                        return removed
                    removed += 1
                    _CLEANUP_METRICS["network"] += 1
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "network listing timed out (%.1fs)",
            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
        )
        _notify_progress(progress)
        return removed
    except Exception as exc:
        logger.debug("unlabeled network cleanup failed: %s", exc)

    return removed


def purge_leftovers() -> None:
    """Remove stale sandbox containers and leftover QEMU overlay files.

    Container commands are matched using :func:`resolve_path` to locate
    ``sandbox_runner.py`` dynamically, ensuring cleanup works even if the
    repository layout changes.
    """
    global _STALE_CONTAINERS_REMOVED
    lock_file = getattr(_PURGE_FILE_LOCK, "lock_file", "<unknown>")
    try:
        with _PURGE_FILE_LOCK:
            try:
                reconcile_active_containers()
            except FileNotFoundError:
                logger.debug("docker not available; skipping purge")
                return
            removed_containers = 0
            try:
                ids = _read_active_containers()
                remaining_ids = []
                for cid in ids:
                    try:
                        logger.info("removing recorded sandbox container %s", cid)
                    except Exception as exc:
                        logger.debug("failed to log container removal %s: %s", cid, exc)
                    try:
                        _run_subprocess_with_progress(
                            ["docker", "rm", "-f", cid],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                        )
                    except subprocess.TimeoutExpired as exc:
                        logger.warning(
                            "purge leftovers timed out removing %s (%.1fs)",
                            cid,
                            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                        )
                        _record_failed_cleanup(
                            f"container:{cid}",
                            reason="docker rm timeout during purge",
                        )
                        remaining_ids.append(cid)
                        continue
                    exists = False
                    try:
                        proc = _run_subprocess_with_progress(
                            ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False,
                            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                        )
                        if proc.returncode == 0 and proc.stdout.strip():
                            exists = True
                    except subprocess.TimeoutExpired as exc:
                        logger.warning(
                            "purge leftovers timed out verifying %s (%.1fs)",
                            cid,
                            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                        )
                        _record_failed_cleanup(
                            f"container:{cid}",
                            reason="docker ps timeout during purge",
                        )
                        remaining_ids.append(cid)
                        continue
                    except Exception as exc:
                        logger.debug(
                            "container existence check failed for %s: %s", cid, exc
                        )
                        exists = True

                    if exists:
                        _record_failed_cleanup(cid)
                        remaining_ids.append(cid)
                    else:
                        _remove_failed_cleanup(cid)
                        removed_containers += 1
                if ids:
                    _write_active_containers(remaining_ids)
            except Exception as exc:
                logger.debug("active container cleanup failed: %s", exc)

            # remove any recorded overlay directories first
            try:
                overlays = _read_active_overlays()
                for d in overlays:
                    try:
                        logger.info("removing recorded overlay dir %s", d)
                    except Exception as exc:
                        logger.debug("failed to log overlay dir removal %s: %s", d, exc)
                    try:
                        shutil.rmtree(d)
                        _remove_failed_overlay(d)
                        _remove_failed_cleanup(d)
                    except Exception:
                        if os.name == "nt" and _rmtree_windows(d):
                            _remove_failed_overlay(d)
                            _remove_failed_cleanup(d)
                        else:
                            logger.exception("temporary directory removal failed for %s", d)
                            _record_failed_overlay(d)
                if overlays:
                    _write_active_overlays([])
            except Exception as exc:
                logger.debug("overlay cleanup failed: %s", exc)

            try:
                proc = _run_subprocess_with_progress(
                    ["docker", "ps", "-aq", "--filter", f"label={_POOL_LABEL}=1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
                if proc.returncode == 0:
                    for cid in proc.stdout.splitlines():
                        cid = cid.strip()
                        if not cid:
                            continue
                        try:
                            logger.info("removing stale sandbox container %s", cid)
                        except Exception as exc:
                            logger.debug("failed to log stale container %s: %s", cid, exc)
                        try:
                            proc_rm = _run_subprocess_with_progress(
                                ["docker", "rm", "-f", cid],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                            )
                        except subprocess.TimeoutExpired as exc:
                            logger.warning(
                                "purge leftovers timed out removing %s (%.1fs)",
                                cid,
                                float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                            )
                            _record_failed_cleanup(
                                f"container:{cid}",
                                reason="docker rm timeout during purge",
                            )
                            continue
                        _log_cleanup_event(cid, "shutdown", proc_rm.returncode == 0)
                        _remove_failed_cleanup(cid)
                        removed_containers += 1
            except subprocess.TimeoutExpired as exc:
                logger.warning(
                    "purge leftovers timed out listing containers (%.1fs)",
                    float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                )
            except Exception as exc:
                logger.debug("leftover container cleanup failed: %s", exc)

            try:
                threshold = time.time() - _CONTAINER_MAX_LIFETIME
                proc = _run_subprocess_with_progress(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--no-trunc",
                        "--format",
                        "{{.ID}}\t{{.CreatedAt}}\t{{.Command}}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
                if proc.returncode == 0:
                    for line in proc.stdout.splitlines():
                        parts = line.split("\t", 2)
                        if len(parts) < 3:
                            continue
                        cid, created_at, cmd = parts
                        ts_str = " ".join(created_at.split()[:3])
                        try:
                            created_ts = datetime.strptime(
                                ts_str, "%Y-%m-%d %H:%M:%S %z"
                            ).timestamp()
                        except Exception:
                            continue
                        if (
                            created_ts <= threshold
                            and resolve_path("sandbox_runner.py").name in cmd
                        ):
                            try:
                                logger.info("removing stale sandbox container %s", cid)
                            except Exception as exc:
                                logger.debug(
                                    "failed to log stale container %s: %s", cid, exc
                                )
                            try:
                                proc_rm = _run_subprocess_with_progress(
                                    ["docker", "rm", "-f", cid],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    check=False,
                                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                                )
                            except subprocess.TimeoutExpired as exc:
                                logger.warning(
                                    "purge leftovers timed out removing %s (%.1fs)",
                                    cid,
                                    float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                                )
                                _record_failed_cleanup(
                                    f"container:{cid}",
                                    reason="docker rm timeout during purge",
                                )
                                continue
                            _log_cleanup_event(cid, "shutdown", proc_rm.returncode == 0)
                            _remove_failed_cleanup(cid)
                            removed_containers += 1
            except subprocess.TimeoutExpired as exc:
                logger.warning(
                    "purge leftovers timed out inspecting containers (%.1fs)",
                    float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                )
            except Exception as exc:
                logger.debug("unlabeled container cleanup failed: %s", exc)

            removed_vms = _purge_stale_vms()

            _prune_volumes()
            _prune_networks()

            _STALE_CONTAINERS_REMOVED += removed_containers
    except Timeout:
        logger.warning(
            "skipping purge leftovers; lock %s is held by another process",
            lock_file,
        )
        return

    global _LAST_AUTOPURGE_TS
    _LAST_AUTOPURGE_TS = time.time()
    _write_last_autopurge(_LAST_AUTOPURGE_TS)

    report_failed_cleanup(alert=True)


def autopurge_if_needed() -> None:
    """Run :func:`purge_leftovers` when the configured threshold has elapsed."""
    if _SANDBOX_DISABLE_CLEANUP:
        _log_cleanup_disabled("automatic purge")
        return
    if _SANDBOX_AUTOPURGE_THRESHOLD <= 0:
        return
    try:
        if time.time() - _LAST_AUTOPURGE_TS >= _SANDBOX_AUTOPURGE_THRESHOLD:
            purge_leftovers()
            retry_failed_cleanup()
    except Exception as exc:  # pragma: no cover - unexpected runtime issues
        logger.exception("automatic purge failed: %s", exc)


def _docker_available() -> bool:
    """Return ``True`` when Docker client is usable."""
    try:
        if docker is None:
            return False
        client = _DOCKER_CLIENT or docker.from_env()
        client.ping()
        return True
    except DockerException:
        return False


def ensure_docker_client() -> None:
    """Recreate ``_DOCKER_CLIENT`` if disconnected."""
    global _DOCKER_CLIENT
    if docker is None:
        return

    client = _DOCKER_CLIENT
    reconnect = client is None
    ping_error: Exception | None = None
    if client is not None:
        healthy, ping_error = _ping_docker(client, timeout=_DOCKER_PING_TIMEOUT)
        if healthy:
            return
        reconnect = True
        try:
            if ping_error is None:
                logger.warning("docker client ping failed")
            else:
                logger.warning("docker client ping failed: %s", ping_error)
        except Exception as log_exc:
            description = "docker client ping failed"
            if ping_error is not None:
                description += f": {ping_error}"
            _fallback_logger().warning(
                "%s (logging failed: %s)",
                description,
                log_exc,
                exc_info=True,
            )
        _close_docker_client(client)
        _DOCKER_CLIENT = None

    if not reconnect:
        return

    try:
        logger.info("reconnecting docker client")
    except Exception as log_exc:
        _fallback_logger().warning(
            "reconnecting docker client (logging failed: %s)",
            log_exc,
            exc_info=True,
        )

    new_client, error = _create_docker_client()
    if new_client is None:
        combined_error = error or ping_error
        if combined_error is not None:
            logger.error("docker client reconnection failed: %s", combined_error)
        else:
            logger.error("docker client reconnection failed")
        _suspend_cleanup_workers("docker client unavailable; background cleanup paused")
        return

    _DOCKER_CLIENT = new_client
    logger.info("docker client reconnected")


def _ensure_pool_size_async(image: str) -> None:
    """Warm up pool for ``image`` asynchronously."""
    if _SANDBOX_DISABLE_CLEANUP:
        _log_cleanup_disabled("pool warmup request")
        return
    if _DOCKER_CLIENT is None:
        return
    with _POOL_LOCK:
        pool = _CONTAINER_POOLS.setdefault(image, [])
        t = _WARMUP_TASKS.get(image)
    _cleanup_idle_containers()
    if len(pool) >= _CONTAINER_POOL_SIZE:
        return
    if _MAX_CONTAINER_COUNT > 0 and len(_read_active_containers()) >= _MAX_CONTAINER_COUNT:
        logger.warning(
            "active container limit reached (%s)", _MAX_CONTAINER_COUNT
        )
        global _ACTIVE_CONTAINER_LIMIT_REACHED
        _ACTIVE_CONTAINER_LIMIT_REACHED += 1
        return
    if _MAX_OVERLAY_COUNT > 0 and len(_read_active_overlays()) >= _MAX_OVERLAY_COUNT:
        removed = _purge_stale_vms()
        try:
            logger.warning(
                "active overlay limit reached (%s), removed %d overlay(s)",
                _MAX_OVERLAY_COUNT,
                removed,
            )
        except Exception as exc:
            logger.debug("failed to log overlay limit warning: %s", exc)
        global _ACTIVE_OVERLAY_LIMIT_REACHED
        _ACTIVE_OVERLAY_LIMIT_REACHED += 1
    if t and not t.done():
        return

    async def _worker() -> None:
        try:
            needed = _CONTAINER_POOL_SIZE - len(pool)
            for _ in range(needed):
                try:
                    c, _ = await _create_pool_container(image)
                except RuntimeError as exc:
                    if "active container limit" in str(exc).lower():
                        break
                    raise
                with _POOL_LOCK:
                    pool.append(c)
                    _CONTAINER_LAST_USED[c.id] = time.time()
        except Exception as exc:
            logger.warning("container warm up failed: %s", exc)
        finally:
            with _POOL_LOCK:
                _WARMUP_TASKS.pop(image, None)

    task = _schedule_coroutine(_worker())
    with _POOL_LOCK:
        _WARMUP_TASKS[image] = task


async def _create_pool_container(image: str) -> tuple[Any, str]:
    """Create a long-lived container running ``sleep infinity`` with retries."""
    ensure_docker_client()
    assert _DOCKER_CLIENT is not None
    async with pool_lock():
        if _MAX_CONTAINER_COUNT > 0 and len(_read_active_containers()) >= _MAX_CONTAINER_COUNT:
            logger.warning(
                "active container limit reached (%s)", _MAX_CONTAINER_COUNT
            )
            global _ACTIVE_CONTAINER_LIMIT_REACHED
            _ACTIVE_CONTAINER_LIMIT_REACHED += 1
            raise RuntimeError("active container limit reached")
        attempt = 0
        delay = _CREATE_BACKOFF_BASE
        last_exc: Exception | None = None
        while attempt < _CREATE_RETRY_LIMIT:
            fails = _CONSECUTIVE_CREATE_FAILURES.get(image, 0)
            if attempt or fails >= 3:
                wait = min(60.0, delay * (2 ** max(attempt, fails)))
                logger.warning(
                    "container creation backoff %.2fs before attempt %s/%s for %s",
                    wait,
                    attempt + 1,
                    _CREATE_RETRY_LIMIT,
                    image,
                )
                await asyncio.sleep(wait)
        td = tempfile.mkdtemp(prefix="pool_")
        start_time = time.monotonic()
        try:
            run_kwargs = {
                "detach": True,
                "network_disabled": True,
                "volumes": {td: {"bind": "/code", "mode": "rw"}},
                "labels": {_POOL_LABEL: "1"},
            }
            if _CONTAINER_DISK_LIMIT > 0:
                run_kwargs["storage_opt"] = {"size": str(_CONTAINER_DISK_LIMIT)}
            if _CONTAINER_USER:
                run_kwargs["user"] = _CONTAINER_USER

            container = await asyncio.to_thread(
                _DOCKER_CLIENT.containers.run,
                image,
                ["sleep", "infinity"],
                **run_kwargs,
            )
            _register_container_finalizer(container)
            _record_active_container(container.id)
            _CONSECUTIVE_CREATE_FAILURES[image] = 0
            _log_pool_metrics(image)
            try:
                from . import metrics_exporter as _me
            except Exception:
                try:  # pragma: no cover - package may not be available
                    import metrics_exporter as _me  # type: ignore
                except Exception:
                    _me = None  # type: ignore
            gauge = (
                getattr(_me, "container_creation_success_total", None)
                if _me
                else None
            )
            if gauge is not None:
                try:
                    gauge.labels(image=image).inc()
                except (AttributeError, ValueError):
                    logger.exception(
                        "failed to increment container_creation_success_total"
                    )
            gauge_dur = (
                getattr(_me, "container_creation_seconds", None) if _me else None
            )
            if gauge_dur is not None:
                try:
                    gauge_dur.labels(image=image).set(time.monotonic() - start_time)
                except (AttributeError, ValueError):
                    logger.exception(
                        "failed to update container_creation_seconds"
                    )
            with _POOL_LOCK:
                _CONTAINER_DIRS[container.id] = td
                _CONTAINER_LAST_USED[container.id] = time.time()
                _CONTAINER_CREATED[container.id] = time.time()
            try:
                img_tag = getattr(getattr(container, "image", None), "tags", None)
                img_tag = img_tag[0] if img_tag else image
                logger.info(
                    "created container %s from image %s", container.id, img_tag
                )
            except Exception as exc:
                logger.debug("failed to log container creation %s: %s", container.id, exc)
            return container, td
        except Exception as exc:
            last_exc = exc
            shutil.rmtree(td, ignore_errors=True)
            _CREATE_FAILURES[image] += 1
            fails = _CONSECUTIVE_CREATE_FAILURES.get(image, 0) + 1
            _CONSECUTIVE_CREATE_FAILURES[image] = fails
            logger.warning(
                "docker run failed on attempt %s/%s: %s; cmd: docker run %s sleep infinity",
                attempt + 1,
                _CREATE_RETRY_LIMIT,
                exc,
                image,
            )
            if fails >= _FAILURE_WARNING_THRESHOLD:
                logger.warning(
                    "container creation failing %s times consecutively for %s",
                    fails,
                    image,
                )
            _log_pool_metrics(image)
            try:
                from . import metrics_exporter as _me
            except Exception:
                try:  # pragma: no cover - package may not be available
                    import metrics_exporter as _me  # type: ignore
                except Exception:
                    _me = None  # type: ignore
            gauge = (
                getattr(_me, "container_creation_failures_total", None)
                if _me
                else None
            )
            if gauge is not None:
                try:
                    gauge.labels(image=image).inc()
                except (AttributeError, ValueError):
                    logger.exception(
                        "failed to increment container_creation_failures_total"
                    )
            gauge_dur = (
                getattr(_me, "container_creation_seconds", None) if _me else None
            )
            if gauge_dur is not None:
                try:
                    gauge_dur.labels(image=image).set(time.monotonic() - start_time)
                except (AttributeError, ValueError):
                    logger.exception(
                        "failed to update container_creation_seconds"
                    )
            try:
                dispatch_alert(
                    "container_creation_failures",
                    2,
                    "repeated container creation failures",
                    {"image": image, "failures": fails},
                )
                gauge_alerts = (
                    getattr(_me, "container_creation_alerts_total", None) if _me else None
                )
                if gauge_alerts is not None:
                    try:
                        gauge_alerts.labels(image=image).inc()
                    except Exception:
                        logger.exception(
                            "failed to increment container_creation_alerts_total"
                        )
            except Exception:
                logger.exception(
                    "failed to dispatch container creation alert"
                )
            attempt += 1
        assert last_exc is not None
        raise last_exc


def _verify_container(
    container: Any,
    *,
    progress: Callable[[], None] | None = None,
) -> bool:
    """Return ``True`` if ``container`` is healthy and running."""
    heartbeat_timeout = max(_DOCKER_CLIENT_TIMEOUT, 1.0)
    try:
        _call_with_progress(
            container.reload,
            progress=progress,
            heartbeat_timeout=heartbeat_timeout,
        )
        if getattr(container, "status", "running") != "running":
            return False
        attrs = getattr(container, "attrs", {})
        health = attrs.get("State", {}).get("Health", {}).get("Status")
        if health and health != "healthy":
            return False
        return True
    except Exception as exc:
        logger.warning("container reload failed: %s", exc)
        return False


async def _get_pooled_container(image: str) -> tuple[Any, str]:
    """Return a container for ``image`` from the pool, creating if needed."""
    async with pool_lock():
        while True:
            with _POOL_LOCK:
                pool = _CONTAINER_POOLS.setdefault(image, [])
                if pool:
                    c = pool.pop()
                    _CONTAINER_LAST_USED.pop(c.id, None)
                else:
                    c = None
            if c is None:
                break
            if _verify_container(c):
                _ensure_pool_size_async(image)
                with _POOL_LOCK:
                    dir_path = _CONTAINER_DIRS[c.id]
                try:
                    img_tag = getattr(getattr(c, "image", None), "tags", None)
                    img_tag = img_tag[0] if img_tag else image
                    logger.info(
                        "reusing pooled container %s for image %s", c.id, img_tag
                    )
                except Exception as exc:
                    logger.debug("failed to log pooled container reuse %s: %s", c.id, exc)
                return c, dir_path
            success = _stop_and_remove(c)
            _log_cleanup_event(c.id, "unhealthy", success)
            _CREATE_FAILURES[image] += 1
            with _POOL_LOCK:
                td = _CONTAINER_DIRS.pop(c.id, None)
            if td:
                try:
                    shutil.rmtree(td)
                except Exception:
                    logger.exception("temporary directory removal failed for %s", td)
            with _POOL_LOCK:
                _CONTAINER_LAST_USED.pop(c.id, None)
            _ensure_pool_size_async(image)
        container, td = await _create_pool_container(image)
        _ensure_pool_size_async(image)
        return container, td


@asynccontextmanager
async def pooled_container(image: str) -> Any:
    """Async context manager yielding a pooled container."""
    ensure_docker_client()
    container, td = await _get_pooled_container(image)
    try:
        yield container, td
    finally:
        _release_container(image, container)


def _release_container(image: str, container: Any) -> None:
    """Return ``container`` to the pool for ``image``."""
    try:
        container.reload()
        if getattr(container, "status", "running") != "running":
            raise RuntimeError("container not running")
        with _POOL_LOCK:
            _CONTAINER_POOLS.setdefault(image, []).append(container)
            _CONTAINER_LAST_USED[container.id] = time.time()
        _register_container_finalizer(container)
    except Exception:
        try:
            container.remove(force=True)
        except Exception:
            logger.exception("container remove failed")
        cid = getattr(container, "id", "")
        if cid:
            _remove_active_container(cid)
        with _POOL_LOCK:
            td = _CONTAINER_DIRS.pop(cid, None)
            _CONTAINER_LAST_USED.pop(cid, None)
        if td:
            try:
                shutil.rmtree(td)
            except Exception:
                logger.exception("temporary directory removal failed")
    finally:
        _ensure_pool_size_async(image)


def collect_metrics(
    prev_roi: float,
    roi: float,
    resources: Dict[str, float] | None,
) -> Dict[str, float]:
    """Return metrics about container failures and cleanup."""
    result = {
        f"container_failures_{img}": float(cnt) for img, cnt in _CREATE_FAILURES.items()
    }
    result.update(
        {
            f"consecutive_failures_{img}": float(val)
            for img, val in _CONSECUTIVE_CREATE_FAILURES.items()
        }
    )
    result["container_backoff_base"] = float(_CREATE_BACKOFF_BASE)
    result.update({f"cleanup_{k}": float(v) for k, v in _CLEANUP_METRICS.items()})
    result.update({f"watchdog_{k}": float(v) for k, v in _WATCHDOG_METRICS.items()})
    result["stale_containers_removed"] = float(_STALE_CONTAINERS_REMOVED)
    result["stale_vms_removed"] = float(_STALE_VMS_REMOVED)
    result["cleanup_failures"] = float(_CLEANUP_FAILURES)
    result["force_kills"] = float(_FORCE_KILLS)
    result["runtime_vms_removed"] = float(_RUNTIME_VMS_REMOVED)
    result["overlay_cleanup_failures"] = float(_OVERLAY_CLEANUP_FAILURES)
    result["active_container_limit_reached"] = float(_ACTIVE_CONTAINER_LIMIT_REACHED)
    result["active_overlay_limit_reached"] = float(_ACTIVE_OVERLAY_LIMIT_REACHED)
    result["active_containers"] = float(len(_read_active_containers()))
    result["active_overlays"] = float(len(_read_active_overlays()))
    result["cleanup_retry_successes"] = float(_CLEANUP_RETRY_SUCCESSES)
    result["cleanup_retry_failures"] = float(_CLEANUP_RETRY_FAILURES)
    result["consecutive_cleanup_failures"] = float(_CONSECUTIVE_CLEANUP_FAILURES)
    result["cleanup_duration_seconds_cleanup"] = float(_CLEANUP_DURATIONS["cleanup"])
    result["cleanup_duration_seconds_reaper"] = float(_CLEANUP_DURATIONS["reaper"])
    if _LAST_AUTOPURGE_TS:
        result["hours_since_autopurge"] = float((time.time() - _LAST_AUTOPURGE_TS) / 3600.0)
    else:
        result["hours_since_autopurge"] = float("inf")
    stats = _read_cleanup_stats()
    for k, v in stats.items():
        result[f"{k}_total"] = float(v)

    try:
        from . import metrics_exporter as _me
    except Exception:
        try:  # pragma: no cover - package may not be available
            import metrics_exporter as _me  # type: ignore
        except Exception:
            _me = None  # type: ignore
    if _me is not None:
        keys = [
            "cleanup_idle",
            "cleanup_unhealthy",
            "cleanup_lifetime",
            "cleanup_disk",
            "stale_containers_removed",
            "stale_vms_removed",
            "cleanup_failures",
            "force_kills",
            "runtime_vms_removed",
            "overlay_cleanup_failures",
            "active_container_limit_reached",
            "active_containers",
            "active_overlays",
            "cleanup_retry_successes",
            "cleanup_retry_failures",
            "hours_since_autopurge",
        ]
        keys.extend(f"{k}_total" for k in stats)
        for key in keys:
            gauge = getattr(_me, key, None)
            if gauge is not None:
                try:
                    gauge.set(result.get(key, 0.0))
                except (AttributeError, ValueError):
                    logger.exception("failed to update gauge %s", key)
        dur_gauge = getattr(_me, "cleanup_duration_gauge", None)
        if dur_gauge is not None:
            try:
                dur_gauge.labels(worker="cleanup").set(
                    result.get("cleanup_duration_seconds_cleanup", 0.0)
                )
                dur_gauge.labels(worker="reaper").set(
                    result.get("cleanup_duration_seconds_reaper", 0.0)
                )
            except (AttributeError, ValueError):
                logger.exception("failed to update gauge cleanup_duration_gauge")
    return result


async def collect_metrics_async(
    prev_roi: float,
    roi: float,
    resources: Dict[str, float] | None,
) -> Dict[str, float]:
    """Async wrapper for :func:`collect_metrics`."""
    return collect_metrics(prev_roi, roi, resources)


def _stop_and_remove(
    container: Any,
    retries: int = 3,
    base_delay: float = 0.1,
    *,
    progress: Callable[[], None] | None = None,
) -> bool:
    """Stop and remove ``container`` with retries.

    Returns ``True`` when the container no longer exists after attempts.
    """
    global _CLEANUP_FAILURES, _FORCE_KILLS
    cid = getattr(container, "id", "")
    for attempt in range(retries):
        try:
            _call_with_progress(
                container.stop,
                progress=progress,
                heartbeat_timeout=_DOCKER_CLIENT_TIMEOUT,
                timeout=0,
            )
            break
        except Exception as exc:
            if attempt == retries - 1:
                logger.error("failed to stop container %s: %s", cid, exc)
            else:
                time.sleep(base_delay * (2**attempt))
    for attempt in range(retries):
        try:
            _call_with_progress(
                container.remove,
                progress=progress,
                heartbeat_timeout=_DOCKER_CLIENT_TIMEOUT,
                force=True,
            )
            break
        except Exception as exc:
            if attempt == retries - 1:
                logger.error("failed to remove container %s: %s", cid, exc)
            else:
                time.sleep(base_delay * (2**attempt))

    exists = False
    if cid:
        try:
            proc = _run_subprocess_with_progress(
                ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                exists = True
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - timeout handling
            logger.warning(
                "container existence check timed out for %s (%.1fs)",
                cid,
                float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
            )
            exists = True
        except Exception as exc:  # pragma: no cover - unexpected runtime issues
            logger.debug("container existence check failed for %s: %s", cid, exc)

    if cid and exists:
        try:
            proc = _run_subprocess_with_progress(
                ["docker", "rm", "-f", cid],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
            exists = False
            try:
                confirm = _run_subprocess_with_progress(
                    ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
                if confirm.returncode == 0 and confirm.stdout.strip():
                    exists = True
            except subprocess.TimeoutExpired as exc:
                logger.warning(
                    "container removal verification timed out for %s (%.1fs)",
                    cid,
                    float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                )
                exists = True
            except Exception as exc:
                logger.debug("container existence re-check failed for %s: %s", cid, exc)
        except subprocess.TimeoutExpired as exc:
            _CLEANUP_FAILURES += 1
            logger.error(
                "docker rm fallback timed out for container %s (%.1fs)",
                cid,
                float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
            )
        except Exception as exc:
            _CLEANUP_FAILURES += 1
            logger.error("docker rm fallback failed for container %s: %s", cid, exc)

    if cid and exists:
        try:
            _run_subprocess_with_progress(
                ["docker", "kill", cid],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
            _run_subprocess_with_progress(
                ["docker", "rm", "-f", cid],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
            exists = False
            try:
                confirm = _run_subprocess_with_progress(
                    ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
                if confirm.returncode == 0 and confirm.stdout.strip():
                    exists = True
            except subprocess.TimeoutExpired as exc:
                logger.warning(
                    "container kill verification timed out for %s (%.1fs)",
                    cid,
                    float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
                )
                exists = True
            except Exception as exc:
                logger.debug("container existence re-check failed for %s: %s", cid, exc)
            _FORCE_KILLS += 1
        except subprocess.TimeoutExpired as exc:
            _CLEANUP_FAILURES += 1
            logger.error(
                "docker kill escalation timed out for container %s (%.1fs)",
                cid,
                float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
            )
        except Exception as exc:
            _CLEANUP_FAILURES += 1
            logger.error("docker kill escalation failed for container %s: %s", cid, exc)

    if cid and not exists:
        _remove_active_container(cid)
        _remove_failed_cleanup(cid)
        try:
            img_tag = getattr(getattr(container, "image", None), "tags", None)
            if img_tag:
                img_tag = img_tag[0]
            logger.info(
                "container %s (image %s) shutdown and removed", cid, img_tag or "?"
            )
        except Exception as exc:
            logger.debug("failed to log container removal %s: %s", cid, exc)
    elif cid and exists:
        _CLEANUP_FAILURES += 1
        logger.error("container %s still exists after removal attempts", cid)
        _record_failed_cleanup(cid)
    return not exists


def _log_pool_metrics(image: str) -> None:
    """Persist container failure metrics for ``image``."""
    metrics: Dict[str, Any] = {}
    try:
        if _POOL_METRICS_FILE.exists():
            metrics = json.loads(_POOL_METRICS_FILE.read_text())
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed reading pool metrics %s: %s", _POOL_METRICS_FILE, exc)
    img = metrics.get(image, {})
    img["failures"] = float(_CREATE_FAILURES.get(image, 0))
    img["consecutive"] = float(_CONSECUTIVE_CREATE_FAILURES.get(image, 0))
    metrics[image] = img
    try:
        _POOL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _POOL_METRICS_FILE.write_text(json.dumps(metrics))
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed writing pool metrics %s: %s", _POOL_METRICS_FILE, exc)


def report_failed_cleanup(
    threshold: float | None = None, *, alert: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Return failed cleanup entries older than ``threshold``.

    When ``alert`` is ``True``, log errors and send a diagnostic record
    if any entries exceed the threshold.
    """
    if threshold is None:
        threshold = _FAILED_CLEANUP_ALERT_AGE
    data = _read_failed_cleanup()
    now = time.time()
    stale = {
        item: meta
        for item, meta in data.items()
        if now - float(meta.get("ts", 0.0)) >= threshold
    }
    if alert and stale:
        try:
            logger.error(
                "failed cleanup items: %s",
                {item: meta.get("reason", "") for item, meta in stale.items()},
            )
        except Exception as exc:
            _fallback_logger().error(
                "failed cleanup items %s (logging failed: %s)",
                {item: meta.get("reason", "") for item, meta in stale.items()},
                exc,
                exc_info=True,
            )
        _log_diagnostic("failed_cleanup", False)
    return stale


def _get_dir_usage(path: str) -> int:
    """Return total size of files under ``path`` in bytes."""
    total = 0
    for root_dir, _, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root_dir, fname))
            except OSError as exc:
                logger.warning(
                    "size check failed for %s: %s", path_for_prompt(path), exc
                )
    return total


def _check_disk_usage(cid: str) -> bool:
    """Return ``True`` if container directory exceeds ``_CONTAINER_DISK_LIMIT``."""
    with _POOL_LOCK:
        td = _CONTAINER_DIRS.get(cid)
    if not td:
        return False
    try:
        return _get_dir_usage(td) > _CONTAINER_DISK_LIMIT
    except Exception as exc:
        logger.warning("disk usage check failed for %s: %s", td, exc)
        return False


def _cleanup_idle_containers(
    *, progress: Callable[[], None] | None = None
) -> tuple[int, int]:
    """Remove idle or unhealthy containers.

    Returns a tuple ``(cleaned, replaced)`` where ``cleaned`` is the number of
    idle containers removed and ``replaced`` is the number of unhealthy
    containers purged. When ``progress`` is supplied, the callback is invoked
    as work advances to provide heartbeat updates.
    """
    if _DOCKER_CLIENT is None:
        _notify_progress(progress)
        return 0, 0
    cleaned = 0
    replaced = 0
    now = time.time()
    _notify_progress(progress)
    with _POOL_LOCK:
        pools_snapshot = {img: list(pool) for img, pool in _CONTAINER_POOLS.items()}
    for image, pool in pools_snapshot.items():
        _notify_progress(progress)
        for c in list(pool):
            _notify_progress(progress)
            with _POOL_LOCK:
                last = _CONTAINER_LAST_USED.get(c.id, 0)
                created = _CONTAINER_CREATED.get(c.id, last)
            reason = None
            if _CONTAINER_DISK_LIMIT and _check_disk_usage(c.id):
                reason = "disk"
            elif now - created > _CONTAINER_MAX_LIFETIME:
                reason = "lifetime"
            elif now - last > _CONTAINER_IDLE_TIMEOUT:
                reason = "idle"
            elif not _verify_container(c, progress=progress):
                reason = "unhealthy"
            if reason:
                _notify_progress(progress)
                with _POOL_LOCK:
                    actual_pool = _CONTAINER_POOLS.get(image, [])
                    if c in actual_pool:
                        actual_pool.remove(c)
                success = _stop_and_remove(c, progress=progress)
                _notify_progress(progress)
                _log_cleanup_event(c.id, reason, success)
                global _STALE_CONTAINERS_REMOVED
                _STALE_CONTAINERS_REMOVED += 1
                with _POOL_LOCK:
                    td = _CONTAINER_DIRS.pop(c.id, None)
                if td:
                    try:
                        shutil.rmtree(td)
                    except Exception:
                        logger.exception(
                            "temporary directory removal failed for %s", td
                        )
                    _notify_progress(progress)
                with _POOL_LOCK:
                    _CONTAINER_LAST_USED.pop(c.id, None)
                    _CONTAINER_CREATED.pop(c.id, None)
                if reason == "idle":
                    cleaned += 1
                else:
                    replaced += 1
                _CLEANUP_METRICS[reason] += 1
                _ensure_pool_size_async(image)
                _notify_progress(progress)
    _notify_progress(progress)
    return cleaned, replaced


def _reap_orphan_containers(*, progress: Callable[[], None] | None = None) -> int:
    """Remove containers labeled with :data:`_POOL_LABEL` not in the pool."""
    if _DOCKER_CLIENT is None:
        return 0
    try:
        containers = _call_with_progress(
            _DOCKER_CLIENT.containers.list,
            progress=progress,
            heartbeat_timeout=_DOCKER_CLIENT_TIMEOUT,
            all=True,
            filters={"label": f"{_POOL_LABEL}=1"},
        )
    except DockerException as exc:
        logger.warning("orphan container listing failed: %s", exc)
        return 0
    with _POOL_LOCK:
        active = {c.id for pool in _CONTAINER_POOLS.values() for c in pool}
    removed = 0
    for c in list(containers):
        if c.id in active:
            continue
        try:
            _verify_container(c, progress=progress)
        except Exception as exc:
            logger.debug("failed to verify container %s: %s", c.id, exc)
        success = _stop_and_remove(c, progress=progress)
        _log_cleanup_event(c.id, "orphan", success)
        removed += 1
        _notify_progress(progress)
        with _POOL_LOCK:
            td = _CONTAINER_DIRS.pop(c.id, None)
        if td:
            try:
                shutil.rmtree(td)
            except Exception:
                logger.exception("temporary directory removal failed for %s", td)
        with _POOL_LOCK:
            _CONTAINER_LAST_USED.pop(c.id, None)
            _CONTAINER_CREATED.pop(c.id, None)
        _CLEANUP_METRICS["orphan"] += 1
        _notify_progress(progress)
    return removed


def reconcile_active_containers() -> None:
    """Remove untracked containers labeled with :data:`_POOL_LABEL`."""
    try:
        proc = _run_subprocess_with_progress(
            ["docker", "ps", "-aq", "--filter", f"label={_POOL_LABEL}=1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode != 0:
            return
        ids = [cid.strip() for cid in proc.stdout.splitlines() if cid.strip()]
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "docker ps timed out during reconciliation (%.1fs)",
            float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
        )
        return
    except DockerException as exc:
        logger.debug("active container reconciliation failed: %s", exc)
        return

    if not ids:
        return

    recorded = set(_read_active_containers())
    with _POOL_LOCK:
        pooled = {c.id for pool in _CONTAINER_POOLS.values() for c in pool}

    for cid in ids:
        if cid in pooled or cid in recorded:
            continue
        try:
            logger.info("removing untracked sandbox container %s", cid)
        except Exception as exc:
            logger.debug("failed to log untracked container removal %s: %s", cid, exc)
        try:
            _run_subprocess_with_progress(
                ["docker", "rm", "-f", cid],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "removing untracked sandbox container %s timed out (%.1fs)",
                cid,
                float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
            )
            _record_failed_cleanup(
                f"container:{cid}",
                reason="docker rm timeout during reconciliation",
            )
            return
        _remove_active_container(cid)
        with _POOL_LOCK:
            td = _CONTAINER_DIRS.pop(cid, None)
            _CONTAINER_LAST_USED.pop(cid, None)
            _CONTAINER_CREATED.pop(cid, None)
        if td:
            try:
                shutil.rmtree(td)
            except Exception:
                if os.name == "nt":
                    if _rmtree_windows(td):
                        logger.debug(
                            "temporary directory removed via Windows fallback for %s",
                            td,
                        )
                    else:
                        logger.warning(
                            "temporary directory removal failed for %s", td,
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "temporary directory removal failed for %s", td,
                        exc_info=True,
                    )


def _resolve_cleanup_command(item: str) -> tuple[List[str], List[str] | None]:
    """Return cleanup and verification commands for ``item``.

    Raises ``ValueError`` when ``item`` is not a recognized cleanup target.
    """

    if not item:
        raise ValueError("empty cleanup item")

    target = item
    verify: List[str] | None = None
    if ":" in item:
        prefix, suffix = item.split(":", 1)
        if prefix == "volume" and suffix:
            target = suffix
            return (
                ["docker", "volume", "rm", "-f", target],
                ["docker", "volume", "ls", "-q", "--filter", f"name={target}"],
            )
        if prefix == "network" and suffix:
            target = suffix
            return (
                ["docker", "network", "rm", "-f", target],
                ["docker", "network", "ls", "-q", "--filter", f"name={target}"],
            )
        if prefix == "container" and suffix:
            target = suffix

    command = ["docker", "rm", "-f", target]
    verify = ["docker", "ps", "-aq", "--filter", f"id={target}"]
    return command, verify


def retry_failed_cleanup(*, progress: Callable[[], None] | None = None) -> tuple[int, int]:
    """Retry deletion of items recorded in :data:`FAILED_CLEANUP_FILE`.

    Each entry represents a previous cleanup failure. This function retries the
    removal and returns a ``(successes, failures)`` tuple.
    """
    data = _read_failed_cleanup()
    successes = 0
    failures = 0
    for item, meta in list(data.items()):
        _notify_progress(progress)
        reason = str(meta.get("reason", ""))
        is_path = os.path.sep in item or os.path.exists(item)
        if is_path:
            try:
                shutil.rmtree(item)
                _remove_failed_cleanup(item)
                successes += 1
                _notify_progress(progress)
                continue
            except OSError as exc:
                logger.warning(
                    "cleanup retry failed",
                    extra={"path": path_for_prompt(item)},
                    exc_info=exc,
                )
                if os.name == "nt" and _rmtree_windows(item):
                    _remove_failed_cleanup(item)
                    successes += 1
                    _notify_progress(progress)
                    continue
                _record_failed_cleanup(
                    item,
                    reason=reason or f"path removal failed: {exc}",
                )
                failures += 1
                _notify_progress(progress)
                continue
        try:
            command, verify = _resolve_cleanup_command(item)
        except ValueError:
            failures += 1
            _notify_progress(progress)
            continue
        try:
            _run_subprocess_with_progress(
                command,
                progress=progress,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "cleanup retry timed out for %s (%.1fs)",
                item,
                float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
            )
            _record_failed_cleanup(
                item,
                reason=reason or f"timeout running {' '.join(command)}",
            )
            failures += 1
            _notify_progress(progress)
            continue
        except Exception as exc:
            logger.debug("cleanup retry failed for %s: %s", item, exc)
            _record_failed_cleanup(
                item,
                reason=reason or f"command failed: {' '.join(command)}",  # type: ignore[arg-type]
            )
            failures += 1
            _notify_progress(progress)
            continue

        if verify is None:
            _remove_failed_cleanup(item)
            successes += 1
            _notify_progress(progress)
            continue

        try:
            proc = _run_subprocess_with_progress(
                verify,
                progress=progress,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
            _notify_progress(progress)
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "cleanup verification timed out for %s (%.1fs)",
                item,
                float(exc.timeout or _CLEANUP_SUBPROCESS_TIMEOUT),
            )
            _record_failed_cleanup(
                item,
                reason=reason or f"timeout verifying {' '.join(verify)}",
            )
            failures += 1
            _notify_progress(progress)
            continue
        except Exception as exc:
            logger.debug("cleanup verification failed for %s: %s", item, exc)
            _record_failed_cleanup(
                item,
                reason=reason or f"verification failed: {' '.join(verify)}",
            )
            failures += 1
            _notify_progress(progress)
            continue

        if proc.returncode == 0 and not proc.stdout.strip():
            _remove_failed_cleanup(item)
            successes += 1
            _notify_progress(progress)
        else:
            failures += 1
            _notify_progress(progress)
    
    global _CLEANUP_RETRY_SUCCESSES, _CLEANUP_RETRY_FAILURES
    if successes:
        _CLEANUP_RETRY_SUCCESSES += successes
        _increment_cleanup_stat("cleanup_retry_successes", successes)
    if failures:
        _CLEANUP_RETRY_FAILURES += failures
        _increment_cleanup_stat("cleanup_retry_failures", failures)
        if failures > _MAX_FAILURE_ATTEMPTS:
            try:
                logger.warning(
                    "failsafe prune triggered after %s failures", failures
                )
            except Exception as exc:
                logger.debug("failed to log failsafe prune: %s", exc)
            try:
                _run_subprocess_with_progress(
                    ["docker", "system", "prune", "-f", "--volumes"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                )
            except Exception:
                logger.exception("failsafe docker prune failed")

    try:
        from . import metrics_exporter as _me
    except Exception:
        try:  # pragma: no cover - package may not be available
            import metrics_exporter as _me  # type: ignore
        except Exception:
            _me = None  # type: ignore
    if _me is not None:
        gauge = getattr(_me, "cleanup_retry_successes", None)
        if gauge is not None and successes:
            try:
                gauge.inc(successes)
            except Exception:
                logger.exception("failed to increment cleanup_retry_successes")
        gauge = getattr(_me, "cleanup_retry_failures", None)
        if gauge is not None and failures:
            try:
                gauge.inc(failures)
            except Exception:
                logger.exception("failed to increment cleanup_retry_failures")

    stale = report_failed_cleanup(alert=True)
    global _CONSECUTIVE_CLEANUP_FAILURES
    if stale:
        try:
            logger.warning("persistent cleanup failures: %s", list(stale.keys()))
        except Exception as exc:
            _fallback_logger().warning(
                "persistent cleanup failures %s (logging failed: %s)",
                list(stale.keys()),
                exc,
                exc_info=True,
            )
        _log_diagnostic("cleanup_retry_failure", False)
        _CONSECUTIVE_CLEANUP_FAILURES += 1
        if _CONSECUTIVE_CLEANUP_FAILURES > _CLEANUP_ALERT_THRESHOLD:
            try:
                logger.error(
                    "cleanup retries failing %s times consecutively",
                    _CONSECUTIVE_CLEANUP_FAILURES,
                )
            except Exception as exc:
                _fallback_logger().error(
                    "cleanup retries failing %s times (logging failed: %s)",
                    _CONSECUTIVE_CLEANUP_FAILURES,
                    exc,
                    exc_info=True,
                )
            _log_diagnostic("persistent_cleanup_failure", False)
    else:
        _CONSECUTIVE_CLEANUP_FAILURES = 0

    return successes, failures


def _get_metrics_module() -> Any | None:
    """Return the metrics exporter module if available."""

    try:
        from . import metrics_exporter as _me
    except Exception:
        try:  # pragma: no cover - package may not be available
            import metrics_exporter as _me  # type: ignore
        except Exception:
            return None
    return _me


def _update_worker_heartbeat(worker: str, *, when: float | None = None) -> None:
    """Record a heartbeat timestamp for the given cleanup worker."""

    timestamp = when if when is not None else time.monotonic()
    global _LAST_CLEANUP_TS, _LAST_REAPER_TS
    if worker == "cleanup":
        _LAST_CLEANUP_TS = timestamp
        _WORKER_ACTIVITY["cleanup"] = True
    elif worker == "reaper":
        _LAST_REAPER_TS = timestamp
        _WORKER_ACTIVITY["reaper"] = True
    else:
        return

    module = _get_metrics_module()
    gauge = getattr(module, "cleanup_heartbeat_gauge", None) if module else None
    if gauge is not None:
        try:
            gauge.labels(worker=worker).set(timestamp)
        except (AttributeError, ValueError):
            logger.exception("failed to update cleanup heartbeat gauge")


def _notify_progress(callback: Callable[[], None] | None) -> None:
    """Invoke ``callback`` if provided, suppressing unexpected errors."""

    if callback is None:
        return
    try:
        callback()
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("cleanup progress callback failed")


def _heartbeat_interval(timeout: float | None) -> float:
    """Return a conservative heartbeat interval for long running commands."""

    if timeout is None or timeout <= 0:
        return 1.0
    return min(5.0, max(0.5, timeout / 10.0))


_PROGRESS_SCOPE = threading.local()


@contextmanager
def _progress_scope(callback: Callable[[], None] | None):
    """Temporarily register ``callback`` for nested subprocess helpers."""

    previous = getattr(_PROGRESS_SCOPE, "callback", None)
    _PROGRESS_SCOPE.callback = callback
    try:
        yield
    finally:
        _PROGRESS_SCOPE.callback = previous


def _current_progress_callback() -> Callable[[], None] | None:
    """Return the progress callback associated with the current thread."""

    return getattr(_PROGRESS_SCOPE, "callback", None)


@contextmanager
def _background_progress_guard(
    progress: Callable[[], None] | None,
    *,
    interval: float | None = None,
    max_duration: float | None = None,
) -> Iterator[None]:
    """Emit synthetic progress heartbeats while a blocking call is running."""

    if progress is None:
        yield
        return

    beat = float(interval or _HEARTBEAT_GUARD_INTERVAL)
    if not beat or beat <= 0.0:
        beat = 1.0
    else:
        beat = max(0.25, min(5.0, beat))

    limit = max_duration
    if limit is None or limit <= 0.0:
        limit = _HEARTBEAT_GUARD_MAX_DURATION
    try:
        limit = float(limit)
    except (TypeError, ValueError):
        limit = _HEARTBEAT_GUARD_MAX_DURATION
    if limit < 0:
        limit = 0.0

    stop_event = threading.Event()
    start_ts = time.monotonic()

    def _publisher() -> None:
        try:
            _notify_progress(progress)
            while not stop_event.wait(beat):
                _notify_progress(progress)
                if limit > 0 and time.monotonic() - start_ts >= limit:
                    logger.warning(
                        "background heartbeat exceeded %.1fs without foreground completion; allowing watchdog to intervene",
                        limit,
                    )
                    break
        except Exception:
            logger.exception("background heartbeat publisher failed")
        finally:
            stop_event.set()

    thread = threading.Thread(
        target=_publisher,
        name="sandbox-progress-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=max(beat * 2.0, 1.0))
        try:
            _notify_progress(progress)
        except Exception:
            logger.debug("final progress notification failed", exc_info=True)


@contextmanager
def _progress_heartbeat_scope(
    progress: Callable[[], None] | None,
    *,
    interval: float | None = None,
    max_duration: float | None = None,
) -> Iterator[None]:
    """Combine :func:`_progress_scope` with the heartbeat guard."""

    if progress is None:
        with nullcontext():
            yield
        return

    with _progress_scope(progress):
        with _background_progress_guard(
            progress,
            interval=interval,
            max_duration=max_duration,
        ):
            yield


def _run_subprocess_with_progress(
    args: Sequence[str] | str,
    *,
    progress: Callable[[], None] | None = None,
    heartbeat_interval: float | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Execute ``subprocess.run`` while emitting periodic progress heartbeats.

    Docker commands on Windows can hang for noticeable periods when the
    daemon is waking up or proxying requests through Hyper-V.  During that
    time the cleanup workers would previously stop emitting heartbeats,
    triggering the watchdog to emit ``worker stalled`` warnings.  This helper
    keeps the heartbeat flowing from a lightweight background thread so the
    watchdog only reacts to genuine failures.
    """

    stop_event: threading.Event | None = None
    hb_thread: threading.Thread | None = None
    interval = heartbeat_interval or _heartbeat_interval(kwargs.get("timeout"))
    callback = progress if progress is not None else _current_progress_callback()

    try:
        if callback is not None:
            _notify_progress(callback)
            stop_event = threading.Event()

            def _beat() -> None:
                try:
                    while not stop_event.wait(interval):
                        _notify_progress(callback)
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception("heartbeat notifier failed")

            hb_thread = threading.Thread(
                target=_beat,
                name="sandbox-subprocess-heartbeat",
                daemon=True,
            )
            hb_thread.start()

        return subprocess.run(args, **kwargs)
    finally:
        if stop_event is not None:
            stop_event.set()
        if hb_thread is not None:
            hb_thread.join(timeout=1.0)
        if callback is not None:
            _notify_progress(callback)


def _call_with_progress(
    func: Callable[..., T],
    *args: Any,
    progress: Callable[[], None] | None = None,
    heartbeat_timeout: float | None = None,
    max_duration: float | None = None,
    **kwargs: Any,
) -> T:
    """Invoke ``func`` while emitting periodic heartbeat notifications.

    The helper mirrors :func:`_run_subprocess_with_progress` but targets
    synchronous Docker SDK interactions where ``func`` blocks the calling
    thread.  Windows installations proxy Docker requests through a named pipe
    exposed by Docker Desktop which frequently incurs multi-second delays when
    the daemon is resuming from sleep.  Without explicit heartbeats those
    pauses look indistinguishable from hung workers and the watchdog eagerly
    resets the cleanup tasks.  By funnelling blocking SDK calls through this
    helper we guarantee that the cleanup loop continues to emit heartbeats for
    the duration of the call while still respecting the configured watchdog
    guard rails.
    """

    interval = _heartbeat_interval(heartbeat_timeout)
    guard = (
        _HEARTBEAT_GUARD_MAX_DURATION
        if max_duration is None
        else max(float(max_duration), 0.0)
    )
    with _progress_heartbeat_scope(
        progress,
        interval=interval,
        max_duration=guard,
    ):
        return func(*args, **kwargs)


async def _cleanup_worker() -> None:
    """Background task to clean idle containers."""
    total_cleaned = 0
    total_replaced = 0

    def _progress() -> None:
        _update_worker_heartbeat("cleanup")

    try:
        while True:
            _update_worker_heartbeat("cleanup")
            try:
                with _progress_heartbeat_scope(
                    _progress,
                    interval=_heartbeat_interval(None),
                ):
                    autopurge_if_needed()
                _notify_progress(_progress)
                await asyncio.sleep(0)
                with _progress_heartbeat_scope(
                    _progress,
                    interval=_heartbeat_interval(None),
                    max_duration=_HEARTBEAT_GUARD_MAX_DURATION,
                ):
                    ensure_docker_client()
                _notify_progress(_progress)
                await asyncio.sleep(0)
                with _progress_heartbeat_scope(
                    _progress,
                    interval=_heartbeat_interval(None),
                ):
                    reconcile_active_containers()
                _notify_progress(_progress)
                await asyncio.sleep(0)
            finally:
                _update_worker_heartbeat("cleanup")
            await asyncio.sleep(_POOL_CLEANUP_INTERVAL)
            _update_worker_heartbeat("cleanup")
            start = time.monotonic()
            _CLEANUP_CURRENT_RUNTIME["cleanup"] = 0.0
            try:
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    retry_failed_cleanup(progress=_progress)
                _notify_progress(_progress)
                _CLEANUP_CURRENT_RUNTIME["cleanup"] = time.monotonic() - start
                _update_worker_heartbeat("cleanup")
                await asyncio.sleep(0)
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    cleaned, replaced = _cleanup_idle_containers(progress=_progress)
                total_cleaned += cleaned
                total_replaced += replaced
                _notify_progress(_progress)
                _CLEANUP_CURRENT_RUNTIME["cleanup"] = time.monotonic() - start
                _update_worker_heartbeat("cleanup")
                await asyncio.sleep(0)
                with _progress_heartbeat_scope(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    vm_removed = _purge_stale_vms(record_runtime=True)
                _CLEANUP_CURRENT_RUNTIME["cleanup"] = time.monotonic() - start
                _update_worker_heartbeat("cleanup")
                await asyncio.sleep(0)
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    _prune_volumes(progress=_progress)
                _notify_progress(_progress)
                _CLEANUP_CURRENT_RUNTIME["cleanup"] = time.monotonic() - start
                _update_worker_heartbeat("cleanup")
                await asyncio.sleep(0)
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    _prune_networks(progress=_progress)
                _notify_progress(_progress)
                _CLEANUP_CURRENT_RUNTIME["cleanup"] = time.monotonic() - start
                _update_worker_heartbeat("cleanup")
                await asyncio.sleep(0)
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    report_failed_cleanup(alert=True)
                _CLEANUP_CURRENT_RUNTIME["cleanup"] = time.monotonic() - start
                _update_worker_heartbeat("cleanup")
                await asyncio.sleep(0)
                if cleaned:
                    logger.info(
                        "cleaned %d idle containers (total %d)", cleaned, total_cleaned
                    )
                if replaced:
                    logger.info(
                        "replaced %d unhealthy containers (total %d)",
                        replaced,
                        total_replaced,
                    )
                if vm_removed:
                    logger.info(
                        "removed %d stale VM files (total %d)",
                        vm_removed,
                        _RUNTIME_VMS_REMOVED,
                    )
            except Exception:
                logger.exception("idle container cleanup failed")
            finally:
                duration = time.monotonic() - start
                _CLEANUP_DURATIONS["cleanup"] = duration
                budget = globals().get("_WATCHDOG_BUDGET")
                record = getattr(budget, "record", None) if budget is not None else None
                if callable(record):
                    try:
                        record("cleanup", duration)
                    except Exception:
                        logger.debug(
                            "watchdog budget record failed", exc_info=True
                        )
                _me = _get_metrics_module()
                gauge = getattr(_me, "cleanup_duration_gauge", None) if _me else None
                if gauge is not None:
                    try:
                        gauge.labels(worker="cleanup").set(duration)
                    except (AttributeError, ValueError):
                        logger.exception("failed to update cleanup duration")
                _update_worker_heartbeat("cleanup")
                _CLEANUP_CURRENT_RUNTIME["cleanup"] = 0.0
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        logger.debug("cleanup worker cancelled")
        raise


async def _reaper_worker() -> None:
    """Background task to reap orphan containers."""
    total_removed = 0

    def _progress() -> None:
        _update_worker_heartbeat("reaper")

    try:
        while True:
            _update_worker_heartbeat("reaper")
            await asyncio.sleep(_POOL_CLEANUP_INTERVAL)
            _update_worker_heartbeat("reaper")
            try:
                with _progress_heartbeat_scope(
                    _progress,
                    interval=_heartbeat_interval(None),
                ):
                    autopurge_if_needed()
                _notify_progress(_progress)
                _update_worker_heartbeat("reaper")
                await asyncio.sleep(0)
                with _progress_heartbeat_scope(
                    _progress,
                    interval=_heartbeat_interval(None),
                ):
                    reconcile_active_containers()
                _notify_progress(_progress)
                _update_worker_heartbeat("reaper")
                await asyncio.sleep(0)
            finally:
                _update_worker_heartbeat("reaper")
            start = time.monotonic()
            _CLEANUP_CURRENT_RUNTIME["reaper"] = 0.0
            try:
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    removed = _reap_orphan_containers(progress=_progress)
                _notify_progress(_progress)
                _CLEANUP_CURRENT_RUNTIME["reaper"] = time.monotonic() - start
                _update_worker_heartbeat("reaper")
                await asyncio.sleep(0)
                with _progress_heartbeat_scope(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    vm_removed = _purge_stale_vms(record_runtime=True)
                _CLEANUP_CURRENT_RUNTIME["reaper"] = time.monotonic() - start
                _update_worker_heartbeat("reaper")
                await asyncio.sleep(0)
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    vol_removed = _prune_volumes(progress=_progress)
                _notify_progress(_progress)
                _CLEANUP_CURRENT_RUNTIME["reaper"] = time.monotonic() - start
                _update_worker_heartbeat("reaper")
                await asyncio.sleep(0)
                with _background_progress_guard(
                    _progress,
                    interval=_heartbeat_interval(_CLEANUP_SUBPROCESS_TIMEOUT),
                ):
                    net_removed = _prune_networks(progress=_progress)
                _notify_progress(_progress)
                _CLEANUP_CURRENT_RUNTIME["reaper"] = time.monotonic() - start
                _update_worker_heartbeat("reaper")
                await asyncio.sleep(0)
                total_removed += removed + vm_removed
                if removed or vm_removed or vol_removed or net_removed:
                    logger.info(
                        "reaped %d orphan containers, %d stale VMs, %d volumes, %d networks (totals containers=%d)",
                        removed,
                        vm_removed,
                        vol_removed,
                        net_removed,
                        total_removed,
                    )
            except Exception:
                logger.exception("reaper cleanup failed")
            finally:
                duration = time.monotonic() - start
                _CLEANUP_DURATIONS["reaper"] = duration
                budget = globals().get("_WATCHDOG_BUDGET")
                record = getattr(budget, "record", None) if budget is not None else None
                if callable(record):
                    try:
                        record("reaper", duration)
                    except Exception:
                        logger.debug(
                            "watchdog budget record failed", exc_info=True
                        )
                _me = _get_metrics_module()
                gauge = getattr(_me, "cleanup_duration_gauge", None) if _me else None
                if gauge is not None:
                    try:
                        gauge.labels(worker="reaper").set(duration)
                    except (AttributeError, ValueError):
                        logger.exception("failed to update cleanup duration")
                _update_worker_heartbeat("reaper")
                _CLEANUP_CURRENT_RUNTIME["reaper"] = 0.0
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        logger.debug("reaper worker cancelled")
        raise


def _cleanup_pools() -> None:
    """Stop and remove pooled containers and stale ones."""
    global _CLEANUP_TASK, _REAPER_TASK
    try:
        _POOL_FILE_LOCK.acquire(timeout=_POOL_LOCK_ACQUIRE_TIMEOUT)
    except Timeout:
        logger.warning(
            "skipping pool cleanup because the pool lock is held by another process"
        )
        return
    try:
        stop_container_event_listener()
        if _CLEANUP_TASK:
            _await_cleanup_task()
        if _REAPER_TASK:
            _await_reaper_task()
        cancel_cleanup_check()
        for t in list(_WARMUP_TASKS.values()):
            t.cancel()
        _WARMUP_TASKS.clear()

        if _DOCKER_CLIENT is not None:
            with _POOL_LOCK:
                pools_snapshot = [list(pool) for pool in _CONTAINER_POOLS.values()]
            for pool in pools_snapshot:
                for c in list(pool):
                    success = _stop_and_remove(c)
                    _log_cleanup_event(c.id, "shutdown", success)
                    with _POOL_LOCK:
                        td = _CONTAINER_DIRS.pop(c.id, None)
                    if td:
                        try:
                            shutil.rmtree(td)
                        except Exception:
                            logger.exception(
                                "temporary directory removal failed for %s", td
                            )
                    with _POOL_LOCK:
                        _CONTAINER_LAST_USED.pop(c.id, None)
                        _CONTAINER_CREATED.pop(c.id, None)
            with _POOL_LOCK:
                _CONTAINER_POOLS.clear()

            vm_removed = _purge_stale_vms(record_runtime=True)
            if vm_removed:
                try:
                    logger.info(
                        "removed %d stale VM files (total %d)",
                        vm_removed,
                        _RUNTIME_VMS_REMOVED,
                    )
                except Exception:
                    logger.exception('unexpected error')

        try:
            proc = _run_subprocess_with_progress(
                ["docker", "ps", "-aq", "--filter", f"label={_POOL_LABEL}=1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
            )
            if proc.returncode == 0:
                for cid in proc.stdout.splitlines():
                    cid = cid.strip()
                    if cid:
                        try:
                            logger.info("removing stale sandbox container %s", cid)
                        except Exception:
                            logger.exception('unexpected error')
                        _run_subprocess_with_progress(
                            ["docker", "rm", "-f", cid],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=_CLEANUP_SUBPROCESS_TIMEOUT,
                        )
                    else:
                        try:
                            logger.warning("encountered empty container id during prune")
                        except Exception:
                            logger.exception('unexpected error')
            else:
                try:
                    stderr = (proc.stderr or "").strip()
                except Exception:
                    stderr = ""
                try:
                    logger.error(
                        "failed to list stale sandbox containers (rc=%s): %s",
                        proc.returncode,
                        stderr,
                    )
                except Exception:
                    logger.exception('unexpected error')
        except Exception:
            logger.exception('unexpected error')

        # also prune any leftover volumes and networks so runtime
        # resources are fully released when the sandbox exits
        _prune_volumes()
        _prune_networks()
    finally:
        _release_pool_lock()


def _await_cleanup_task() -> None:
    """Cancel ``_CLEANUP_TASK`` if running and await completion."""
    global _CLEANUP_TASK
    task = _CLEANUP_TASK
    if not task:
        return
    try:
        loop = getattr(task, "get_loop", lambda: None)()
    except Exception:
        loop = None
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(task.cancel)
    else:
        try:
            task.cancel()
        except Exception:
            logger.exception('unexpected error')
    _CLEANUP_TASK = None


def _await_reaper_task() -> None:
    """Cancel ``_REAPER_TASK`` if running and await completion."""
    global _REAPER_TASK
    task = _REAPER_TASK
    if not task:
        return
    try:
        loop = getattr(task, "get_loop", lambda: None)()
    except Exception:
        loop = None
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(task.cancel)
    else:
        try:
            task.cancel()
        except Exception:
            logger.exception('unexpected error')
    _REAPER_TASK = None


def _run_cleanup_sync() -> None:
    """Run cleanup and reaper workers once in a temporary event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        interval = _POOL_CLEANUP_INTERVAL
        try:
            globals()["_POOL_CLEANUP_INTERVAL"] = 0.0

            async def runner() -> None:
                t1 = asyncio.create_task(_cleanup_worker())
                t2 = asyncio.create_task(_reaper_worker())
                await asyncio.sleep(0.05)
                t1.cancel()
                t2.cancel()
                with suppress(asyncio.CancelledError):
                    await t1
                with suppress(asyncio.CancelledError):
                    await t2

            loop.run_until_complete(runner())
        finally:
            globals()["_POOL_CLEANUP_INTERVAL"] = interval
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def start_container_event_listener() -> None:
    """Start background thread listening for container exit events."""
    global _EVENT_THREAD, _EVENT_STOP
    if _SANDBOX_DISABLE_CLEANUP:
        _log_cleanup_disabled("container event listener")
        return
    if _DOCKER_CLIENT is None or (
        _EVENT_THREAD is not None and _EVENT_THREAD.is_alive()
    ):
        return
    if _EVENT_THREAD is not None and not _EVENT_THREAD.is_alive():
        _EVENT_THREAD = None

    stop_event = threading.Event()
    _EVENT_STOP = stop_event

    def _worker() -> None:
        api = None
        try:
            api = docker.APIClient() if docker is not None else None
        except DockerException as exc:
            logger.warning("API client init failed: %s", exc)
            return

        filters = {"label": f"{_POOL_LABEL}=1"}
        while not stop_event.is_set():
            try:
                for ev in api.events(decode=True, filters=filters):
                    if stop_event.is_set():
                        break
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("Type") != "container":
                        continue
                    action = ev.get("Action") or ev.get("status")
                    if action not in {"die", "stop", "exit", "died"}:
                        continue
                    cid = str(ev.get("id") or ev.get("Actor", {}).get("ID", ""))
                    if not cid:
                        continue
                    try:
                        container = _DOCKER_CLIENT.containers.get(cid)
                    except Exception:
                        container = None
                    if container is None:
                        continue
                    try:
                        success = _stop_and_remove(container)
                        _log_cleanup_event(cid, "event", success)
                    except Exception:
                        logger.exception("event cleanup failed for %s", cid)
                    with _POOL_LOCK:
                        for pool in _CONTAINER_POOLS.values():
                            if container in pool:
                                pool.remove(container)
                        td = _CONTAINER_DIRS.pop(container.id, None)
                        _CONTAINER_LAST_USED.pop(container.id, None)
                        _CONTAINER_CREATED.pop(container.id, None)
                    if td:
                        try:
                            shutil.rmtree(td)
                        except Exception:
                            logger.exception(
                                "temporary directory removal failed for %s", td
                            )
                    _CLEANUP_METRICS["event"] += 1
            except Exception as exc:
                logger.exception("container event listener failed: %s", exc)
                time.sleep(1.0)
        if api is not None:
            try:
                api.close()
            except Exception:
                logger.exception('unexpected error')
        global _EVENT_THREAD
        _EVENT_THREAD = None

    thread = threading.Thread(
        target=_worker, daemon=True, name="sandbox-event-listener"
    )
    _EVENT_THREAD = thread
    thread.start()


def stop_container_event_listener() -> None:
    """Stop the background container event listener if running."""
    global _EVENT_THREAD, _EVENT_STOP
    thread = _EVENT_THREAD
    if thread is None:
        return
    if _EVENT_STOP is not None:
        _EVENT_STOP.set()

    # Attempt to join the worker thread, logging errors and retrying if needed.
    for attempt in range(1, 4):
        try:
            thread.join(timeout=1.0)
        except Exception:
            logger.exception("failed to join event listener thread", exc_info=True)
        if not thread.is_alive():
            break
        logger.warning(
            "container event listener thread still running after stop attempt %s", attempt
        )
        time.sleep(0.1)
    else:
        logger.error("container event listener thread did not terminate after retries")

    _EVENT_THREAD = None
    _EVENT_STOP = None


def ensure_cleanup_worker() -> None:
    """Ensure background cleanup worker task is active."""
    global _CLEANUP_TASK, _REAPER_TASK, _LAST_CLEANUP_TS, _LAST_REAPER_TS
    if _SANDBOX_DISABLE_CLEANUP:
        _log_cleanup_disabled("cleanup worker bootstrap")
        _suspend_cleanup_workers()
        return
    if _DOCKER_CLIENT is None:
        _suspend_cleanup_workers()
        return
    if _EVENT_THREAD is None or not _EVENT_THREAD.is_alive():
        start_container_event_listener()
    task = _CLEANUP_TASK
    if task is None:
        _CLEANUP_TASK = _schedule_coroutine(_cleanup_worker())
        if _CLEANUP_TASK is None:
            _run_cleanup_sync()
            return
        _update_worker_heartbeat("cleanup")
        _WORKER_ACTIVITY["cleanup"] = False
        if _REAPER_TASK is None:
            _REAPER_TASK = _schedule_coroutine(_reaper_worker())
            if _REAPER_TASK is None:
                _run_cleanup_sync()
                return
            _update_worker_heartbeat("reaper")
            _WORKER_ACTIVITY["reaper"] = False
        return
    try:
        done = task.done()
    except Exception:
        done = True
    if not done:
        if _REAPER_TASK is None:
            _REAPER_TASK = _schedule_coroutine(_reaper_worker())
        return
    cancelled = False
    exc = None
    try:
        cancelled = task.cancelled()
        exc = task.exception()
    except Exception:
        logger.exception('unexpected error')
    if cancelled or exc is not None:
        _CLEANUP_TASK = _schedule_coroutine(_cleanup_worker())
        if _CLEANUP_TASK is None:
            _run_cleanup_sync()
            return
        _update_worker_heartbeat("cleanup")
        _WORKER_ACTIVITY["cleanup"] = False
    task = _REAPER_TASK
    if task is None:
        _REAPER_TASK = _schedule_coroutine(_reaper_worker())
        if _REAPER_TASK is None:
            _run_cleanup_sync()
            return
        _update_worker_heartbeat("reaper")
        _WORKER_ACTIVITY["reaper"] = False
        return
    try:
        done = task.done()
    except Exception:
        done = True
    if not done:
        return
    cancelled = False
    exc = None
    try:
        cancelled = task.cancelled()
        exc = task.exception()
    except Exception:
        logger.exception('unexpected error')
    if done or cancelled or exc is not None:
        _REAPER_TASK = _schedule_coroutine(_reaper_worker())
        if _REAPER_TASK is None:
            _run_cleanup_sync()
            return
        _update_worker_heartbeat("reaper")
        _WORKER_ACTIVITY["reaper"] = False


def watchdog_check() -> None:
    """Verify background workers are alive and restart if needed."""

    global _CLEANUP_TASK, _REAPER_TASK, _LAST_CLEANUP_TS, _LAST_REAPER_TS

    if _SANDBOX_DISABLE_CLEANUP:
        _log_cleanup_disabled("watchdog cycle")
        return

    if _DOCKER_CLIENT is None:
        return

    budget = globals().get("_WATCHDOG_BUDGET")
    if budget is None:
        class _NoopWatchdogBudget:
            def effective_limit(self, worker: str, baseline: float) -> float:  # pragma: no cover - shim for tests
                return baseline

            def note_restart(self, worker: str) -> None:  # pragma: no cover - shim for tests
                return

            def reset(self, worker: str) -> None:  # pragma: no cover - shim for tests
                return

        budget = _NoopWatchdogBudget()

    prev_cleanup = _CLEANUP_TASK
    prev_reaper = _REAPER_TASK
    prev_event = _EVENT_THREAD

    ensure_cleanup_worker()

    now = time.monotonic()
    margin = max(0.0, float(_CLEANUP_WATCHDOG_MARGIN))
    cleanup_limit_base = max(
        2 * _POOL_CLEANUP_INTERVAL,
        float(_CLEANUP_DURATIONS.get("cleanup", 0.0)) + margin,
        float(_CLEANUP_CURRENT_RUNTIME.get("cleanup", 0.0)) + margin,
    )
    cleanup_limit = budget.effective_limit("cleanup", cleanup_limit_base)
    reaper_limit_base = max(
        2 * _POOL_CLEANUP_INTERVAL,
        float(_CLEANUP_DURATIONS.get("reaper", 0.0)) + margin,
        float(_CLEANUP_CURRENT_RUNTIME.get("reaper", 0.0)) + margin,
    )
    reaper_limit = budget.effective_limit("reaper", reaper_limit_base)

    cleanup_elapsed = now - _LAST_CLEANUP_TS
    reaper_elapsed = now - _LAST_REAPER_TS

    if _WORKER_ACTIVITY.get("cleanup") and cleanup_elapsed > cleanup_limit:
        logger.warning("cleanup worker stalled; restarting")
        try:
            if _CLEANUP_TASK is not None:
                _CLEANUP_TASK.cancel()
        finally:
            budget.note_restart("cleanup")
            _CLEANUP_TASK = _schedule_coroutine(_cleanup_worker())
            if _CLEANUP_TASK is None:
                _run_cleanup_sync()
            else:
                _update_worker_heartbeat("cleanup")
                _WORKER_ACTIVITY["cleanup"] = False
    elif cleanup_elapsed <= cleanup_limit_base:
        budget.reset("cleanup")

    if _WORKER_ACTIVITY.get("reaper") and reaper_elapsed > reaper_limit:
        logger.warning("reaper worker stalled; restarting")
        try:
            if _REAPER_TASK is not None:
                _REAPER_TASK.cancel()
        finally:
            budget.note_restart("reaper")
            _REAPER_TASK = _schedule_coroutine(_reaper_worker())
            if _REAPER_TASK is None:
                _run_cleanup_sync()
            else:
                _update_worker_heartbeat("reaper")
                _WORKER_ACTIVITY["reaper"] = False
    elif reaper_elapsed <= reaper_limit_base:
        budget.reset("reaper")

    if prev_cleanup is not _CLEANUP_TASK:
        logger.warning("cleanup worker restarted by watchdog")
        _WATCHDOG_METRICS["cleanup"] += 1
    if prev_reaper is not _REAPER_TASK:
        logger.warning("reaper worker restarted by watchdog")
        _WATCHDOG_METRICS["reaper"] += 1
    if prev_event is not _EVENT_THREAD:
        logger.warning("event listener restarted by watchdog")
        _WATCHDOG_METRICS["event"] += 1


def schedule_cleanup_check(interval: float = _WORKER_CHECK_INTERVAL) -> None:
    """Periodically call :func:`ensure_cleanup_worker`."""

    global _WORKER_CHECK_TIMER

    if _SANDBOX_DISABLE_CLEANUP:
        _log_cleanup_disabled("cleanup watchdog scheduling")
        return

    def _loop() -> None:
        global _WORKER_CHECK_TIMER
        watchdog_check()
        timer = threading.Timer(interval, _loop)
        timer.daemon = True
        _WORKER_CHECK_TIMER = timer
        timer.start()

    _WORKER_CHECK_TIMER = threading.Timer(interval, _loop)
    _WORKER_CHECK_TIMER.daemon = True
    _WORKER_CHECK_TIMER.start()


def cancel_cleanup_check() -> None:
    """Stop periodic cleanup worker checks if scheduled."""

    global _WORKER_CHECK_TIMER
    if _WORKER_CHECK_TIMER is not None:
        _WORKER_CHECK_TIMER.cancel()
        _WORKER_CHECK_TIMER = None


import atexit

if not _SANDBOX_DISABLE_CLEANUP and (
    time.time() - _LAST_AUTOPURGE_TS >= _SANDBOX_AUTOPURGE_THRESHOLD
    or _read_active_containers()
    or _read_active_overlays()
):
    purge_leftovers()
    retry_failed_cleanup()
elif _SANDBOX_DISABLE_CLEANUP:
    _log_cleanup_disabled("startup purge")

if _DOCKER_CLIENT is not None and not _SANDBOX_DISABLE_CLEANUP:
    default_img = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
    _ensure_pool_size_async(default_img)
    if _CLEANUP_TASK is None:
        _CLEANUP_TASK = _schedule_coroutine(_cleanup_worker())
        if _CLEANUP_TASK is None:
            _run_cleanup_sync()
    if _REAPER_TASK is None:
        _REAPER_TASK = _schedule_coroutine(_reaper_worker())
        if _REAPER_TASK is None:
            _run_cleanup_sync()
    schedule_cleanup_check(_WORKER_CHECK_INTERVAL)
    atexit.register(_await_cleanup_task)
    atexit.register(_await_reaper_task)
    atexit.register(cancel_cleanup_check)
    atexit.register(stop_container_event_listener)
elif _SANDBOX_DISABLE_CLEANUP:
    _log_cleanup_disabled("cleanup worker bootstrap")

atexit.register(_release_pool_lock)
if _SANDBOX_DISABLE_CLEANUP:
    _log_cleanup_disabled("atexit cleanup")
else:
    atexit.register(_cleanup_pools)
    atexit.register(purge_leftovers)
atexit.register(stop_background_loop)


def register_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""

    def _handler(signum, frame) -> None:  # pragma: no cover - signal path
        try:
            _RADAR_MANAGER.__exit__(None, None, None)
            try:
                from .cycle import _stop_usage_worker

                _stop_usage_worker()
            except Exception:  # pragma: no cover - best effort
                logger.exception("usage worker shutdown failed")
            _cleanup_pools()
            _await_cleanup_task()
            _await_reaper_task()
        except Exception:
            logger.exception("signal cleanup failed")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except Exception:
            logger.exception("signal handler setup failed")


# ----------------------------------------------------------------------
async def _execute_in_container(
    code_str: str,
    env: Dict[str, Any],
    *,
    mounts: Dict[str, str] | None = None,
    network_disabled: bool = True,
    workdir: str | None = None,
) -> Dict[str, float]:
    """Return runtime metrics after executing ``code_str`` in a container.

    If Docker is unavailable or repeatedly fails, the snippet is executed
    locally with the same environment variables and resource limits.
    """

    snippet_path = Path(
        env.get("CONTAINER_SNIPPET_PATH", path_for_prompt(SNIPPET_NAME))
    )
    snippet_name = snippet_path.name
    snippet_dir = snippet_path.parent.as_posix()

    def _execute_locally(err_msg: str | None = None) -> Dict[str, float]:
        """Fallback local execution with basic metrics."""
        with tempfile.TemporaryDirectory(prefix="sim_local_") as td:
            path = Path(td) / snippet_name
            path.write_text(code_str, encoding="utf-8")
            stdout_path = Path(td) / "stdout.log"
            stderr_path = Path(td) / "stderr.log"

            env_vars = os.environ.copy()
            env_vars.update({k: str(v) for k, v in env.items()})

            rlimit_ok = _rlimits_supported()
            use_cgroup = False
            cgroup_path: Path | None = None
            ipr = None
            tc_idx = None
            if (
                not rlimit_ok
                and not _psutil_rlimits_supported()
                and _cgroup_v2_supported()
            ):
                cgroup_path = _create_cgroup(
                    env.get("CPU_LIMIT"), env.get("MEMORY_LIMIT")
                )
                use_cgroup = cgroup_path is not None
            ipr, tc_idx = _setup_tc_netem(env_vars)

            def _limits() -> None:
                cpu = env.get("CPU_LIMIT")
                mem = env.get("MEMORY_LIMIT")
                if rlimit_ok and resource is not None:
                    try:
                        if cpu:
                            sec = int(float(cpu)) * 10
                            resource.setrlimit(resource.RLIMIT_CPU, (sec, sec))
                    except Exception as exc:
                        logger.warning("failed to set CPU limit: %s", exc)
                    try:
                        if mem:
                            size = _parse_size(mem)
                            if size:
                                resource.setrlimit(resource.RLIMIT_AS, (size, size))
                    except Exception as exc:
                        logger.warning("failed to set memory limit: %s", exc)
                if use_cgroup and cgroup_path is not None:
                    try:
                        with open(
                            cgroup_path / "cgroup.procs", "w", encoding="utf-8"
                        ) as fh:
                            fh.write(str(os.getpid()))
                    except Exception as exc:
                        logger.warning("failed to join cgroup: %s", exc)

            start = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
            net_start = psutil.net_io_counters() if psutil else None
            try:
                proc = subprocess.Popen(
                    ["python", str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env_vars,
                    cwd=workdir or td,
                    preexec_fn=_limits if (rlimit_ok or use_cgroup) else None,
                )
                p = psutil.Process(proc.pid) if psutil else None
                if p and not rlimit_ok:
                    _apply_psutil_rlimits(
                        p, env.get("CPU_LIMIT"), env.get("MEMORY_LIMIT")
                    )
                out, err = proc.communicate(timeout=int(env.get("TIMEOUT", "30")))
                stdout_path.write_text(out, encoding="utf-8")
                stderr_path.write_text(err, encoding="utf-8")
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("timeout", encoding="utf-8")
                exit_code = -1
            end = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
            net_end = psutil.net_io_counters() if psutil else None
            if cgroup_path is not None:
                _cleanup_cgroup(cgroup_path)
            _cleanup_tc(ipr, tc_idx)

            if p:
                try:
                    cpu_total = p.cpu_times().user + p.cpu_times().system
                    mem_usage = p.memory_info().rss
                    io = p.io_counters()
                    disk_io = io.read_bytes + io.write_bytes
                except Exception:
                    cpu_total = 0.0
                    mem_usage = 0.0
                    disk_io = 0.0
            else:
                if start is not None and end is not None:
                    cpu_total = (end.ru_utime + end.ru_stime) - (
                        start.ru_utime + start.ru_stime
                    )
                    mem_usage = float(end.ru_maxrss - start.ru_maxrss) * 1024
                    disk_io = (
                        float(
                            (end.ru_inblock - start.ru_inblock)
                            + (end.ru_oublock - start.ru_oublock)
                        )
                        * 512
                    )
                else:
                    cpu_total = 0.0
                    mem_usage = 0.0
                    disk_io = 0.0

            if net_start and net_end:
                net_io = float(
                    (net_end.bytes_recv - net_start.bytes_recv)
                    + (net_end.bytes_sent - net_start.bytes_sent)
                )
            else:
                net_io = 0.0

            gpu_usage = 0.0
            try:
                import GPUtil  # type: ignore

                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = sum(g.load for g in gpus) / len(gpus)
            except Exception:
                gpu_usage = 0.0

            result = {
                "exit_code": float(exit_code),
                "cpu": float(cpu_total),
                "memory": float(mem_usage),
                "disk_io": float(disk_io),
                "net_io": float(net_io),
                "gpu_usage": float(gpu_usage),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
            if err_msg:
                result["container_error"] = str(err_msg)
            return result

    error_msg = ""
    try:
        import docker  # type: ignore
        from docker.errors import DockerException, APIError
    except Exception as exc:  # pragma: no cover - optional dependency
        error_msg = str(exc)
        logger.warning("docker import failed: %s", exc)
        return _execute_locally(error_msg)

    try:
        client = _DOCKER_CLIENT or docker.from_env()
    except DockerException as exc:
        error_msg = str(exc)
        logger.error("failed to create docker client: %s", exc)
        return _execute_locally(error_msg)
    if _DOCKER_CLIENT is None:
        _globals = globals()
        _globals["_DOCKER_CLIENT"] = client

    def _run_ephemeral() -> Dict[str, float]:
        """Run snippet using a one-off container (legacy behaviour)."""
        nonlocal client, error_msg
        attempt = 0
        delay = _CREATE_BACKOFF_BASE
        while attempt < _CREATE_RETRY_LIMIT:
            try:
                with tempfile.TemporaryDirectory(prefix="sim_cont_") as td:
                    path = Path(td) / snippet_name
                    path.write_text(code_str, encoding="utf-8")

                    image = env.get("CONTAINER_IMAGE")
                    if not image:
                        image = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
                        os_type = env.get("OS_TYPE")
                        if os_type:
                            image = os.getenv(
                                f"SANDBOX_CONTAINER_IMAGE_{os_type.upper()}", image
                            )

                    volumes = {td: {"bind": snippet_dir, "mode": "rw"}}
                    if mounts:
                        for host, dest in mounts.items():
                            volumes[host] = {"bind": dest, "mode": "rw"}

                    kwargs: Dict[str, Any] = {
                        "volumes": volumes,
                        "environment": {k: str(v) for k, v in env.items()},
                        "detach": True,
                        "network_disabled": network_disabled,
                    }

                    mem = env.get("MEMORY_LIMIT")
                    if mem:
                        kwargs["mem_limit"] = str(mem)

                    cpu = env.get("CPU_LIMIT")
                    if cpu:
                        try:
                            kwargs["cpu_quota"] = int(float(cpu) * 100000)
                        except Exception as exc:
                            logger.warning("invalid CPU limit %s: %s", cpu, exc)

                    disk = env.get("DISK_LIMIT")
                    if disk:
                        kwargs["storage_opt"] = {"size": str(disk)}

                    user = os.getenv("SANDBOX_CONTAINER_USER")
                    if user:
                        kwargs["user"] = user

                    gpu = env.get("GPU_LIMIT")
                    if gpu:
                        try:
                            from docker.types import DeviceRequest

                            kwargs["device_requests"] = [
                                DeviceRequest(
                                    count=int(float(gpu)), capabilities=[["gpu"]]
                                )
                            ]
                        except Exception as exc:
                            logger.warning("GPU limit ignored: %s", exc)

                    if workdir:
                        kwargs["working_dir"] = workdir

                    kwargs["labels"] = {_POOL_LABEL: "1"}

                    container = client.containers.run(
                        image,
                        ["python", snippet_path.as_posix()],
                        **kwargs,
                    )
                    _register_container_finalizer(container)
                    _record_active_container(container.id)

                    timeout = int(env.get("TIMEOUT", 300))
                    try:
                        result = container.wait(timeout=timeout)
                    except (docker.errors.APIError, subprocess.TimeoutExpired):
                        try:
                            container.kill()
                        except Exception:
                            logger.exception("container kill failed")
                        result = {"StatusCode": -1}
                    stdout_path = Path(td) / "stdout.log"
                    stderr_path = Path(td) / "stderr.log"
                    try:
                        out = container.logs(stdout=True, stderr=False)
                        err = container.logs(stdout=False, stderr=True)
                        stdout_path.write_bytes(out or b"")
                        stderr_path.write_bytes(err or b"")
                    except Exception:
                        stdout_path.write_text("", encoding="utf-8")
                        stderr_path.write_text("", encoding="utf-8")
                    stats = container.stats(stream=False)
                    container.remove()
                    _remove_active_container(container.id)

                    blk = stats.get("blkio_stats", {}).get(
                        "io_service_bytes_recursive", []
                    )
                    disk_io = float(sum(x.get("value", 0) for x in blk))

                    cpu_total = float(
                        stats.get("cpu_stats", {})
                        .get("cpu_usage", {})
                        .get("total_usage", 0)
                    )
                    mem_usage = float(stats.get("memory_stats", {}).get("max_usage", 0))

                    net = stats.get("networks", {})
                    net_io = float(
                        sum(
                            v.get("rx_bytes", 0) + v.get("tx_bytes", 0)
                            for v in net.values()
                        )
                    )

                    gstats = (
                        stats.get("gpu_stats") or stats.get("accelerator_stats") or []
                    )
                    gpu_usage = 0.0
                    if isinstance(gstats, list) and gstats:
                        gpu_usage = float(
                            sum(
                                float(
                                    g.get("utilization_gpu", 0)
                                    or g.get("gpu_utilization", 0)
                                )
                                for g in gstats
                            )
                        )

                    if attempt:
                        _log_diagnostic("container_failure", True)

                    return {
                        "exit_code": float(result.get("StatusCode", 0)),
                        "cpu": cpu_total,
                        "memory": mem_usage,
                        "disk_io": disk_io,
                        "net_io": net_io,
                        "gpu_usage": gpu_usage,
                        "stdout_log": str(stdout_path),
                        "stderr_log": str(stderr_path),
                    }
            except Exception as exc:  # pragma: no cover - runtime failures
                error_msg = str(exc)
                if _ERROR_CONTEXT_BUILDER is not None:
                    record_error(exc, context_builder=_ERROR_CONTEXT_BUILDER)
                else:
                    logger.exception("container execution failed", exc_info=exc)
                _log_diagnostic(error_msg, False)
                _CREATE_FAILURES[image] += 1
                fails = _CONSECUTIVE_CREATE_FAILURES.get(image, 0) + 1
                _CONSECUTIVE_CREATE_FAILURES[image] = fails
                if attempt >= _CREATE_RETRY_LIMIT - 1:
                    logger.warning(
                        "docker repeatedly failed; falling back to local execution"
                    )
                    _log_diagnostic("local_fallback", True)
                    return _execute_locally(error_msg)
                attempt += 1
                time.sleep(min(60.0, delay * (2**fails)))
        return _execute_locally(error_msg)

    # use legacy container mode when advanced features are requested
    if (
        mounts
        or not network_disabled
        or workdir
        or any(
            k in env for k in ("CPU_LIMIT", "MEMORY_LIMIT", "DISK_LIMIT", "GPU_LIMIT")
        )
    ):
        return _run_ephemeral()

    # pooled execution path
    attempt = 0
    delay = _CREATE_BACKOFF_BASE
    while attempt < _CREATE_RETRY_LIMIT:
        try:
            image = env.get("CONTAINER_IMAGE")
            if not image:
                image = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
                os_type = env.get("OS_TYPE")
                if os_type:
                    image = os.getenv(
                        f"SANDBOX_CONTAINER_IMAGE_{os_type.upper()}", image
                    )

            async with pooled_container(image) as (container, td):
                path = Path(td) / snippet_name
                path.write_text(code_str, encoding="utf-8")

                timeout = int(env.get("TIMEOUT", 300))
                try:
                    result = container.exec_run(
                        ["python", snippet_path.as_posix()],
                        environment={k: str(v) for k, v in env.items()},
                        workdir=workdir,
                        demux=True,
                        timeout=timeout,
                    )
                except (docker.errors.APIError, subprocess.TimeoutExpired):
                    try:
                        container.kill()
                    except Exception:
                        logger.exception("container kill failed")
                    result = type(
                        "ExecResult", (), {"exit_code": -1, "output": (b"", b"")}
                    )()

                stdout_path = Path(td) / "stdout.log"
                stderr_path = Path(td) / "stderr.log"
                try:
                    out, err = result.output or (b"", b"")
                    stdout_path.write_bytes(out or b"")
                    stderr_path.write_bytes(err or b"")
                except Exception:
                    stdout_path.write_text("", encoding="utf-8")
                    stderr_path.write_text("", encoding="utf-8")

                stats = container.stats(stream=False)

            blk = stats.get("blkio_stats", {}).get("io_service_bytes_recursive", [])
            disk_io = float(sum(x.get("value", 0) for x in blk))

            cpu_total = float(
                stats.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
            )
            mem_usage = float(stats.get("memory_stats", {}).get("max_usage", 0))

            net = stats.get("networks", {})
            net_io = float(
                sum(v.get("rx_bytes", 0) + v.get("tx_bytes", 0) for v in net.values())
            )

            gstats = stats.get("gpu_stats") or stats.get("accelerator_stats") or []
            gpu_usage = 0.0
            if isinstance(gstats, list) and gstats:
                gpu_usage = float(
                    sum(
                        float(
                            g.get("utilization_gpu", 0) or g.get("gpu_utilization", 0)
                        )
                        for g in gstats
                    )
                )

            if attempt:
                _log_diagnostic("container_failure", True)
            _CONSECUTIVE_CREATE_FAILURES[image] = 0

            exit_code = getattr(result, "exit_code", 0)
            if isinstance(result, tuple):
                exit_code = result[0]

            return {
                "exit_code": float(exit_code),
                "cpu": cpu_total,
                "memory": mem_usage,
                "disk_io": disk_io,
                "net_io": net_io,
                "gpu_usage": gpu_usage,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        except Exception as exc:  # pragma: no cover - runtime failures
            error_msg = str(exc)
            if _ERROR_CONTEXT_BUILDER is not None:
                record_error(exc, context_builder=_ERROR_CONTEXT_BUILDER)
            else:
                logger.exception("container execution failed", exc_info=exc)
            _log_diagnostic(error_msg, False)
            _CREATE_FAILURES[image] += 1
            fails = _CONSECUTIVE_CREATE_FAILURES.get(image, 0) + 1
            _CONSECUTIVE_CREATE_FAILURES[image] = fails
            if attempt >= _CREATE_RETRY_LIMIT - 1:
                logger.warning(
                    "docker repeatedly failed; falling back to local execution"
                )
                _log_diagnostic("local_fallback", True)
                return _execute_locally(error_msg)
            attempt += 1
            time.sleep(min(60.0, delay * (2**fails)))
    return _execute_locally(error_msg)


# ----------------------------------------------------------------------
def simulate_execution_environment(
    code_str: str,
    input_stub: Dict[str, Any] | None = None,
    *,
    container: bool | None = None,
) -> Dict[str, Any]:
    """Mock runtime environment and optionally execute code in a container."""

    logger.debug(
        "simulate_execution_environment called with input stub=%s container=%s",
        bool(input_stub),
        container,
    )

    ensure_cleanup_worker()

    analysis = static_behavior_analysis(code_str)
    env_result = {
        "functions_called": analysis.get("calls", []),
        "files_accessed": analysis.get("files_written", []),
        "risk_flags_triggered": analysis.get("flags", []),
    }

    if input_stub:
        env_result["input_stub"] = input_stub

    if analysis.get("regex_flags"):
        env_result["risk_flags_triggered"].extend(analysis["regex_flags"])

    if container is None:
        container = str(os.getenv("SANDBOX_DOCKER", "0")).lower() not in {
            "0",
            "false",
            "no",
            "",
        }

    runtime_metrics: Dict[str, float] = {}
    if container:
        try:
            runtime_metrics = asyncio.run(
                _execute_in_container(code_str, input_stub or {})
            )
        except Exception as exc:  # pragma: no cover - best effort
            if _ERROR_CONTEXT_BUILDER is not None:
                record_error(exc, context_builder=_ERROR_CONTEXT_BUILDER)
            else:
                logger.exception(
                    "container execution failed", exc_info=exc
                )

    if runtime_metrics:
        env_result["runtime_metrics"] = runtime_metrics

    logger.debug("environment simulation result: %s", env_result)
    return env_result


# ----------------------------------------------------------------------
def generate_sandbox_report(analysis_result: Dict[str, Any], output_path: str) -> None:
    """Write ``analysis_result`` to ``output_path`` as JSON with timestamp."""
    logger.debug("writing sandbox report to %s", path_for_prompt(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = dict(analysis_result)
    data["timestamp"] = datetime.utcnow().isoformat()
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    logger.debug("sandbox report written: %s", path_for_prompt(output_path))


# ----------------------------------------------------------------------
def _parse_size(value: str | int | float) -> int:
    """Return ``value`` interpreted as bytes."""
    try:
        s = str(value).strip().lower()
        if s.endswith("mi"):
            return int(float(s[:-2])) * 1024 * 1024
        if s.endswith("gi"):
            return int(float(s[:-2])) * 1024 * 1024 * 1024
        return int(float(s))
    except Exception:
        return 0


def _parse_bandwidth(value: str | int | float) -> int:
    """Return ``value`` interpreted as bits per second."""
    try:
        s = str(value).strip().lower()
        if s.endswith("kbps"):
            return int(float(s[:-4]) * 1_000)
        if s.endswith("mbps"):
            return int(float(s[:-4]) * 1_000_000)
        if s.endswith("gbps"):
            return int(float(s[:-4]) * 1_000_000_000)
        return int(float(s))
    except Exception:
        return 0


def _parse_timespan(value: str | int | float) -> float:
    """Return ``value`` interpreted as seconds."""
    try:
        s = str(value).strip().lower()
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) * 60.0
        if s.endswith("h"):
            return float(s[:-1]) * 3600.0
        if s.endswith("d"):
            return float(s[:-1]) * 86400.0
        if s.endswith("w"):
            return float(s[:-1]) * 604800.0
        return float(s)
    except Exception:
        return 0.0


def validate_preset(preset: Dict[str, Any]) -> bool:
    """Return ``True`` if numeric values in ``preset`` are sane."""
    try:
        cpu = preset.get("CPU_LIMIT")
        if cpu is not None:
            val = float(cpu)
            if val <= 0 or val > 64:
                raise ValueError("CPU_LIMIT out of range")

        mem = preset.get("MEMORY_LIMIT")
        if mem is not None:
            size = _parse_size(mem)
            if size <= 0 or size > 64 * 1024 * 1024 * 1024:
                raise ValueError("MEMORY_LIMIT out of range")

        for key in ("BANDWIDTH_LIMIT", "MIN_BANDWIDTH", "MAX_BANDWIDTH"):
            bw = preset.get(key)
            if bw is not None:
                bw_val = _parse_bandwidth(bw)
                if bw_val <= 0 or bw_val > 10_000_000_000:
                    raise ValueError(f"{key} out of range")

        return True
    except Exception as exc:
        logger.warning("invalid preset skipped: %s", exc)
        return False


_CONTAINER_DISK_LIMIT = _parse_size(_CONTAINER_DISK_LIMIT_STR)

# finalise overlay age setting now that helper is defined
_OVERLAY_MAX_AGE = _parse_timespan(os.getenv("SANDBOX_OVERLAY_MAX_AGE", "7d"))

_FAILED_CLEANUP_ALERT_AGE = _parse_timespan(
    os.getenv("SANDBOX_FAILED_CLEANUP_AGE", "1d")
)

_SANDBOX_AUTOPURGE_THRESHOLD = _parse_timespan(
    os.getenv("SANDBOX_AUTOPURGE_THRESHOLD", "24h")
)

_LAST_AUTOPURGE_TS = _read_last_autopurge()


def _rlimits_supported() -> bool:
    """Return ``True`` if ``resource`` limits appear usable."""
    if resource is None:
        return False
    try:
        resource.getrlimit(resource.RLIMIT_CPU)
        resource.getrlimit(resource.RLIMIT_AS)
        return True
    except Exception:
        return False


def _psutil_rlimits_supported() -> bool:
    """Return ``True`` if psutil can set rlimits."""
    if psutil is None:
        return False
    try:
        p = psutil.Process()
        return hasattr(p, "rlimit")
    except Exception:
        return False


_CGROUP_BASE = Path("/sys/fs/cgroup")


def _cgroup_v2_supported() -> bool:
    """Return ``True`` if cgroup v2 is available."""
    return (_CGROUP_BASE / "cgroup.controllers").exists()


def _create_cgroup(cpu: Any, mem: Any) -> Path | None:
    """Create a cgroup with optional CPU and memory limits."""
    path = _CGROUP_BASE / f"sandbox_{os.getpid()}_{random.randint(0, 9999)}"
    try:
        path.mkdir()
        if cpu:
            try:
                quota = int(float(cpu) * 100000)
                (path / "cpu.max").write_text(f"{quota} 100000")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("failed to set cgroup cpu limit: %s", exc)
        if mem:
            size = _parse_size(mem)
            if size:
                try:
                    (path / "memory.max").write_text(str(size))
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("failed to set cgroup memory limit: %s", exc)
        return path
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.warning("failed to create cgroup: %s", exc)
        try:
            if path.exists():
                for f in path.iterdir():
                    try:
                        f.unlink()
                    except Exception as exc2:
                        logger.exception("failed to remove cgroup file %s", f)
                path.rmdir()
        except Exception as exc2:
            logger.exception(
                "failed to remove cgroup directory %s", path_for_prompt(path)
            )
        return None


def _cleanup_cgroup(path: Path) -> None:
    """Remove ``path`` and contained files."""
    try:
        if path.exists():
            for f in path.iterdir():
                try:
                    f.unlink()
                except Exception:
                    logger.exception("failed to remove cgroup file %s", f)
            path.rmdir()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("failed to remove cgroup: %s", exc)


def _apply_psutil_rlimits(proc: Any, cpu: Any, mem: Any) -> None:
    """Apply CPU and memory limits to ``proc`` using psutil."""
    if psutil is None or not hasattr(proc, "rlimit"):
        return
    try:
        if cpu:
            sec = int(float(cpu)) * 10
            proc.rlimit(getattr(psutil, "RLIMIT_CPU", 0), (sec, sec))
    except Exception as exc:
        logger.warning("failed to set CPU limit via psutil: %s", exc)
    try:
        if mem:
            size = _parse_size(mem)
            if size:
                proc.rlimit(getattr(psutil, "RLIMIT_AS", 9), (size, size))
    except Exception as exc:
        logger.warning("failed to set memory limit via psutil: %s", exc)


def _setup_tc_netem(env: Dict[str, Any]) -> tuple[Any | None, int | None]:
    """Configure ``tc netem`` qdisc using pyroute2 based on environment vars."""
    if IPRoute is None:
        return None, None
    latency = env.get("NETWORK_LATENCY_MS")
    jitter = env.get("NETWORK_JITTER_MS")
    loss = env.get("PACKET_LOSS")
    dup = env.get("PACKET_DUPLICATION")
    if not any((latency, jitter, loss, dup)):
        return None, None
    ipr = None
    try:
        ipr = IPRoute()
        idx = ipr.link_lookup(ifname="lo")[0]
        kwargs = {}
        if latency or jitter:
            parts = []
            if latency:
                parts.append(f"{latency}ms")
            else:
                parts.append("0ms")
            if jitter:
                parts.append(f"{jitter}ms")
            kwargs["delay"] = " ".join(parts)
        if loss:
            kwargs["loss"] = float(loss)
        if dup:
            kwargs["duplicate"] = float(dup)
        ipr.tc("add", "root", idx, "netem", **kwargs)
        return ipr, idx
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("failed to apply netem: %s", exc)
        if ipr is not None:
            try:
                ipr.close()
            except Exception:
                logger.exception("failed to close IPRoute")
        return None, None


def _cleanup_tc(ipr: Any | None, idx: int | None) -> None:
    """Remove ``tc netem`` qdisc."""
    if ipr is None or idx is None:
        return
    try:
        ipr.tc("del", "root", idx, "netem")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("failed to remove netem: %s", exc)
    finally:
        try:
            ipr.close()
        except Exception:
            logger.exception("failed to close IPRoute")


def _parse_failure_modes(value: Any) -> set[str]:
    """Return a normalized set of failure modes from ``value``.

    The helper accepts strings, iterables or mixed inputs and is agnostic to
    the actual mode names so new modes such as ``hostile_input`` are supported
    automatically.
    """
    if not value:
        return set()
    modes: set[str] = set()
    if isinstance(value, str):
        for part in value.split(","):
            part = part.strip()
            if part:
                modes.add(part)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            modes.update(_parse_failure_modes(item))
    else:
        modes.add(str(value))
    return modes


def _inject_failure_modes(snippet: str, modes: set[str]) -> str:
    """Return ``snippet`` with failure mode hooks prepended."""

    parts: list[str] = []
    if "disk" in modes or "disk_corruption" in modes:
        corruption = ""
        if "disk_corruption" in modes:
            corruption = (
                "            if isinstance(data, bytes):\n"
                "                data = b'CORRUPTED' + data\n"
                "            else:\n"
                "                data = 'CORRUPTED' + data\n"
            )
        delay = "            time.sleep(0.05)\n" if "disk" in modes else ""
        parts.append(
            "import builtins, time\n"
            "_orig_open = builtins.open\n"
            'def _open(f, mode="r", *a, **k):\n'
            "    file = _orig_open(f, mode, *a, **k)\n"
            '    if "w" in mode:\n'
            "        orig = file.write\n"
            "        def _write(data, *aa, **kk):\n"
            f"{delay}"
            f"{corruption}"
            "            return orig(data, *aa, **kk)\n"
            "        file.write = _write\n"
            "    return file\n"
            "builtins.open = _open\n"
        )

    if "network" in modes or "network_partition" in modes:
        parts.append(
            "import socket\n"
            "class _BlockSocket(socket.socket):\n"
            "    def connect(self, *a, **k):\n"
            "        raise OSError('network blocked')\n"
            "socket.socket = _BlockSocket\n"
        )

    if "cpu_spike" in modes:
        parts.append(
            "import threading, time\n"
            "def _burn():\n"
            "    end = time.time() + 0.2\n"
            "    while time.time() < end:\n"
            "        pass\n"
            "threading.Thread(target=_burn, daemon=True).start()\n"
        )

    if "concurrency_spike" in modes:
        parts.append(
            "import os, threading, asyncio, time, json, sys, traceback\n"
            "def _run():\n"
            "    n_t=int(os.getenv('THREAD_BURST','50'))\n"
            "    n_a=int(os.getenv('ASYNC_TASK_BURST','50'))\n"
            "    def _burn():\n"
            "        time.sleep(0.05)\n"
            "    threads=[threading.Thread(target=_burn) for _ in range(n_t)]\n"
            "    [t.start() for t in threads]\n"
            "    async def _burst():\n"
            "        async def _noop():\n"
            "            await asyncio.sleep(0.01)\n"
            "        await asyncio.gather(*[asyncio.create_task(_noop()) for _ in range(n_a)])\n"
            "    try:\n"
            "        asyncio.run(_burst())\n"
            "    except Exception as exc:\n"
            "        print(f'concurrency spike burst failed: {exc}', file=sys.stderr)\n"
            "        traceback.print_exc()\n"
            "        raise\n"
            "    path=os.getenv('SANDBOX_CONCURRENCY_OUT')\n"
            "    if path:\n"
            "        try:\n"
            "            with open(path,'w') as fh:\n"
            "                json.dump({'threads':len(threads),'tasks':n_a},fh)\n"
            "        except Exception as exc:\n"
            "            print(f'concurrency spike write failed: {exc}', file=sys.stderr)\n"
            "            traceback.print_exc()\n"
            "            raise\n"
            "_run()\n"
        )

    if "memory" in modes:
        parts.append(" _mem_fail = bytearray(10_000_000)\n")

    if "timeout" in modes:
        parts.append(
            "import threading, os, time\n"
            "def _abort():\n"
            "    time.sleep(0.05)\n"
            "    os._exit(1)\n"
            "threading.Thread(target=_abort, daemon=True).start()\n"
        )

    if "hostile_input" in modes:
        parts.append(
            "import os, json\n"
            "_payloads=[\"' OR '1'='1\", \"<script>alert(1)</script>\", \"A\"*10000]\n"
            "_env=os.environ\n"
            "_stubs=[]\n"
            "for i, k in enumerate(list(_env)):\n"
            "    if not k.isupper():\n"
            "        val=_payloads[i % len(_payloads)]\n"
            "        _env[k]=val\n"
            "        _stubs.append({k: val})\n"
            "if _stubs:\n"
            "    _env['SANDBOX_INPUT_STUBS']=json.dumps(_stubs)\n"
        )

    if "user_misuse" in modes:
        parts.append(
            "import sys, os\n"
            "def _misuse():\n"
            "    try:\n"
            "        len('x', 'y')\n"
            "    except Exception as exc:\n"
            "        print(exc, file=sys.stderr)\n"
            "    try:\n"
            "        open('/root/forbidden', 'r')\n"
            "    except Exception as exc:\n"
            "        print(exc, file=sys.stderr)\n"
            "_misuse()\n"
        )

    if not parts:
        return snippet

    return "\n".join(parts) + "\n" + snippet


async def _section_worker(
    snippet: str,
    env_input: Dict[str, Any],
    threshold: float,
    runner_config: Dict[str, Any] | None = None,
) -> tuple[
    Dict[str, Any],
    list[tuple[float, float, Dict[str, float]]],
    Dict[str, List[str]],
]:
    """Execute ``snippet`` with resource limits and return results."""

    if env_input:
        try:
            _get_history_db().add(env_input)
        except Exception:
            logger.exception("failed to record input history")

    cov_map: Dict[str, List[str]] = {}
    cov = coverage.Coverage(data_file=None) if coverage else None
    if cov:
        cov.start()

    def _run_snippet() -> Dict[str, Any]:
        _reset_runtime_state()
        with tempfile.TemporaryDirectory(prefix="run_") as td:
            path = Path(td) / path_for_prompt(SNIPPET_NAME)
            modes = _parse_failure_modes(env_input.get("FAILURE_MODES"))
            snip = _inject_failure_modes(snippet, modes)
            path.write_text(snip, encoding="utf-8")
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_input.items()})
            try:
                rc = dict(runner_config or {})
                rc.setdefault("safe_mode", True)
                rc.setdefault("use_subprocess", True)
                if env_input.get("INJECT_EDGE_CASES"):
                    rc["inject_edge_cases"] = True
                profiles = get_edge_case_profiles()
                if profiles:
                    stubs: Dict[str, Any] = {}
                    for prof in profiles:
                        stubs.update(prof)
                    td = dict(rc.get("test_data") or {})
                    stubs.update(td)
                    rc["test_data"] = stubs
                    existing = list(rc.get("edge_case_profiles", []))
                    existing.extend(profiles)
                    rc["edge_case_profiles"] = existing
                runner = WorkflowSandboxRunner()
                metrics = runner.run(lambda: exec(snip, {}), **rc)
                if _SandboxMetaLogger:
                    try:
                        data_dir = Path(resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")))
                        meta = _SandboxMetaLogger(data_dir / "sandbox_meta.log")
                        succ = sum(1 for m in metrics.modules if m.success)
                        fail = sum(1 for m in metrics.modules if not m.success)
                        dur = sum(m.duration for m in metrics.modules)
                        meta.log_cycle(
                            cycle=len(meta.records),
                            roi=0.0,
                            modules=[m.name for m in metrics.modules],
                            reason="environment_run",
                            module_metrics=metrics.modules,
                            successes=succ,
                            failures=fail,
                            duration=dur,
                        )
                    except Exception:
                        logger.warning("failed to log sandbox metadata", exc_info=True)
            except Exception:
                logger.exception('unexpected error')
            conc_path = Path(td) / "concurrency.json"
            env["SANDBOX_CONCURRENCY_OUT"] = str(conc_path)
            if "memory" in modes and "MEMORY_LIMIT" not in env_input:
                env["MEMORY_LIMIT"] = "32Mi"
                env_input["MEMORY_LIMIT"] = "32Mi"
            if "cpu_spike" in modes and "CPU_LIMIT" not in env_input:
                env["CPU_LIMIT"] = "0.1"
                env_input["CPU_LIMIT"] = "0.1"

            netem_args = []
            latency = env_input.get("NETWORK_LATENCY_MS")
            if latency:
                netem_args += ["delay", f"{latency}ms"]
                jitter = env_input.get("NETWORK_JITTER_MS")
                if jitter:
                    netem_args.append(f"{jitter}ms")
            loss = env_input.get("PACKET_LOSS")
            if loss:
                netem_args += ["loss", f"{loss}%"]
            dup = env_input.get("PACKET_DUPLICATION")
            if dup:
                netem_args += ["duplicate", f"{dup}%"]

            netem_metrics = {
                "netem_latency_ms": float(latency or 0),
                "netem_jitter_ms": float(env_input.get("NETWORK_JITTER_MS", 0) or 0),
                "netem_packet_loss": float(loss or 0),
                "netem_packet_duplication": float(dup or 0),
            }

            def _load_conc() -> Dict[str, float]:
                try:
                    with open(conc_path) as fh:
                        data = json.load(fh)
                    return {
                        f"concurrency_{k}": float(v) for k, v in data.items()
                    }
                except Exception:
                    return {}

            if {"network", "network_partition"} & modes:
                if "loss" not in netem_args:
                    netem_args += ["loss", "100%"]
            _use_netem = False
            _ns_name = None
            _ipr = None
            if (
                netem_args
                and IPRoute is not None
                and NSPopen is not None
                and netns is not None
            ):
                try:
                    _ns_name = (
                        f"sandbox_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
                    )
                    netns.create(_ns_name)
                    _ipr = IPRoute()
                    _ipr.bind(netns=_ns_name)
                    idx = _ipr.link_lookup(ifname="lo")[0]
                    _ipr.link("set", index=idx, state="up")
                    kwargs = {}
                    if latency:
                        if jitter:
                            kwargs["delay"] = f"{latency}ms {jitter}ms"
                        else:
                            kwargs["delay"] = f"{latency}ms"
                    if loss:
                        kwargs["loss"] = float(loss)
                    if dup:
                        kwargs["duplicate"] = float(dup)
                    _ipr.tc("add", "root", idx, "netem", **kwargs)
                    _use_netem = True
                except Exception as exc:
                    logger.warning("pyroute2 netem setup failed: %s", exc)
                    try:
                        if _ipr:
                            _ipr.close()
                    finally:
                        _ipr = None
                        if _ns_name:
                            try:
                                netns.remove(_ns_name)
                            except Exception:
                                logger.exception("failed to remove netns %s", _ns_name)
                            _ns_name = None
                    _use_netem = False
            elif netem_args:
                try:
                    code = (
                        "from pyroute2 import IPRoute, netns, NSPopen\n"
                        "ns='sn'\n"
                        "netns.create(ns)\n"
                        "ipr=IPRoute()\n"
                        "ipr.bind(netns=ns)\n"
                        "idx=ipr.link_lookup(ifname='eth0')[0]\n"
                        "ipr.link('set', index=idx, state='up')\n"
                        f"ipr.tc('add','root',idx,'netem',"
                        f"delay='{latency}ms{' '+str(jitter)+'ms' if env_input.get('NETWORK_JITTER_MS') else ''}',"
                        f"loss={float(loss) if loss else 0},"
                        f"duplicate={float(dup) if dup else 0})\n"
                        "try:\n"
                    )
                    code += textwrap.indent(snip, "    ")
                    code += (
                        "\nfinally:\n"
                        "    ipr.tc('del','root',idx,'netem')\n"
                        "    ipr.close()\n"
                        "    netns.remove(ns)\n"
                    )
                    metrics = asyncio.run(
                        _execute_in_container(
                            code,
                            env_input,
                            network_disabled={
                                "network" in modes or "network_partition" in modes
                            },
                        )
                    )
                    metrics.update(netem_metrics)
                    return {
                        "stdout": "",
                        "stderr": "",
                        "exit_code": int(metrics.get("exit_code", 0)),
                        **netem_metrics,
                        **_load_conc(),
                    }
                except Exception as exc:
                    logger.warning("pyroute2 docker netem failed: %s", exc)

            rlimit_ok = _rlimits_supported()

            def _limits() -> None:
                if not rlimit_ok or resource is None:
                    return
                cpu = env_input.get("CPU_LIMIT")
                mem = env_input.get("MEMORY_LIMIT")
                try:
                    if cpu:
                        sec = int(float(cpu)) * 10
                        resource.setrlimit(resource.RLIMIT_CPU, (sec, sec))
                except Exception as exc:
                    logger.warning("failed to set CPU limit: %s", exc)
                try:
                    if mem:
                        size = _parse_size(mem)
                        if size:
                            resource.setrlimit(resource.RLIMIT_AS, (size, size))
                except Exception as exc:
                    logger.warning("failed to set memory limit: %s", exc)

            def _run_psutil() -> Dict[str, Any]:
                args = ["python", str(path)]
                if _use_netem and NSPopen is not None and _ns_name is not None:
                    proc = NSPopen(
                        _ns_name,
                        args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                    )
                else:
                    proc = subprocess.Popen(
                        args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                    )
                p = psutil.Process(proc.pid) if psutil else None
                if p:
                    _apply_psutil_rlimits(
                        p, env_input.get("CPU_LIMIT"), env_input.get("MEMORY_LIMIT")
                    )
                net_start = psutil.net_io_counters() if psutil else None
                cpu_lim = (
                    int(float(env_input.get("CPU_LIMIT", 0))) * 10
                    if env_input.get("CPU_LIMIT")
                    else None
                )
                mem_lim = (
                    _parse_size(env_input.get("MEMORY_LIMIT", 0))
                    if env_input.get("MEMORY_LIMIT")
                    else None
                )
                timeout = int(env_input.get("TIMEOUT", "30"))
                start = time.monotonic()
                reason = ""
                while proc.poll() is None:
                    if time.monotonic() - start > timeout:
                        reason = "timeout"
                        proc.kill()
                        break
                    try:
                        if p is not None:
                            times = p.cpu_times()
                            cpu = times.user + times.system
                            mem = p.memory_info().rss
                            if cpu_lim and cpu > cpu_lim:
                                reason = "cpu"
                                proc.kill()
                                break
                            if mem_lim and mem > mem_lim:
                                reason = "memory"
                                proc.kill()
                                break
                    except Exception:
                        logger.warning(
                            "psutil metrics collection failed", exc_info=True
                        )
                    time.sleep(0.1)
                out, err = proc.communicate()
                net_end = psutil.net_io_counters() if psutil else None
                if p is not None:
                    try:
                        cpu_total = p.cpu_times().user + p.cpu_times().system
                        mem_usage = p.memory_info().rss
                        io = p.io_counters()
                        disk_io = io.read_bytes + io.write_bytes
                    except Exception:
                        cpu_total = mem_usage = disk_io = 0.0
                else:
                    cpu_total = mem_usage = disk_io = 0.0

                if net_start and net_end:
                    net_io = float(
                        (net_end.bytes_recv - net_start.bytes_recv)
                        + (net_end.bytes_sent - net_start.bytes_sent)
                    )
                else:
                    net_io = 0.0

                gpu_usage = 0.0
                try:
                    import GPUtil  # type: ignore

                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = sum(g.load for g in gpus) / len(gpus)
                except Exception:
                    gpu_usage = 0.0
                if reason == "timeout":
                    err = err or "timeout"
                    code = -1
                elif reason:
                    code = -1
                else:
                    code = proc.returncode
                return {
                    "stdout": out,
                    "stderr": err,
                    "exit_code": code,
                    "cpu": cpu_total,
                    "memory": mem_usage,
                    "disk_io": disk_io,
                    "net_io": net_io,
                    "gpu_usage": gpu_usage,
                    **_load_conc(),
                }

            try:
                if rlimit_ok:
                    proc = subprocess.run(
                        ["python", str(path)],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=int(env_input.get("TIMEOUT", "30")),
                        preexec_fn=_limits,
                    )
                    return {
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "exit_code": proc.returncode,
                        "cpu": 0.0,
                        "memory": 0.0,
                        "disk_io": 0.0,
                        "net_io": 0.0,
                        "gpu_usage": 0.0,
                        **netem_metrics,
                        **_load_conc(),
                    }
                if psutil is not None:
                    out = _run_psutil()
                    out.update(netem_metrics)
                    return out
                metrics = asyncio.run(
                    _execute_in_container(
                        snip,
                        env_input,
                        network_disabled={
                            "network" in modes or "network_partition" in modes
                        },
                    )
                )
                return {
                    "stdout": "",
                    "stderr": "",
                    "exit_code": int(metrics.get("exit_code", 0)),
                    "cpu": float(metrics.get("cpu", 0.0)),
                    "memory": float(metrics.get("memory", 0.0)),
                    "disk_io": float(metrics.get("disk_io", 0.0)),
                    "net_io": float(metrics.get("net_io", 0.0)),
                    "gpu_usage": float(metrics.get("gpu_usage", 0.0)),
                    **netem_metrics,
                    **_load_conc(),
                }
            except subprocess.TimeoutExpired as exc:
                return {
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "timeout",
                    "exit_code": -1,
                }
            except Exception as exc:  # pragma: no cover - unexpected failure
                return {"stdout": "", "stderr": str(exc), "exit_code": -1}
            finally:
                if _use_netem:
                    try:
                        if _ipr is not None:
                            idx = _ipr.link_lookup(ifname="lo")[0]
                            _ipr.tc("del", "root", idx, "netem")
                            _ipr.close()
                        if _ns_name is not None:
                            netns.remove(_ns_name)
                    except Exception:
                        subprocess.run(
                            ["tc", "qdisc", "del", "dev", "lo", "root", "netem"],
                            check=False,
                        )

    async def _run() -> Dict[str, Any]:
        return await asyncio.to_thread(_run_snippet)

    try:
        concurrency = int(env_input.get("CONCURRENCY_LEVEL", 1))
    except Exception:
        concurrency = 1
    concurrency = max(1, concurrency)

    updates: list[tuple[float, float, Dict[str, float]]] = []
    prev = 0.0
    attempt = 0
    delay = 0.5
    retried = False
    final_result: Dict[str, Any] = {}
    metrics: Dict[str, float] = {}
    duration = 0.0
    while True:
        try:
            start = time.perf_counter()
            if concurrency > 1:
                results = await asyncio.gather(
                    *(_run() for _ in range(concurrency))
                )
            else:
                results = [await _run()]
            duration = time.perf_counter() - start
        except Exception as exc:  # pragma: no cover - runtime failures
            if _ERROR_CONTEXT_BUILDER is not None:
                record_error(exc, context_builder=_ERROR_CONTEXT_BUILDER)
            else:
                logger.exception("sandbox execution failed", exc_info=exc)
            _log_diagnostic(str(exc), False)
            if attempt >= 2:
                raise
            attempt += 1
            retried = True
            await asyncio.sleep(delay)
            delay *= 2
            continue
        attempt = 0
        delay = 0.5

        stdout_combined = ""
        stderr_combined = ""
        metrics = {}
        error_count = 0
        neg = False
        for res in results:
            stdout_combined += res.get("stdout", "")
            stderr_combined += res.get("stderr", "")
            ec = res.get("exit_code", 0)
            if ec != 0:
                error_count += 1
            if ec < 0:
                neg = True
            for k, v in res.items():
                if isinstance(v, (int, float)):
                    metrics[k] = metrics.get(k, 0.0) + float(v)
            metrics["stdout_len"] = metrics.get("stdout_len", 0.0) + float(
                len(res.get("stdout", ""))
            )
            metrics["stderr_len"] = metrics.get("stderr_len", 0.0) + float(
                len(res.get("stderr", ""))
            )
        for k in list(metrics.keys()):
            metrics[k] /= float(concurrency)
        error_rate = error_count / float(concurrency)
        throughput = float(concurrency) / duration if duration > 0 else 0.0
        success_rate = 1.0 - error_rate
        avg_time = duration / float(concurrency)
        metrics.update(
            {
                "concurrency_level": float(concurrency),
                "concurrency_error_rate": error_rate,
                "concurrency_throughput": throughput,
                "success_rate": success_rate,
                "avg_completion_time": avg_time,
            }
        )
        if SANDBOX_EXTRA_METRICS:
            metrics.update(SANDBOX_EXTRA_METRICS)

        result = results[0]
        result = result.copy()
        result["stdout"] = stdout_combined
        result["stderr"] = stderr_combined
        result["exit_code"] = -1 if neg else (0 if error_count == 0 else 1)

        if result.get("exit_code", 0) < 0:
            _log_diagnostic(str(result.get("stderr", "error")), False)
            if attempt >= 2:
                final_result = result
                break
            attempt += 1
            retried = True
            await asyncio.sleep(delay)
            delay *= 2
            continue

        actual = success_rate
        updates.append((prev, actual, metrics))
        if abs(actual - prev) <= threshold:
            if retried:
                _log_diagnostic("section_worker_retry", True)
            final_result = result
            break
        prev = actual
    if cov:
        try:
            cov.stop()
            data = cov.get_data()
            root = repo_root()
            for f in data.measured_files():
                try:
                    rel = Path(f).resolve().relative_to(root).as_posix()
                except Exception:
                    continue
                lines = data.lines(f) or []
                try:
                    source = Path(f).read_text(encoding="utf-8")
                except Exception:
                    continue
                try:
                    tree = ast.parse(source)
                except Exception:
                    continue
                funcs: List[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        end = getattr(node, "end_lineno", node.lineno)
                        if any(l in lines for l in range(node.lineno, end + 1)):
                            funcs.append(node.name)
                if funcs:
                    cov_map[rel] = funcs
        except Exception:
            logger.exception("coverage collection failed")
        try:
            cov.save()
        except Exception:
            pass
    try:
        _score_record_run(
            final_result,
            {
                "roi": updates[-1][1] if updates else 0.0,
                "coverage": cov_map,
                "entropy_delta": metrics.get("entropy_delta", 0.0),
                "runtime": duration,
            },
        )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to record run score")
    return final_result, updates, cov_map


# ----------------------------------------------------------------------
def _load_metrics_file(path: str | Path) -> Dict[str, float]:
    """Return metrics specified in a YAML or JSON file as a dictionary."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        if p.suffix.lower() in {".json", ".jsn"}:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            import yaml  # type: ignore

            with open(p, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
    except Exception:
        logger.exception("failed to load metrics file: %s", p)
        return {}

    metrics = data.get("extra_metrics", data) if isinstance(data, dict) else data
    if isinstance(metrics, list):
        return {str(m): 0.0 for m in metrics}
    if isinstance(metrics, dict):
        out = {}
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                out[str(k)] = 0.0
        return out
    return {}


_SANDBOX_METRICS_FILE = _env_path("SANDBOX_METRICS_FILE", "sandbox_metrics.yaml")
SANDBOX_EXTRA_METRICS: Dict[str, float] = _load_metrics_file(
    str(_SANDBOX_METRICS_FILE)
)


def load_presets() -> List[Dict[str, Any]]:
    """Return sandbox environment presets from ``SANDBOX_ENV_PRESETS``.

    The environment variable is expected to contain a JSON encoded list of
    mappings.  A single mapping is also accepted and automatically wrapped in a
    list.  If the variable is missing or empty a list with one empty mapping is
    returned.  When decoding fails a :class:`ValueError` is raised and the error
    is logged for debugging purposes.
    """

    raw = os.getenv("SANDBOX_ENV_PRESETS", "[]")
    if not raw:
        return [{}]
    try:
        data = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "failed to parse SANDBOX_ENV_PRESETS", extra={"value": raw}, exc_info=exc
        )
        raise ValueError("SANDBOX_ENV_PRESETS is not valid JSON") from exc

    if isinstance(data, dict):
        data = [data]
    try:
        presets = [dict(p) for p in data]
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "invalid preset structure in SANDBOX_ENV_PRESETS",
            extra={"value": raw},
            exc_info=exc,
        )
        raise ValueError("SANDBOX_ENV_PRESETS must be a list of objects") from exc

    if not presets:
        presets = [{}]
    return presets


try:  # maintain legacy global for callers that mutate it in-place
    SANDBOX_ENV_PRESETS: List[Dict[str, Any]] = load_presets()
except ValueError:  # pragma: no cover - fallback when env value is invalid
    SANDBOX_ENV_PRESETS = [{}]

_stub_env = os.getenv("SANDBOX_INPUT_STUBS", "")
try:
    SANDBOX_INPUT_STUBS: List[Dict[str, Any]] = (
        json.loads(_stub_env) if _stub_env else []
    )
    if isinstance(SANDBOX_INPUT_STUBS, dict):
        SANDBOX_INPUT_STUBS = [SANDBOX_INPUT_STUBS]
    SANDBOX_INPUT_STUBS = [dict(s) for s in SANDBOX_INPUT_STUBS]
except Exception:
    SANDBOX_INPUT_STUBS = []

from .stub_providers import discover_stub_providers, StubProvider


def _load_templates(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        logger.exception("failed to load input templates: %s", p)
        return []
    if isinstance(data, dict):
        data = data.get("templates", [])
    if isinstance(data, list):
        return [dict(d) for d in data if isinstance(d, dict)]
    return []


def _load_history(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        if p.suffix.lower() == ".db":
            db = InputHistoryDB(p)
            return db.sample(50)
        if p.suffix.lower() in {".json", ".jsn"}:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                records.extend(dict(r) for r in data if isinstance(r, dict))
        else:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip())
                        if isinstance(obj, dict):
                            records.append(dict(obj))
                    except Exception:
                        continue
    except Exception:
        logger.exception("failed to load input history: %s", p)
    return records


def aggregate_history_stubs() -> Dict[str, Any]:
    """Return aggregated example values from the entire history database."""
    try:
        _get_history_db()
        conn = history_router.get_connection("history")
        rows = conn.execute("SELECT data FROM history").fetchall()
    except Exception:
        logger.exception("failed to aggregate input history")
        return {}

    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            obj = json.loads(row[0])
            if isinstance(obj, dict):
                records.append(obj)
        except Exception:
            continue

    if not records:
        return {}

    stats: dict[str, list[Any]] = {}
    for rec in records:
        for k, v in rec.items():
            stats.setdefault(k, []).append(v)

    result: dict[str, Any] = {}
    for key, vals in stats.items():
        if all(isinstance(v, (int, float)) for v in vals):
            avg = sum(float(v) for v in vals) / len(vals)
            if all(isinstance(v, int) for v in vals):
                avg = int(round(avg))
            result[key] = avg
        else:
            try:
                result[key] = Counter(vals).most_common(1)[0][0]
            except TypeError:
                # Skip unhashable entries such as lists or dicts
                continue
    return result


def recent_failure_stubs(limit: int = 10) -> List[Dict[str, Any]]:
    """Return input-like feature dictionaries from recent failure records."""
    try:
        router = GLOBAL_ROUTER or init_db_router("default")
        conn = router.get_connection("failures")
        rows = conn.execute(
            "SELECT features FROM failures ORDER BY rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    except Exception:
        logger.exception("failed to load recent failure stubs")
        return []

    stubs: List[Dict[str, Any]] = []
    for row in rows:
        try:
            obj = json.loads(row[0])
            if isinstance(obj, dict):
                stubs.append(dict(obj))
        except Exception:
            continue
    return stubs


def _load_callable(path: str | None) -> Callable[..., Any] | None:
    """Return a callable referenced by ``path``.

    ``path`` should be of the form ``"module:attr"``.  When import or
    resolution fails ``None`` is returned and the exception is logged.
    """

    if not path:
        return None
    mod_name, _, attr = path.partition(":")
    try:
        mod = importlib.import_module(mod_name)
        hook = getattr(mod, attr)
        if callable(hook):
            return hook  # type: ignore[return-value]
    except Exception:
        logger.exception(
            "failed to load strategy hook: %s", path_for_prompt(path)
        )
    return None


def _load_strategy_hook(env_var: str) -> Callable[..., Any] | None:
    """Load a strategy hook specified by environment variable ``env_var``."""

    return _load_callable(os.getenv(env_var))


def _random_strategy(
    count: int,
    conf: Dict[str, Any] | None = None,
    *,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    rng = rng or random
    conf = conf or {}
    gen = conf.get("generator") or conf.get("hook")
    if isinstance(gen, str):
        gen = _load_callable(gen)
    if gen is None:
        gen = _load_strategy_hook("SANDBOX_RANDOM_STRATEGY_HOOK")
    if gen:
        try:
            try:
                res = gen(count, conf, rng=rng)
            except TypeError:
                res = gen(count, conf)
            if res:
                return [dict(r) for r in res if isinstance(r, dict)]
        except Exception:
            logger.exception("random strategy hook failed")

    history = conf.get("history")
    if history is None:
        history = _load_history(os.getenv("SANDBOX_INPUT_HISTORY"))
    if history:
        return [dict(rng.choice(history)) for _ in range(count)]

    config = conf.get("config") or {}
    if not config:
        cfg_path = conf.get("config_file")
        if cfg_path:
            p = Path(cfg_path)
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        if p.suffix.lower() in {".yml", ".yaml"}:
                            data = yaml.safe_load(fh)
                        else:
                            data = json.load(fh)
                    if isinstance(data, dict):
                        config = dict(data)
                except Exception:
                    logger.exception(
                        "failed to load random strategy config: %s", p
                    )
    if not config:
        agg = aggregate_history_stubs()
        if agg:
            return [dict(agg) for _ in range(count)]
        return [{} for _ in range(count)]

    modes = config.get("modes") or []
    level_range = config.get("level_range") or []
    flags = config.get("flags") or []
    flag_prob = float(config.get("flag_prob", 0.0))
    stubs: List[Dict[str, Any]] = []
    for _ in range(count):
        stub: Dict[str, Any] = {}
        if modes:
            stub["mode"] = rng.choice(modes)
        if level_range:
            start = int(level_range[0])
            end = int(level_range[1]) if len(level_range) > 1 else start
            stub["level"] = rng.randint(start, end)
        if flags and rng.random() < flag_prob:
            stub["flag"] = rng.choice(flags)
        stubs.append(stub)
    return stubs


def _hostile_strategy(
    count: int,
    target: Callable[..., Any] | None = None,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Return adversarial input dictionaries.

    Payloads can be customised via the ``SANDBOX_HOSTILE_PAYLOADS``
    environment variable or ``SANDBOX_HOSTILE_PAYLOADS_FILE`` pointing to a
    YAML/JSON file.  When unset, a built-in list of common adversarial
    payloads is used.
    """
    hook = _load_strategy_hook("SANDBOX_HOSTILE_STRATEGY_HOOK")
    if hook:
        try:
            res = hook(count, target)
            if res:
                return [dict(r) for r in res if isinstance(r, dict)]
        except Exception:
            logger.exception("hostile strategy hook failed")

    def _parse_payloads(data: str, source: str) -> List[Any]:
        try:
            obj = json.loads(data)
        except Exception:
            try:
                obj = yaml.safe_load(data)
            except Exception:
                logger.warning("could not parse hostile payloads from %s", source)
                return []
        if not isinstance(obj, list):
            logger.warning("hostile payloads from %s not a list", source)
            return []
        valid: List[Any] = []
        for item in obj:
            if isinstance(item, (str, int, float)):
                valid.append(item)
            else:
                logger.warning("ignoring invalid hostile payload %r from %s", item, source)
        return valid

    payloads: List[Any] = []
    raw_env = os.getenv("SANDBOX_HOSTILE_PAYLOADS")
    if raw_env:
        payloads += _parse_payloads(raw_env, "environment")

    file_path = os.getenv("SANDBOX_HOSTILE_PAYLOADS_FILE")
    if file_path:
        p = Path(file_path)
        if p.exists():
            try:
                payloads += _parse_payloads(p.read_text(encoding="utf-8"), str(p))
            except Exception:
                logger.exception("failed to read hostile payloads file: %s", p)
        else:
            logger.warning("hostile payloads file missing: %s", p)

    history = _load_history(os.getenv("SANDBOX_INPUT_HISTORY"))
    for rec in history:
        for val in rec.values():
            if isinstance(val, (str, int, float)):
                payloads.append(val)

    if not payloads:
        payloads = [
            "' OR '1'='1",
            "<script>alert(1)</script>",
            "A" * 10_000,
            "../../etc/passwd",
            '{"id": 1,',  # malformed JSON
            "[1, 2,]",  # malformed JSON array
            "",
            "\x00",
            0,
            -1,
            2**31,
            float('inf'),
            float('-inf'),
            float('nan'),
            None,
        ]

    tmpl_path = str(
        _env_path(
            "SANDBOX_INPUT_TEMPLATES_FILE",
            "sandbox_data/input_stub_templates.json",
        )
    )
    if tmpl_path:
        p = Path(tmpl_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                extras = data.get("hostile") if isinstance(data, dict) else None
                if isinstance(extras, list):
                    payloads += [str(e) for e in extras if isinstance(e, str)]
            except Exception:
                logger.exception("failed to load hostile templates: %s", p)

    base = _stub_from_signature(target, rng=rng) if target else {}
    keys = list(base) or ["payload"]
    stubs: List[Dict[str, Any]] = []
    for i in range(count):
        stub: Dict[str, Any] = {}
        for j, key in enumerate(keys):
            stub[key] = payloads[(i + j) % len(payloads)]
        stubs.append(stub)
    return stubs


def _wrong_type(val: Any) -> Any:
    """Return a value of a different type to ``val``."""
    if isinstance(val, str):
        return 123
    if isinstance(val, (int, float)):
        return "not_a_number"
    if isinstance(val, list):
        return {}
    if isinstance(val, dict):
        return []
    if isinstance(val, bool):
        return "not_bool"
    return None


def _misuse_strategy(
    count: int,
    target: Callable[..., Any] | None = None,
    *,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Return stubs with missing fields or wrong types."""

    base = _stub_from_signature(target, rng=rng) if target else {}
    keys = list(base) or ["value"]
    stubs: List[Dict[str, Any]] = []
    for i in range(count):
        stub = dict(base) or {keys[0]: 0}
        if stub:
            key = keys[i % len(keys)]
            if key in stub:
                stub[key] = _wrong_type(stub[key])
            if len(keys) > 1:
                missing = keys[(i + 1) % len(keys)]
                stub.pop(missing, None)
            else:
                stub.pop(key, None)
        stubs.append(stub)
    return stubs


def _misuse_provider(
    stubs: List[Dict[str, Any]] | None, ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Append adversarial payloads when misuse stubs are enabled."""

    settings = SandboxSettings()
    if not settings.misuse_stubs:
        return stubs or []
    if isinstance(ctx, dict) and ctx.get("strategy") in {"hostile", "misuse"}:
        return stubs or []
    count = len(stubs or []) or 1
    target = ctx.get("target") if isinstance(ctx, dict) else None
    rng = ctx.get("rng") if isinstance(ctx, dict) else None
    hostile = _hostile_strategy(count, target, rng=rng)
    return (stubs or []) + hostile


def _validate_stubs(
    stubs: List[Dict[str, Any]] | None, target: Callable[..., Any] | None
) -> List[Dict[str, Any]]:
    """Return ``stubs`` filtered to match ``target`` signature."""

    if not stubs or target is None:
        return stubs or []
    try:
        sig = inspect.signature(target)
    except Exception:
        return stubs or []
    required = {
        name
        for name, p in sig.parameters.items()
        if p.default is inspect._empty
    }
    allowed = set(sig.parameters)
    valid: List[Dict[str, Any]] = []
    for stub in stubs:
        if not isinstance(stub, dict):
            continue
        if required - set(stub):
            continue
        cleaned = {k: v for k, v in stub.items() if k in allowed}
        valid.append(cleaned)
    return valid or (stubs or [])


def _smart_value(name: str, hint: Any, rng: random.Random | None = None) -> Any:
    """Return a realistic value for ``name`` with type ``hint``."""
    val = None
    if _FAKER is not None:
        faker_random = None
        if rng is not None:
            faker_random = _FAKER.random
            _FAKER.random = rng
        try:
            if hint is str:
                lowered = name.lower()
                if "email" in lowered:
                    val = _FAKER.email()
                elif "name" in lowered:
                    val = _FAKER.name()
                elif "url" in lowered:
                    val = _FAKER.url()
                else:
                    val = _FAKER.word()
            elif hint is int:
                val = _FAKER.random_int(min=0, max=1000)
            elif hint is float:
                val = float(_FAKER.pyfloat(left_digits=2, right_digits=2, positive=True))
            elif hint is bool:
                val = _FAKER.pybool()
            elif hint is datetime:
                val = _FAKER.date_time()
        finally:
            if faker_random is not None:
                _FAKER.random = faker_random
    if val is None and _hyp_strats is not None and hint is not inspect._empty:
        try:
            val = _hyp_strats.from_type(hint).example()
        except Exception:
            val = None
    return val


def _stub_from_signature(
    func: Callable[..., Any], *, smart: bool = False, rng: random.Random | None = None
) -> Dict[str, Any]:
    """Return an input stub derived from ``func`` signature."""
    stub: Dict[str, Any] = {}
    try:
        sig = inspect.signature(func)
    except Exception:
        return stub
    for name, param in sig.parameters.items():
        if param.default is not inspect._empty:
            stub[name] = param.default
            continue
        hint = param.annotation
        val: Any = None
        if smart:
            val = _smart_value(name, hint, rng=rng)
        if hint is not inspect._empty and val is None:
            origin = get_origin(hint)
            if origin is list or hint is list:
                val = []
            elif origin is dict or hint is dict:
                val = {}
            elif origin is tuple or hint is tuple:
                val = []
            elif origin is set or hint is set:
                val = set()
            elif hint in (int, float):
                val = 0
            elif hint is bool:
                val = False
            elif hint is str:
                val = ""
        stub[name] = val
    return stub


def generate_input_stubs(
    count: int | None = None,
    *,
    target: Callable[..., Any] | None = None,
    strategy: str | None = None,
    providers: List[StubProvider] | None = None,
    strategy_order: Sequence[str] | None = None,
    seed: int | None = None,
    settings: "SandboxSettings" | None = None,
) -> List[Dict[str, Any]]:
    """Return example input dictionaries.

    ``SANDBOX_INPUT_STUBS`` overrides all other behaviour. When unset the
    generator consults ``providers`` discovered via ``SANDBOX_STUB_PLUGINS``.
    Strategy selection and template paths are resolved from ``SandboxSettings``
    when ``settings`` is not supplied.  The default fallback sequence is a
    deterministic pipeline that consults history and failure records, applies
    smart generation and model providers, and only then considers adversarial
    modes. ``strategy_order`` controls the fallback sequence when ``strategy``
    is not specified. The ``hostile`` strategy crafts adversarial payloads such
    as SQL injection strings, oversized buffers and malformed JSON. The
    ``misuse`` strategy omits fields or supplies values of incorrect types to
    mimic common user errors. The ``smart`` strategy attempts to generate
    realistic values using ``faker`` or ``hypothesis`` when available. The
    ``synthetic`` strategy mirrors ``smart`` but is intended for language model
    based stub providers.
    """

    if settings is None:
        from sandbox_settings import SandboxSettings

        settings = SandboxSettings()

    if strategy is None:
        strategy = settings.stub_strategy

    if seed is None:
        seed = settings.stub_seed
    if seed is None:
        rng = random
    else:
        rng = random.Random(seed)
        random.seed(seed)
    if _FAKER is not None:
        _FAKER.random = rng

    if SANDBOX_INPUT_STUBS:
        stubs = [dict(s) for s in SANDBOX_INPUT_STUBS]
        providers = providers or discover_stub_providers()
        for prov in providers:
            try:
                new = prov(stubs, {"strategy": "env", "target": target, "rng": rng})
                if new:
                    stubs = [dict(s) for s in new if isinstance(s, dict)]
            except Exception:
                logger.exception(
                    "stub provider %s failed", getattr(prov, "__name__", "?")
                )
        return stubs

    num = 2 if count is None else max(0, count)

    providers = providers or discover_stub_providers()
    if settings.misuse_stubs and _misuse_provider not in providers:
        providers = list(providers) + [_misuse_provider]

    history = _load_history(settings.input_history)
    templates: List[Dict[str, Any]] | None = None
    stubs: List[Dict[str, Any]] | None = None

    if strategy_order is None:
        if settings.stub_strategy_order:
            strategy_order = list(settings.stub_strategy_order)
        else:
            strategy_order = [
                "history",
                "failures",
                "smart",
                "synthetic",
                "hostile",
                "misuse",
            ]

    if strategy:
        strategy_order = [strategy] + [s for s in strategy_order if s != strategy]

    chosen = ""
    templates_checked = False

    for strat in strategy_order:
        if strat == "history":
            if history:
                stubs = [dict(rng.choice(history)) for _ in range(num)]
            elif settings.input_history:
                agg = aggregate_history_stubs()
                if agg:
                    stubs = [dict(agg) for _ in range(num)]
            if stubs is not None:
                chosen = "history"
                break
        elif strat == "failures":
            failures = recent_failure_stubs()
            if failures:
                stubs = [dict(rng.choice(failures)) for _ in range(num)]
                chosen = "failures"
                break
        elif strat == "signature" and target is not None:
            base = _stub_from_signature(target, smart=False, rng=rng)
            stubs = [dict(base) for _ in range(num)]
            chosen = "signature"
            break
        elif strat == "templates":
            templates_checked = True
            if templates is None:
                templates = _load_templates(settings.input_templates_file)
            if templates:
                stubs = [dict(rng.choice(templates)) for _ in range(num)]
                chosen = "templates"
                break
        elif strat == "smart" and target is not None:
            base = _stub_from_signature(target, smart=True, rng=rng)
            stubs = [dict(base) for _ in range(num)]
            chosen = "smart"
            break
        elif strat == "synthetic" and target is not None:
            base = _stub_from_signature(target, smart=True, rng=rng)
            stubs = [dict(base) for _ in range(num)]
            chosen = "synthetic"
            break
        elif strat == "hostile":
            stubs = _hostile_strategy(num, target, rng=rng)
            chosen = "hostile"
            break
        elif strat == "misuse":
            stubs = _misuse_strategy(num, target, rng=rng)
            chosen = "misuse"
            break
        elif strat == "random":
            conf = dict(settings.stub_random_config)
            if settings.stub_random_config_file:
                conf.setdefault("config_file", settings.stub_random_config_file)
            if settings.stub_random_generator:
                conf.setdefault("generator", settings.stub_random_generator)
            stubs = _random_strategy(num, conf, rng=rng) or [{}]
            chosen = "random"
            break

    if stubs is None:
        raise RuntimeError("no input stubs could be generated")

    if chosen in {"history", "failures", "smart"} or (
        templates_checked and (templates is None or not templates)
    ):
        try:
            from . import generative_stub_provider as gsp  # local import

            stubs = gsp.generate_stubs(
                stubs, {"strategy": chosen, "target": target, "rng": rng}
            )
        except Exception:
            logger.exception("%s stub generation failed", chosen)

    for prov in providers:
        try:
            new = prov(stubs, {"strategy": chosen, "target": target, "rng": rng})
            if new:
                stubs = [dict(s) for s in new if isinstance(s, dict)]
        except Exception:
            logger.exception("stub provider %s failed", getattr(prov, "__name__", "?"))

    stubs = _validate_stubs(stubs, target)
    if not stubs:
        raise RuntimeError("input stub strategies yielded no valid stubs")
    return stubs


# ----------------------------------------------------------------------

def _scenario_specific_metrics(
    scenario: str, metrics: Dict[str, float]
) -> Dict[str, float]:
    """Return additional metrics for the given *scenario*.

    The helper inspects ``scenario`` and extracts or derives metrics that are
    only meaningful for that scenario type. Returned values are merged into the
    main metrics dictionary by the callers.
    """

    name = scenario.lower()
    extra: Dict[str, float] = {}
    if "latency" in name:
        extra["latency_error_rate"] = float(metrics.get("error_rate", 0.0))
    if "hostile" in name:
        extra["hostile_failures"] = float(metrics.get("failure_count", 0.0))
        extra["hostile_sanitization_failures"] = float(
            metrics.get("sanitization_failures", 0.0)
        )
        extra["hostile_validation_failures"] = float(
            metrics.get("validation_failures", 0.0)
        )
    if "misuse" in name:
        extra["misuse_failures"] = float(metrics.get("failure_count", 0.0))
        extra["misuse_invalid_calls"] = float(
            metrics.get("invalid_call_count", 0.0)
        )
        extra["misuse_recovery_attempts"] = float(
            metrics.get("recovery_attempts", 0.0)
        )
    if "concurrency" in name:
        extra["concurrency_throughput"] = float(
            metrics.get("throughput", metrics.get("concurrency_throughput", 0.0))
        )
        extra["concurrency_thread_saturation"] = float(
            metrics.get("concurrency_threads", 0.0)
        )
        extra["concurrency_async_saturation"] = float(
            metrics.get("concurrency_tasks", 0.0)
        )
        err_rate = float(
            metrics.get("concurrency_error_rate", metrics.get("error_rate", 0.0))
        )
        level = float(metrics.get("concurrency_level", 1.0))
        extra["concurrency_error_count"] = err_rate * level
    if "schema_drift" in name or "schema_mismatch" in name:
        mismatches = float(
            metrics.get("schema_mismatches", metrics.get("schema_mismatch_count", 0.0))
        )
        total = float(
            metrics.get(
                "schema_checks",
                metrics.get("schema_check_count", metrics.get("records_checked", 0.0)),
            )
        )
        extra["schema_errors"] = mismatches
        extra["schema_mismatch_rate"] = mismatches / total if total else 0.0
    if "flaky_upstream" in name or "upstream" in name:
        failures = float(
            metrics.get("upstream_failures", metrics.get("failure_count", 0.0))
        )
        calls = float(
            metrics.get(
                "upstream_requests",
                metrics.get("request_count", metrics.get("calls", 0.0)),
            )
        )
        extra["upstream_failures"] = failures
        extra["upstream_failure_rate"] = failures / calls if calls else 0.0
    return extra


# ----------------------------------------------------------------------
def _preset_concurrency_spike(multiplier: int | None = None) -> Dict[str, Any]:
    """Preset stressing concurrency limits."""

    settings = SandboxSettings()
    mult = multiplier if multiplier is not None else settings.preset_concurrency_multiplier
    return {
        "SCENARIO_NAME": "concurrency_spike",
        "FAILURE_MODES": "concurrency_spike",
        "CONCURRENCY_MULTIPLIER": mult,
        "CONCURRENCY_LEVEL": settings.preset_concurrency_level,
    }


def _preset_hostile_input() -> Dict[str, Any]:
    """Preset injecting adversarial or malformed inputs."""

    settings = SandboxSettings()
    return {
        "SCENARIO_NAME": "hostile_input",
        "FAILURE_MODES": "hostile_input",
        "SANDBOX_STUB_STRATEGY": settings.preset_hostile_stub_strategy,
        "HOSTILE_INPUT": settings.preset_hostile_input,
        "INJECT_EDGE_CASES": True,
    }


def _preset_schema_drift() -> Dict[str, Any]:
    """Preset simulating legacy or mismatched schemas."""

    settings = SandboxSettings()
    return {
        "SCENARIO_NAME": "schema_drift",
        "FAILURE_MODES": "schema_drift",
        "SANDBOX_STUB_STRATEGY": settings.preset_schema_stub_strategy,
        "SCHEMA_MISMATCHES": settings.preset_schema_mismatches,
        "SCHEMA_CHECKS": settings.preset_schema_checks,
    }


def _preset_flaky_upstream() -> Dict[str, Any]:
    """Preset emulating unreliable upstream dependencies."""

    settings = SandboxSettings()
    return {
        "SCENARIO_NAME": "flaky_upstream",
        "FAILURE_MODES": "flaky_upstream",
        "UPSTREAM_FAILURES": settings.preset_upstream_failures,
        "UPSTREAM_REQUESTS": settings.preset_upstream_requests,
        "SANDBOX_STUB_STRATEGY": settings.preset_flaky_stub_strategy,
        "API_LATENCY_MS": settings.preset_api_latency_ms,
    }


def _preset_high_latency() -> Dict[str, Any]:
    """Preset introducing artificial network delays."""

    settings = SandboxSettings()
    return {
        "SCENARIO_NAME": "high_latency",
        "NETWORK_LATENCY_MS": settings.preset_network_latency_ms,
    }


def _preset_resource_strain() -> Dict[str, Any]:
    """Preset throttling compute and disk resources."""

    settings = SandboxSettings()
    return {
        "SCENARIO_NAME": "resource_strain",
        "CPU_LIMIT": settings.preset_cpu_limit,
        "DISK_LIMIT": settings.preset_disk_limit,
    }


def _preset_chaotic_failure() -> Dict[str, Any]:
    """Preset representing broken auth and corrupt payload injection."""

    return {
        "SCENARIO_NAME": "chaotic_failure",
        "BROKEN_AUTH": True,
        "CORRUPT_PAYLOAD": True,
        "INJECT_EDGE_CASES": True,
    }


def default_scenario_presets() -> List[Dict[str, Any]]:
    """Return the standard set of scenario presets used by ``run_scenarios``."""

    return [
        {"SCENARIO_NAME": "normal"},
        _preset_concurrency_spike(),
        _preset_hostile_input(),
        _preset_schema_drift(),
        _preset_flaky_upstream(),
        _preset_high_latency(),
        _preset_resource_strain(),
        _preset_chaotic_failure(),
    ]


def temporal_trajectory_presets() -> List[Dict[str, Any]]:
    """Return canonical presets for temporal trajectory simulations."""

    return [
        {"SCENARIO_NAME": "normal"},
        _preset_high_latency(),
        _preset_resource_strain(),
        _preset_schema_drift(),
        _preset_chaotic_failure(),
    ]


def temporal_presets() -> List[Dict[str, Any]]:
    """Return ordered temporal scenario presets with explicit stages."""

    return [
        {"SCENARIO_NAME": "baseline"},
        {"SCENARIO_NAME": "latency", "NETWORK_LATENCY_MS": 500},
        {"SCENARIO_NAME": "strain", "CPU_LIMIT": 0.5, "DISK_IO_THROTTLE": 100},
        {"SCENARIO_NAME": "drift", "SCHEMA_DRIFT": True, "UNEXPECTED_INPUT": True},
        {"SCENARIO_NAME": "chaos", "BROKEN_AUTH": True, "CORRUPT_PAYLOAD": True},
    ]


# ----------------------------------------------------------------------


@dataclass
class Scorecard:
    """Scorecard summarising scenario performance.

    Attributes
    ----------
    scenario:
        Name of the scenario preset.
    baseline_roi:
        ROI recorded for the baseline "normal" run.
    stress_roi:
        ROI observed when executing under this scenario.
    roi_delta:
        Difference between ``stress_roi`` and ``baseline_roi``.
    metrics_delta:
        Metric deltas relative to the baseline run.
    synergy:
        Recorded synergy metrics for the scenario.
    """

    scenario: str
    baseline_roi: float
    stress_roi: float
    roi_delta: float
    metrics_delta: Dict[str, float]
    synergy: Dict[str, float]
    recommendation: str | None = None
    status: str | None = None


def run_scenarios(
    workflow: Sequence[str] | str,
    tracker: "ROITracker" | None = None,
    presets: Sequence[Mapping[str, Any]] | None = None,
    *,
    foresight_tracker: "ForesightTracker" | None = None,
) -> tuple["ROITracker", Dict[str, Scorecard], Dict[str, Any]]:
    """Run ``workflow`` across predefined sandbox scenarios and compare ROI.

    The workflow is executed in a baseline "normal" environment followed by
    four adverse scenarios: ``concurrency_spike``, ``hostile_input``,
    ``schema_drift`` and ``flaky_upstream``. Each scenario is executed twice –
    once with the workflow enabled and once with it disabled – to measure the
    direct contribution of the workflow. For each run the ROI and metrics are
    recorded, metric deltas are calculated relative to the baseline run and the
    ROI delta between the enabled and disabled states is reported. Synergy
    metrics are tracked through :class:`menace.roi_tracker.ROITracker`.

    Parameters
    ----------
    workflow:
        Sequence of step strings or single step name forming the workflow to
        execute. Each step must include an explicit module reference in the form
        ``module:function`` or ``module.function``. Objects providing a
        ``workflow`` attribute are also accepted for backward compatibility.
    tracker:
        Optional ROI tracker used to calculate diminishing returns and record
        synergy metrics. When omitted a new tracker is created.
    presets:
        Optional sequence of environment preset dictionaries. When omitted the
        :func:`default_scenario_presets` helper provides a baseline "normal"
        run plus canonical adverse scenarios for concurrency pressure, hostile
        input, schema drift and flaky upstreams.
    foresight_tracker:
        Optional foresight tracker for synchronising ROI foresight metrics.

    Returns
    -------
    tuple[ROITracker, Dict[str, Scorecard], Dict[str, Any]]
        A triple consisting of the :class:`ROITracker` used for the simulations,
        a mapping of :class:`Scorecard` instances keyed by scenario name and a
        summary mapping. ``scenarios`` maps scenario names to dictionaries
        containing the ROI, RAROI, ROI delta between workflow on/off,
        RAROI delta, raw metrics, metric deltas and recorded synergy metrics.
        ``scorecards`` mirrors the returned mapping with serialisable
        dictionaries. ``worst_scenario`` identifies the scenario causing the
        largest ROI drop relative to the baseline run.
    """

    from menace.roi_tracker import ROITracker

    if tracker is None:
        tracker = ROITracker()

    def _steps(obj: Sequence[str] | str) -> list[str]:
        if isinstance(obj, str):
            return [obj]
        try:
            return [str(s) for s in obj]
        except TypeError:
            return [str(s) for s in getattr(obj, "workflow", [])]

    def _wf_snippet(steps: list[str]) -> str:
        imports: list[str] = []
        calls: list[str] = []
        for idx, step in enumerate(steps):
            alias = f"_wf_{idx}"
            if ":" in step:
                mod, func = step.split(":", 1)
            else:
                if "." in step:
                    mod, func = step.rsplit(".", 1)
                else:
                    raise ValueError(
                        f"Workflow step '{step}' must include a module path"
                    )
            if importlib.util.find_spec(mod) is None:
                raise ValueError(
                    f"Module '{mod}' for workflow step '{step}' not found"
                )
            imports.append(f"from {mod} import {func} as {alias}")
            calls.append(f"{alias}()")
        if not calls:
            return "pass\n"
        return "\n".join(imports + [""] + calls) + "\n"

    wf_id = getattr(workflow, "wid", getattr(workflow, "id", "0"))
    wf_steps = _steps(workflow)

    synergy_suggestions: Dict[str, List[str]] = {}
    if _USE_MODULE_SYNERGY:
        for step in wf_steps:
            if ":" in step:
                mod, _func = step.split(":", 1)
            elif "." in step:
                mod, _func = step.rsplit(".", 1)
            elif importlib.util.find_spec(step) is not None:
                mod = step
            else:
                raise ValueError(
                    f"Workflow step '{step}' must include a module path"
                )
            try:
                cluster = get_synergy_cluster(mod)
            except Exception:
                cluster = set()
            if mod in cluster:
                cluster.discard(mod)
            if cluster:
                synergy_suggestions[mod] = sorted(cluster)
        if synergy_suggestions:
            logger.info("module synergy suggestions: %s", synergy_suggestions)

    snippet_on = _wf_snippet(wf_steps)
    snippet_off = _wf_snippet([])
    presets = list(presets) if presets is not None else default_scenario_presets()

    results: Dict[str, Dict[str, Any]] = {}
    baseline_roi: float = 0.0
    baseline_raroi: float = 0.0
    baseline_metrics: Dict[str, float] = {}

    async def _run() -> None:
        nonlocal baseline_roi, baseline_metrics, baseline_raroi
        for idx, preset in enumerate(presets):
            env_input = dict(preset)
            scenario = env_input.get("SCENARIO_NAME", "")

            _, updates_on, _ = await _section_worker(
                snippet_on, env_input, tracker.diminishing(), runner_config
            )
            roi_on = updates_on[-1][1] if updates_on else 0.0
            metrics_on = updates_on[-1][2] if updates_on else {}

            _, updates_off, _ = await _section_worker(
                snippet_off, env_input, tracker.diminishing(), runner_config
            )
            roi_off = updates_off[-1][1] if updates_off else 0.0
            metrics_off = updates_off[-1][2] if updates_off else {}

            target_metrics_delta = {
                k: metrics_on.get(k, 0.0) - metrics_off.get(k, 0.0)
                for k in set(metrics_on) | set(metrics_off)
            }

            delta = roi_on - roi_off
            synergy_diff = {
                f"synergy_{k}": float(v) for k, v in target_metrics_delta.items()
            }
            synergy_diff["synergy_roi"] = delta

            if idx == 0:
                baseline_roi, baseline_metrics = roi_on, metrics_on
                metrics_delta = {k: 0.0 for k in metrics_on}
            else:
                metrics_delta = {
                    k: metrics_on.get(k, 0.0) - baseline_metrics.get(k, 0.0)
                    for k in set(metrics_on) | set(baseline_metrics)
                }

            synergy_metrics = {
                f"synergy_{k}": float(v) for k, v in metrics_delta.items()
            }
            synergy_metrics["synergy_roi"] = delta
            synergy_metrics.setdefault("synergy_profitability", delta)
            synergy_metrics.setdefault("synergy_revenue", delta)
            synergy_metrics.setdefault("synergy_projected_lucrativity", delta)
            tracker.scenario_synergy.setdefault(scenario, []).append(synergy_metrics)
            tracker.register_metrics(*synergy_metrics.keys())
            tracker.update(
                baseline_roi,
                roi_on,
                modules=[f"workflow_{wf_id}", scenario],
                metrics={**metrics_on, **synergy_metrics},
            )
            raroi_on = tracker.last_raroi or 0.0
            if idx == 0:
                baseline_raroi = raroi_on
                raroi_delta = 0.0
            else:
                raroi_delta = raroi_on - baseline_raroi
            tracker.record_scenario_delta(
                scenario, delta, metrics_delta, synergy_diff, raroi_on, raroi_delta
            )

            results[scenario] = {
                "roi": roi_on,
                "roi_delta": delta,
                "raroi": raroi_on,
                "raroi_delta": raroi_delta,
                "metrics": metrics_on,
                "metrics_delta": metrics_delta,
                "synergy": synergy_metrics,
                "runs": [
                    {"flag": "on", "roi": roi_on, "metrics": metrics_on},
                    {"flag": "off", "roi": roi_off, "metrics": metrics_off},
                ],
                "target_delta": {
                    "roi": delta,
                    "metrics": target_metrics_delta,
                },
            }

    asyncio.run(_run())

    worst, _ = tracker.biggest_drop()

    export = {
        scen: {
            "roi_delta": delta,
            "raroi_delta": tracker.scenario_raroi_delta.get(scen, 0.0),
            "worst": scen == worst,
        }
        for scen, delta in tracker.scenario_roi_deltas.items()
    }
    try:
        out_path = _env_path("SANDBOX_DATA_DIR", "sandbox_data") / "scenario_deltas.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(export, fh)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to write scenario deltas")
    scenario_cards = {c.scenario: c for c in tracker.generate_scorecards()}
    scorecards: Dict[str, Scorecard] = {}
    for scen, info in results.items():
        rec = scenario_cards.get(scen)
        scorecards[scen] = Scorecard(
            scenario=scen,
            baseline_roi=baseline_roi,
            stress_roi=info["roi"],
            roi_delta=info["roi"] - baseline_roi,
            metrics_delta=info["metrics_delta"],
            synergy=info["synergy"],
            recommendation=rec.recommendation if rec else None,
            status=rec.status if rec else None,
        )
    summary = {
        "scenarios": results,
        "worst_scenario": worst,
        "scorecards": {scen: asdict(card) for scen, card in scorecards.items()},
        "status": tracker.workflow_label,
    }
    summary["run_summary"] = _load_run_summary()
    if synergy_suggestions:
        summary["synergy_suggestions"] = synergy_suggestions
    try:
        summary["workflow_scorecard"] = generate_scorecard(workflow, summary)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to generate workflow scorecard")
        summary["workflow_scorecard"] = {}
    return tracker, scorecards, summary


# ----------------------------------------------------------------------
def generate_scorecard(
    workflow: Sequence[str] | str, summary: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return workflow stress test scorecard and persist it.

    Parameters
    ----------
    workflow:
        Workflow object or sequence of step names identifying the workflow.
    summary:
        Mapping produced by :func:`run_scenarios` containing per-scenario
        results.

    Returns
    -------
    dict
        Mapping containing the ``workflow_id`` and per-scenario ROI delta and
        metrics.
    """

    wf_id = str(getattr(workflow, "wid", getattr(workflow, "id", "0")))
    card: Dict[str, Any] = {"workflow_id": wf_id, "scenarios": {}}
    card["run_summary"] = _load_run_summary()

    # Import lazily to avoid expensive module import when not needed.
    try:  # pragma: no cover - import error shouldn't crash
        from menace.roi_tracker import HARDENING_TIPS
    except Exception:  # pragma: no cover
        HARDENING_TIPS = {}

    failing: List[str] = []
    passing: List[str] = []
    for scen, info in summary.get("scenarios", {}).items():
        metrics = info.get("target_delta", {}).get("metrics", {})
        roi_delta = float(info.get("roi_delta", 0.0))
        card["scenarios"][scen] = {
            "roi_delta": roi_delta,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "recommendation": HARDENING_TIPS.get(scen),
        }

        triggers_failure = any(
            (("failure" in k) or ("error" in k) or k.endswith("_breach"))
            and v > 0
            for k, v in metrics.items()
        )
        if roi_delta < 0.0 or triggers_failure:
            failing.append(scen)
        else:
            passing.append(scen)

    if len(failing) == 1 and passing:
        card["status"] = "situationally weak"
    out_path = _env_path("SANDBOX_DATA_DIR", "sandbox_data") / f"scorecard_{wf_id}.json"
    try:  # pragma: no cover - best effort
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(card, fh)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to write workflow scorecard")
    return card


# ----------------------------------------------------------------------
def simulate_temporal_trajectory(
    workflow_id: str,
    workflow: Sequence[str] | str,
    tracker: "ROITracker" | None = None,
    foresight_tracker: "ForesightTracker" | None = None,
) -> "ROITracker":
    """Execute ``workflow`` through sequential :func:`temporal_presets`."""

    from menace.roi_tracker import ROITracker

    if tracker is None:
        tracker = ROITracker()

    def _steps(obj: Sequence[str] | str) -> list[str]:
        if isinstance(obj, str):
            return [obj]
        try:
            return [str(s) for s in obj]
        except TypeError:
            return [str(s) for s in getattr(obj, "workflow", [])]

    def _wf_snippet(steps: list[str]) -> str:
        imports: list[str] = []
        calls: list[str] = []
        for idx, step in enumerate(steps):
            alias = f"_wf_{idx}"
            if ":" in step:
                mod, func = step.split(":", 1)
            else:
                if "." in step:
                    mod, func = step.rsplit(".", 1)
                else:
                    raise ValueError(
                        f"Workflow step '{step}' must include a module path"
                    )
            if importlib.util.find_spec(mod) is None:
                raise ValueError(
                    f"Module '{mod}' for workflow step '{step}' not found"
                )
            imports.append(f"from {mod} import {func} as {alias}")
            calls.append(f"{alias}()")
        if not calls:
            return "pass\n"
        return "\n".join(imports + [""] + calls) + "\n"

    wf_steps = _steps(workflow)
    snippet = _wf_snippet(wf_steps)
    baseline_roi: float | None = None

    for preset in temporal_presets():
        env_input = dict(preset)
        name = str(env_input.get("SCENARIO_NAME", ""))
        try:
            simulate_execution_environment(snippet, env_input)
        except Exception:
            logger.exception("temporal preset %s pre-check failed", name)
        _, updates, _ = asyncio.run(
            _section_worker(snippet, env_input, tracker.diminishing(), runner_config)
        )
        roi = float(updates[-1][1]) if updates else 0.0
        metrics = updates[-1][2] if updates else {}
        resilience = float(metrics.get("resilience", 0.0))
        metrics.setdefault("synergy_resilience", resilience)
        if baseline_roi is None:
            baseline_roi = roi
        delta = roi - baseline_roi
        tracker.register_metrics(*metrics.keys())
        tracker.update(
            baseline_roi,
            roi,
            modules=[f"workflow_{workflow_id}", name],
            metrics=metrics,
        )
        tracker.record_scenario_delta(name, delta)
        degradation = tracker.scenario_degradation()

        if foresight_tracker is not None:
            foresight_tracker.capture_from_roi(
                tracker,
                str(workflow_id),
                stage=name,
                compute_stability=True,
            )
            _, _, stability = foresight_tracker.get_trend_curve(str(workflow_id))
            logging.info(
                "temporal stage %s: roi=%.3f resilience=%.3f stability=%.3f degradation=%.3f",
                name,
                roi,
                resilience,
                stability,
                degradation,
            )

    return tracker


# ----------------------------------------------------------------------
def run_repo_section_simulations(
    repo_path: str,
    input_stubs: List[Dict[str, Any]] | None = None,
    env_presets: List[Dict[str, Any]] | Mapping[str, List[Dict[str, Any]]] | None = None,
    modules: Iterable[str] | None = None,
    *,
    context_builder: ContextBuilder,
    return_details: bool = False,
) -> "ROITracker" | tuple["ROITracker", Dict[str, Dict[str, list[Dict[str, Any]]]]]:
    """Analyse sections and simulate execution environment per section.

    Parameters
    ----------
    repo_path:
        Root of repository to analyse.
    modules:
        Optional iterable of relative module paths. When provided, only these
        paths are scanned for sections.
    """
    from menace.roi_tracker import ROITracker
    from menace.self_debugger_sandbox import SelfDebuggerSandbox
    from menace.self_coding_engine import SelfCodingEngine
    from menace.code_database import CodeDB
    from menace.menace_memory_manager import MenaceMemoryManager
    from .metrics_plugins import (
        discover_metrics_plugins,
        collect_plugin_metrics,
    )
    get_error_logger(context_builder)
    try:
        from menace.environment_generator import _PROFILE_ALIASES, CANONICAL_PROFILES
    except Exception:  # pragma: no cover - environment generator optional
        _PROFILE_ALIASES = {}
        CANONICAL_PROFILES = [
            "high_latency_api",
            "hostile_input",
            "user_misuse",
            "concurrency_spike",
            "schema_drift",
            "flaky_upstream",
        ]
    try:
        from sandbox_settings import SandboxSettings
        metric_thresholds = (
            SandboxSettings().scenario_metric_thresholds or {}
        )
    except Exception:
        metric_thresholds = {}

    def _check_thresholds(scenario: str, metric_dict: Dict[str, float]) -> Dict[str, float]:
        flags: Dict[str, float] = {}
        for name, limit in metric_thresholds.items():
            if name in metric_dict:
                try:
                    val = float(metric_dict.get(name, 0.0))
                except Exception:
                    val = 0.0
                breach = val > float(limit)
                flags[f"{name}_breach"] = 1.0 if breach else 0.0
                if breach:
                    logger.warning(
                        "scenario %s metric %s=%s exceeds threshold %s",
                        scenario,
                        name,
                        val,
                        limit,
                    )
        return flags

    if input_stubs is None:
        input_stubs = generate_input_stubs()
    if env_presets is None:
        if os.getenv("SANDBOX_GENERATE_PRESETS", "1") != "0":
            try:
                from menace.environment_generator import (
                    generate_presets,
                    generate_canonical_presets,
                )

                if os.getenv("SANDBOX_PRESET_MODE") == "canonical":
                    env_presets = generate_canonical_presets()
                else:
                    env_presets = generate_presets()
            except Exception:
                env_presets = [{}]
        else:
            env_presets = [{}]

    base_builder = context_builder
    try:
        base_builder.refresh_db_weights()
    except Exception:
        pass

    async def _run() -> (
        "ROITracker" | tuple["ROITracker", Dict[str, Dict[str, list[Dict[str, Any]]]]]
    ):
        from sandbox_runner import scan_repo_sections

        logger.info("scanning repository sections in %s", path_for_prompt(repo_path))
        sections = scan_repo_sections(repo_path, modules=modules)
        tracker = ROITracker()
        plugins = discover_metrics_plugins(os.environ)

        if isinstance(env_presets, Mapping):
            keys = {str(k) for k in env_presets}
            section_keys = set(sections.keys())
            if keys & section_keys:
                preset_map: Dict[str, List[Dict[str, Any]]] = {}
                for k, v in env_presets.items():
                    if isinstance(v, Mapping):
                        flattened = [p for lst in v.values() for p in lst]
                    else:
                        flattened = list(v)
                    preset_map[str(k)] = flattened
                all_presets: List[Dict[str, Any]] = [
                    p for lst in preset_map.values() for p in lst
                ]
            else:
                preset_map = {}
                all_presets = []
                for v in env_presets.values():
                    if isinstance(v, Mapping):
                        high = v.get("high")
                        if high:
                            all_presets.append(high)
                        else:
                            all_presets.append(next(iter(v.values())))
                    else:
                        all_presets.extend(list(v))
        else:
            preset_map = {}
            all_presets = list(env_presets)

        try:
            from menace.environment_generator import (
                generate_presets,
                suggest_profiles_for_module,
            )
        except Exception:
            generate_presets = None  # type: ignore
            suggest_profiles_for_module = None  # type: ignore

        for module in sections:
            mod_presets = list(preset_map.get(module, []))
            for preset in all_presets:
                if preset not in mod_presets:
                    mod_presets.append(preset)
            if generate_presets and suggest_profiles_for_module:
                try:
                    profiles = suggest_profiles_for_module(module) or []
                    new_presets = generate_presets(profiles=profiles)
                except Exception:
                    new_presets = []
                for preset in new_presets:
                    if preset not in mod_presets:
                        mod_presets.append(preset)
                    if preset not in all_presets:
                        all_presets.append(preset)
            preset_map[module] = mod_presets

        required_names = {
            "high_latency_api",
            "hostile_input",
            "user_misuse",
            "concurrency_spike",
            "schema_drift",
            "flaky_upstream",
        }
        present_names = {
            _PROFILE_ALIASES.get(p.get("SCENARIO_NAME"), p.get("SCENARIO_NAME"))
            for p in all_presets
            if p.get("SCENARIO_NAME")
        }
        missing_names = required_names - present_names
        if missing_names:
            extra_presets: List[Dict[str, Any]] = []
            try:
                from menace.environment_generator import generate_canonical_presets

                canonical_map = generate_canonical_presets()
                for name in missing_names:
                    levels = canonical_map.get(name)
                    if levels:
                        extra_presets.extend(levels.values())
            except Exception:
                logger.exception('unexpected error')
            for preset in extra_presets:
                if preset not in all_presets:
                    all_presets.append(preset)
            for m in preset_map:
                mod_list = preset_map[m]
                for preset in extra_presets:
                    if preset not in mod_list:
                        mod_list.append(preset)

        scenario_names: List[str] = []
        for i, preset in enumerate(all_presets):
            raw = preset.get("SCENARIO_NAME", f"scenario_{i}")
            name = _PROFILE_ALIASES.get(raw, raw)
            if name not in scenario_names:
                scenario_names.append(name)
            preset["SCENARIO_NAME"] = name

        details: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}
        synergy_data: Dict[str, Dict[str, list]] = {
            name: {"roi": [], "metrics": []} for name in scenario_names
        }
        scenario_synergy: Dict[str, List[Dict[str, float]]] = {
            name: [] for name in scenario_names
        }

        tasks: list[
            tuple[int, asyncio.Task, str, str, str, Dict[str, Any], Dict[str, Any]]
        ] = []
        index = 0
        max_cpu = (
            max(float(p.get("CPU_LIMIT", 1)) for p in all_presets)
            if all_presets
            else 1.0
        )
        max_mem = (
            max(_parse_size(p.get("MEMORY_LIMIT", 0)) for p in all_presets)
            if all_presets
            else 0
        )
        max_gpu = (
            max(int(p.get("GPU_LIMIT", 0)) for p in all_presets) if all_presets else 0
        )
        total_cpu = multiprocessing.cpu_count() or 1
        if psutil:
            total_mem = psutil.virtual_memory().total
        else:
            total_mem = 0
        total_gpu = int(os.getenv("NUM_GPUS", "0"))
        workers_cpu = max(1, int(total_cpu / max(1.0, max_cpu)))
        workers_mem = (
            max(1, int(total_mem / max_mem)) if total_mem and max_mem else workers_cpu
        )
        workers_gpu = max(1, int(total_gpu / max_gpu)) if max_gpu else workers_cpu
        max_workers = min(workers_cpu, workers_mem, workers_gpu, len(sections)) or 1
        sem = asyncio.Semaphore(max_workers)

        all_diminished = True

        async def _gather_tasks() -> None:
            nonlocal index, all_diminished
            for module, sec_map in sections.items():
                tmp_dir = tempfile.mkdtemp(prefix="section_")
                shutil.copytree(repo_path, tmp_dir, dirs_exist_ok=True)
                builder = copy.deepcopy(base_builder)
                debugger = SelfDebuggerSandbox(
                    object(),
                    SelfCodingEngine(
                        CodeDB(), MenaceMemoryManager(), context_builder=builder
                    ),
                    context_builder=builder,
                )
                try:
                    for sec_name, lines in sec_map.items():
                        code_str = "\n".join(lines)
                        module_presets = preset_map.get(module, all_presets)
                        for p_idx, preset in enumerate(module_presets):
                            scenario = preset.get(
                                "SCENARIO_NAME", f"scenario_{p_idx}"
                            )
                            if scenario not in scenario_names:
                                scenario_names.append(scenario)
                                synergy_data.setdefault(
                                    scenario, {"roi": [], "metrics": []}
                                )
                                scenario_synergy.setdefault(scenario, [])
                            logger.info(
                                "simulate %s:%s under scenario %s",
                                module,
                                sec_name,
                                scenario,
                            )
                            for stub in input_stubs:
                                env_input = dict(preset)
                                env_input.update(stub)
                                _radar_track_module_usage(module)

                                for _ in range(3):
                                    result = simulate_execution_environment(
                                        code_str, env_input
                                    )
                                    if not result.get("risk_flags_triggered"):
                                        break
                                    debugger.analyse_and_fix()

                                await sem.acquire()

                                async def _task() -> tuple[
                                    Dict[str, Any],
                                    list[tuple[float, float, Dict[str, float]]],
                                    Dict[str, List[str]],
                                ]:
                                    try:
                                        return await _section_worker(
                                            code_str,
                                            env_input,
                                            tracker.diminishing(),
                                            runner_config,
                                        )
                                    except Exception as exc:
                                        record_error(exc, context_builder=builder)
                                        return {}, [], {}
                                    finally:
                                        sem.release()

                                fut = asyncio.create_task(_task())
                                tasks.append(
                                    (
                                        index,
                                        fut,
                                        module,
                                        sec_name,
                                        scenario,
                                        preset,
                                        stub,
                                    )
                                )
                                index += 1
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            sorted_tasks = sorted(tasks, key=lambda x: x[0])
            results = await asyncio.gather(*(t[1] for t in sorted_tasks))
            for (
                _,
                _fut,
                module,
                sec_name,
                scenario,
                preset,
                stub,
            ), (
                res,
                updates,
                cov_map,
            ) in zip(sorted_tasks, results):
                logger.info(
                    "result %s:%s scenario %s exit=%s",
                    module,
                    sec_name,
                    scenario,
                    res.get("exit_code"),
                )
                for prev, actual, metrics in updates:
                    extra = collect_plugin_metrics(plugins, prev, actual, metrics)
                    if extra:
                        metrics.update(extra)
                    specific = _scenario_specific_metrics(scenario, metrics)
                    if specific:
                        metrics.update(specific)
                    flags = _check_thresholds(scenario, metrics)
                    scenario_metrics = {
                        f"{k}:{scenario}": v for k, v in metrics.items()
                    }
                    pred_roi, _ = tracker.forecast()
                    tracker.record_metric_prediction("roi", pred_roi, actual)
                    tracker.update(
                        prev,
                        actual,
                        modules=[f"{module}:{sec_name}", scenario],
                        metrics={**metrics, **scenario_metrics, **flags},
                    )
                if updates:
                    funcs = _functions_for_module(cov_map, module)
                    _update_coverage(module, scenario, funcs)
                    final_roi = updates[-1][1]
                    final_metrics = updates[-1][2]
                    synergy_data[scenario]["roi"].append(final_roi)
                    synergy_data[scenario]["metrics"].append(final_metrics)
                    logger.info(
                        "scenario %s roi=%s metrics=%s",
                        scenario,
                        final_roi,
                        final_metrics,
                    )
                if return_details:
                    details.setdefault(module, {}).setdefault(scenario, []).append(
                        {
                            "section": sec_name,
                            "preset": preset,
                            "stub": stub,
                            "result": res,
                        }
                    )
                if res.get("exit_code") not in (0, None):
                    all_diminished = False

        await _gather_tasks()

        save_coverage_data()
        settings = SandboxSettings()
        verify_scenario_coverage(raise_on_missing=settings.fail_on_missing_scenarios)

        if all_diminished:
            combined: List[str] = []
            for sec_map in sections.values():
                for lines in sec_map.values():
                    combined.extend(lines)
            all_modules = list(sections)
            scenario_presets: Dict[str, Dict[str, Any]] = {}
            for preset in all_presets:
                scenario = preset.get("SCENARIO_NAME")
                if scenario and scenario not in scenario_presets:
                    scenario_presets[scenario] = preset

            for scenario, preset in scenario_presets.items():
                for stub in input_stubs:
                    env_input = dict(preset)
                    env_input.update(stub)
                    logger.info("combined run for scenario %s", scenario)
                    for m in all_modules:
                        _radar_track_module_usage(m)
                    res, updates, _ = await _section_worker(
                        "\n".join(combined),
                        env_input,
                        tracker.diminishing(),
                        runner_config,
                    )
                    for prev, actual, metrics in updates:
                        extra = collect_plugin_metrics(plugins, prev, actual, metrics)
                        if extra:
                            metrics.update(extra)
                        specific = _scenario_specific_metrics(scenario, metrics)
                        if specific:
                            metrics.update(specific)
                        flags = _check_thresholds(scenario, metrics)
                        scenario_metrics = {
                            f"{k}:{scenario}": v for k, v in metrics.items()
                        }
                        pred_roi, _ = tracker.forecast()
                        tracker.record_metric_prediction("roi", pred_roi, actual)
                        tracker.update(
                            prev,
                            actual,
                            modules=all_modules + [scenario],
                            metrics={**metrics, **scenario_metrics, **flags},
                        )
                    if updates:
                        roi_sum = sum(float(r) for r in synergy_data[scenario]["roi"])
                        metric_totals: Dict[str, float] = {}
                        metric_counts: Dict[str, int] = {}
                        for m_dict in synergy_data[scenario]["metrics"]:
                            for m, val in m_dict.items():
                                metric_totals[m] = metric_totals.get(m, 0.0) + float(
                                    val
                                )
                                metric_counts[m] = metric_counts.get(m, 0) + 1
                        avg_metrics = {
                            m: metric_totals[m] / metric_counts[m]
                            for m in metric_totals
                            if metric_counts.get(m)
                        }
                        combined_metrics = updates[-1][2]
                        synergy_metrics = {
                            f"synergy_{k}": combined_metrics.get(k, 0.0)
                            - avg_metrics.get(k, 0.0)
                            for k in set(avg_metrics) | set(combined_metrics)
                        }
                        synergy_metrics["synergy_roi"] = updates[-1][1] - roi_sum
                        synergy_metrics.setdefault(
                            "synergy_profitability", synergy_metrics["synergy_roi"]
                        )
                        synergy_metrics.setdefault(
                            "synergy_revenue", synergy_metrics["synergy_roi"]
                        )
                        synergy_metrics.setdefault(
                            "synergy_projected_lucrativity",
                            combined_metrics.get("projected_lucrativity", 0.0)
                            - avg_metrics.get("projected_lucrativity", 0.0),
                        )
                        for m in (
                            "maintainability",
                            "code_quality",
                            "network_latency",
                            "throughput",
                            "latency_error_rate",
                            "hostile_failures",
                            "misuse_failures",
                            "concurrency_throughput",
                        ):
                            synergy_metrics.setdefault(
                                f"synergy_{m}",
                                combined_metrics.get(m, 0.0) - avg_metrics.get(m, 0.0),
                            )
                        if hasattr(tracker, "register_metrics"):
                            tracker.register_metrics(*synergy_metrics.keys())
                        tracker.update(
                            roi_sum,
                            updates[-1][1],
                            modules=all_modules + [scenario],
                            metrics=synergy_metrics,
                        )
                        scenario_synergy.setdefault(scenario, []).append(
                            synergy_metrics
                        )
                        logger.info(
                            "scenario synergy %s metrics=%s",
                            scenario,
                            synergy_metrics,
                        )
                        if hasattr(tracker, "scenario_synergy"):
                            tracker.scenario_synergy.setdefault(scenario, []).append(
                                synergy_metrics
                            )
                    if return_details:
                        details.setdefault("_combined", {}).setdefault(
                            "all", []
                        ).append({"preset": preset, "stub": stub, "result": res})

        worst_label, _ = tracker.worst_scenario() if hasattr(tracker, "worst_scenario") else (None, 0.0)
        summary = save_scenario_summary(
            synergy_data,
            getattr(tracker, "scenario_roi_deltas", {}),
            worst_label if worst_label else None,
        )
        setattr(tracker, "scenario_summary", summary)
        if hasattr(tracker, "scenario_synergy"):
            tracker.scenario_synergy = scenario_synergy
        return (tracker, details) if return_details else tracker

    return asyncio.run(_run())


# ----------------------------------------------------------------------
def simulate_full_environment(preset: Dict[str, Any]) -> "ROITracker":
    """Execute an isolated sandbox run using ``preset`` environment vars.

    The path to ``sandbox_runner.py`` is resolved dynamically via
    :func:`resolve_path` so the function remains robust when the repository
    layout changes.
    """

    tmp_dir = tempfile.mkdtemp(prefix="full_env_")
    diagnostics: Dict[str, str] = {}
    try:
        repo_path = sandbox_config.get_sandbox_repo_path()
        runner_path = Path(path_for_prompt("sandbox_runner.py"))
        data_dir = Path(tmp_dir) / "data"
        env = os.environ.copy()
        env.update({k: str(v) for k, v in preset.items()})
        env.pop("SANDBOX_ENV_PRESETS", None)

        use_docker = str(os.getenv("SANDBOX_DOCKER", "0")).lower() not in {
            "0",
            "false",
            "no",
            "",
        }
        if use_docker and not _docker_available():
            logger.warning("docker unavailable; falling back")
            diagnostics["docker_error"] = "unavailable"
            use_docker = False
        os_type = env.get("OS_TYPE", "").lower()
        vm_used = False
        if use_docker:
            container_repo = "/repo"
            sandbox_tmp = "/sandbox_tmp"
            env["SANDBOX_DATA_DIR"] = f"{sandbox_tmp}/data"
            runner_in_container = str(
                Path(container_repo) / runner_path.relative_to(repo_path)
            )

            code = (
                "import subprocess, os\n"
                f"subprocess.run(['python', {runner_in_container!r}], cwd={container_repo!r})\n"
            )
            attempt = 0
            delay = _CREATE_BACKOFF_BASE
            while attempt < _CREATE_RETRY_LIMIT:
                try:
                    asyncio.run(
                        _execute_in_container(
                            code,
                            env,
                            mounts={
                                str(repo_path): container_repo,
                                tmp_dir: sandbox_tmp,
                            },
                            network_disabled=False,
                        )
                    )
                    break
                except DockerException as exc:
                    diagnostics["docker_error"] = str(exc)
                    logger.exception(
                        "docker execution failed: %s; cmd: docker run <image> python %s",
                        exc,
                        str(runner_path),
                    )
                    if attempt >= _CREATE_RETRY_LIMIT - 1:
                        logger.error("docker repeatedly failed; running locally")
                        diagnostics["local_execution"] = "docker"
                        use_docker = False
                        break
                    attempt += 1
                    time.sleep(delay)
                    delay *= 2

        if not use_docker and os_type in {"windows", "macos"}:
            vm = shutil.which("qemu-system-x86_64")
            vm_settings = preset.get("VM_SETTINGS", {})
            image = vm_settings.get(f"{os_type}_image") or vm_settings.get("image")
            memory = str(vm_settings.get("memory", "2G"))
            timeout = int(vm_settings.get("timeout", env.get("TIMEOUT", 300)))
            if vm and image:
                vm_repo = "/repo"
                sandbox_tmp = "/sandbox_tmp"
                env["SANDBOX_DATA_DIR"] = f"{sandbox_tmp}/data"
                runner_in_vm = str(
                    Path(vm_repo) / runner_path.relative_to(repo_path)
                )

                overlay = Path(tmp_dir) / "overlay.qcow2"
                _record_active_overlay(str(overlay.parent))
                try:
                    attempt = 0
                    delay = _CREATE_BACKOFF_BASE
                    while attempt < _CREATE_RETRY_LIMIT:
                        try:
                            subprocess.run(
                                [
                                    "qemu-img",
                                    "create",
                                    "-f",
                                    "qcow2",
                                    "-b",
                                    image,
                                    str(overlay),
                                ],
                                check=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                            cmd = [
                                vm,
                                "-m",
                                memory,
                                "-drive",
                                f"file={overlay},if=virtio",
                                "-virtfs",
                                f"local,path={repo_path},mount_tag=repo,security_model=none",
                                "-virtfs",
                                f"local,path={tmp_dir},mount_tag=sandbox_tmp,security_model=none",
                                "-nographic",
                                "-serial",
                                "stdio",
                                "-append",
                                f"python {runner_in_vm}",
                            ]
                            subprocess.run(
                                cmd,
                                check=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=timeout,
                            )
                            vm_used = True
                            break
                        except subprocess.TimeoutExpired:
                            diagnostics["vm_error"] = "timeout"
                            logger.error(
                                "VM execution timed out; cmd: %s", " ".join(cmd)
                            )
                            _CREATE_FAILURES["vm"] += 1
                            vm_used = False
                        except (FileNotFoundError, PermissionError) as exc:
                            diagnostics["vm_error"] = str(exc)
                            logger.error(
                                "VM setup failed: %s; cmd: %s", exc, " ".join(cmd)
                            )
                            _CREATE_FAILURES["vm"] += 1
                            vm_used = False
                            break
                        except Exception as exc:
                            diagnostics["vm_error"] = str(exc)
                            logger.exception(
                                "VM execution failed: %s; cmd: %s", exc, " ".join(cmd)
                            )
                            _CREATE_FAILURES["vm"] += 1
                            vm_used = False
                        if vm_used or attempt >= _CREATE_RETRY_LIMIT - 1:
                            if attempt >= _CREATE_RETRY_LIMIT - 1 and not vm_used:
                                diagnostics.setdefault("local_execution", "vm")
                            break
                        attempt += 1
                        time.sleep(delay)
                        delay *= 2
                finally:
                    try:
                        overlay.unlink()
                    except Exception:
                        logger.exception('unexpected error')
                    _remove_active_overlay(str(overlay.parent))
            else:
                logger.warning("qemu binary or VM image missing, running locally")
                if not vm:
                    diagnostics["vm_error"] = "qemu missing"
                elif not image:
                    diagnostics["vm_error"] = "image missing"

        if not use_docker and not vm_used:
            diagnostics.setdefault("local_execution", "vm")
            env["SANDBOX_DATA_DIR"] = str(data_dir)
            subprocess.run(
                ["python", str(runner_path)],
                cwd=repo_path,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        from menace.roi_tracker import ROITracker

        tracker = ROITracker()
        tracker.load_history(str(data_dir / "roi_history.json"))
        tracker.diagnostics = diagnostics
        return tracker
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ----------------------------------------------------------------------
def generate_workflows_for_modules(
    modules: Iterable[str],
    workflows_db: str | Path = "workflows.db",
    *,
    router: DBRouter | None = None,
    context_builder: ContextBuilder,
) -> list[int]:
    """Create workflows for ``modules`` including their dependencies.

    The repository import graph is analysed to determine required module
    dependencies. Modules connected through this graph are combined into a
    single multi‑step workflow preserving dependency order.

    Parameters
    ----------
    modules:
        Iterable of module names or file paths. Related modules will be grouped
        into a single workflow.
    workflows_db:
        Path to the workflows database. Defaults to ``"workflows.db"``.
    context_builder:
        :class:`~vector_service.context_builder.ContextBuilder` forwarded to
        orphan integration helpers.

    Returns
    -------
    list[int]
        The database IDs of the newly stored workflows.
    """

    from menace.task_handoff_bot import WorkflowDB, WorkflowRecord

    try:
        from dynamic_module_mapper import build_import_graph
        import networkx as nx
    except Exception:  # pragma: no cover - optional dependency
        build_import_graph = None  # type: ignore
        nx = None  # type: ignore

    wf_db = WorkflowDB(Path(workflows_db), router=router)
    ids: list[int] = []

    repo = Path(resolve_path(os.getenv("SANDBOX_REPO_PATH", ".")))
    graph = None
    if build_import_graph and nx:
        try:
            graph = build_import_graph(repo)
        except Exception:
            logger.exception("failed to build import graph")

    norm_paths = [Path(str(m)).with_suffix("").as_posix() for m in modules]

    processed: set[str] = set()
    if graph is not None:
        sub_nodes: set[str] = set()
        for mp in norm_paths:
            if mp in graph:
                sub_nodes.add(mp)
                try:
                    sub_nodes.update(nx.descendants(graph, mp))
                except Exception:
                    sub_nodes.add(mp)
            else:
                sub_nodes.add(mp)
        sub = graph.subgraph(sub_nodes).copy()
        for comp in nx.connected_components(sub.to_undirected()):
            sg = sub.subgraph(comp)
            try:
                order = list(nx.topological_sort(sg.reverse()))
            except Exception:
                order = list(comp)
            dotted_workflow = [n.replace("/", ".") for n in order]
            deps = [n.replace("/", ".") for n in order if n not in norm_paths]
            reasons = [
                f"{a.replace('/', '.')} -> {b.replace('/', '.')}" for a, b in sg.edges()
            ]
            title = dotted_workflow[-1] if dotted_workflow else "workflow"
            try:
                rec = WorkflowRecord(
                    workflow=dotted_workflow,
                    title=title,
                    dependencies=deps,
                    reasons=reasons,
                )
                wid = wf_db.add(rec)
                if wid is not None:
                    ids.append(wid)
                else:
                    logger.warning("duplicate workflow ignored for %s", title)
            except Exception:
                logger.exception("failed to store workflow for %s", title)
            processed.update(comp)

    for mp in norm_paths:
        if mp in processed:
            continue
        dotted = mp.replace("/", ".")
        try:
            rec = WorkflowRecord(workflow=[dotted], title=dotted)
            wid = wf_db.add(rec)
            if wid is not None:
                ids.append(wid)
            else:
                logger.warning("duplicate workflow ignored for %s", dotted)
        except Exception:
            logger.exception("failed to store workflow for %s", dotted)

    try:
        integrate_new_orphans(repo, router=router, context_builder=context_builder)
    except Exception:
        logger.exception("integrate_new_orphans after workflow generation failed")
    return ids


# ----------------------------------------------------------------------
def try_integrate_into_workflows(
    modules: Iterable[str],
    workflows_db: str | Path = "workflows.db",
    side_effects: Mapping[str, float] | None = None,
    *,
    side_effect_dev_multiplier: float | None = None,
    router: DBRouter | None = None,
    intent_clusterer: IntentClusterer | None = None,
    intent_k: float = 0.5,
    synergy_k: float | None = None,
    context_builder: ContextBuilder,
) -> list[int]:
    """Append orphan ``modules`` to related workflows if possible.

    The ``context_builder`` argument is mandatory and must not be ``None``.
    
    Modules are identified by their repository-relative paths to avoid
    filename collisions. Workflows already containing tasks from the same
    module group will receive the orphan module as an additional step. The list
    of updated workflow IDs is returned. Modules with heavy side-effect metrics
    are skipped based on a dynamic threshold derived from recent history and
    ``side_effect_dev_multiplier``.

    When an :class:`IntentClusterer` is available, each module is expanded by
    its synergy neighbourhood (via :func:`get_synergy_cluster`) and workflows
    whose intent vectors intersect with this cluster above a dynamic
    ``intent_k`` threshold are preferred. Modules without suitable intent
    matches are ignored. Synergy neighbourhood size is also recorded using the
    shared :class:`~self_improvement.baseline_tracker.BaselineTracker`.
    """

    if context_builder is None:
        raise ValueError("context_builder must not be None")

    from menace.task_handoff_bot import WorkflowDB
    from module_index_db import ModuleIndexDB
    try:  # pragma: no cover - service may be unavailable
        from self_test_service import SelfTestService
    except Exception:  # pragma: no cover - fallback for tests
        SelfTestService = None  # type: ignore
    from db_router import GLOBAL_ROUTER
    import ast
    import asyncio

    repo = Path(resolve_path(os.getenv("SANDBOX_REPO_PATH", ".")))

    side_effects = side_effects or {}
    try:
        from . import metrics_exporter as _me
    except Exception:
        import metrics_exporter as _me  # type: ignore
    names: dict[str, str] = {}
    tracker = _get_baseline_tracker()
    settings = SandboxSettings()
    if side_effect_dev_multiplier is None:
        side_effect_dev_multiplier = getattr(
            settings, "side_effect_dev_multiplier", 1.0
        )
    if synergy_k is None:
        synergy_k = getattr(settings, "synergy_dev_multiplier", 1.0)
    for m in modules:
        p = Path(m)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(repo)
            except Exception:
                rel = p.name
        else:
            rel = p
        rel_str = rel.as_posix()
        metric = side_effects.get(rel_str) or side_effects.get(str(p), 0)
        tracker.update(side_effects=metric)
        avg = tracker.get("side_effects")
        std = tracker.std("side_effects")
        threshold = avg + side_effect_dev_multiplier * std
        if metric > threshold:
            logger.info(
                "skipping %s due to side effects score %.2f", rel_str, metric
            )
            try:
                _me.orphan_modules_side_effects_total.inc()
            except Exception:
                logger.exception('unexpected error')
            continue
        names[rel_str] = rel.with_suffix("").as_posix().replace("/", ".")
    if not names:
        return []

    idx = ModuleIndexDB()
    grp_map: dict[str, int] = {}
    for mod in names:
        try:
            grp_map[mod] = idx.get(mod)
        except Exception:
            continue

    def _imports(path: Path) -> set[str]:
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            return set()
        imps: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imps.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imps.add(node.module.split(".")[0])
        return imps

    def _features(path: Path) -> tuple[set[int], set[str]]:
        """Return argument counts and semantic tags for functions in ``path``."""
        sigs: set[int] = set()
        tags: set[str] = set()
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            return sigs, tags
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                count = len(getattr(node.args, "args", []))
                count += len(getattr(node.args, "kwonlyargs", []))
                if getattr(node.args, "vararg", None):
                    count += 1
                if getattr(node.args, "kwarg", None):
                    count += 1
                sigs.add(count)
                tags.add(node.name.lower())
                doc = ast.get_docstring(node) or ""
                for word in re.findall(r"[a-zA-Z_]+", doc.lower()):
                    tags.add(word)
        return sigs, tags

    orphan_imports = {n: _imports(repo / n) for n in names}
    orphan_features = {n: _features(repo / n) for n in names}
    for mod, (_, tags) in orphan_features.items():
        if tags:
            try:
                idx.set_tags(mod, tags)
            except Exception:
                logger.exception('unexpected error')

    clusterer = intent_clusterer

    module_intents: dict[str, set[int]] = {}
    if clusterer is not None:
        for mod, dotted in names.items():
            expanded = {dotted}
            if _USE_MODULE_SYNERGY:
                try:
                    base_thr = tracker.get("synergy") + synergy_k * tracker.std("synergy")
                    cluster = get_synergy_cluster(dotted, base_thr)
                    hist = tracker.to_dict().get("synergy", [])
                    avg_s = tracker.get("synergy")
                    std_s = tracker.std("synergy")
                    metric = max(len(cluster) - 1, 0)
                    delta_s = metric - avg_s
                    tracker.update(synergy=metric)
                    if len(hist) >= 2 and delta_s < synergy_k * std_s:
                        cluster = {dotted}
                    expanded |= cluster
                except Exception:
                    logger.exception('unexpected error')
            ids: set[int] = set()
            for name in expanded:
                path = Path(resolve_path(repo / (name.replace(".", "/") + ".py")))
                ids.update(clusterer.clusters.get(str(path), []))
            module_intents[mod] = ids

    wf_db = WorkflowDB(Path(workflows_db), router=router)
    workflows = wf_db.fetch(limit=1000)
    workflow_intents: dict[int, set[int]] = {}
    if clusterer is not None:
        for wf in workflows:
            ids: set[int] = set()
            for step in wf.workflow:
                path = Path(resolve_path(repo / (step.replace(".", "/") + ".py")))
                ids.update(clusterer.clusters.get(str(path), []))
            workflow_intents[wf.wid] = ids
    updated: list[int] = []
    candidates: dict[int, list[str]] = {}

    for wf in workflows:
        step_groups: set[int] = set()
        step_imports: set[str] = set()
        step_names: list[str] = []
        step_sigs: set[int] = set()
        step_tags: set[str] = set()
        step_intent_ids = workflow_intents.get(wf.wid, set())
        for step in wf.workflow:
            mod = step.split(":")[0]
            step_names.append(mod)
            file = Path(resolve_path(repo / (mod.replace(".", "/") + ".py")))
            step_imports.update(_imports(file))
            try:
                rel = file.resolve().relative_to(repo)
                step_groups.add(idx.get(rel.as_posix()))
                sigs, tags = _features(file)
                step_sigs.update(sigs)
                if tags:
                    step_tags.update(tags)
                    try:
                        idx.set_tags(rel.as_posix(), tags)
                    except Exception:
                        logger.exception('unexpected error')
            except Exception:
                logger.exception('unexpected error')
        existing = set(wf.workflow)
        imported = {Path(s).stem for s in step_names}
        new_mods: list[str] = []
        for mod, gid in grp_map.items():
            dotted = names[mod]
            if dotted in existing:
                continue
            sigs, tags = orphan_features.get(mod, (set(), set()))
            intent_ids = module_intents.get(mod, set())
            if clusterer is not None and intent_ids:
                score = (
                    len(intent_ids & step_intent_ids) / len(intent_ids)
                    if intent_ids
                    else 0.0
                )
                hist = tracker.to_dict().get("intent_similarity", [])
                avg_i = tracker.get("intent_similarity")
                std_i = tracker.std("intent_similarity")
                delta_i = score - avg_i
                tracker.update(intent=score)
                if len(hist) >= 2:
                    if delta_i >= intent_k * std_i:
                        new_mods.append(mod)
                elif score >= intent_k:
                    new_mods.append(mod)
            elif (
                gid in step_groups
                or (tags & step_tags and sigs & step_sigs)
                or dotted.split(".")[0] in step_imports
                or orphan_imports[mod] & imported
            ):
                new_mods.append(mod)
        if new_mods:
            candidates[wf.wid] = new_mods

    if not candidates:
        return []

    test_paths = [
        (repo / m).as_posix() for mods in candidates.values() for m in mods
    ]
    builder = context_builder
    svc = SelfTestService(
        include_orphans=False,
        discover_orphans=False,
        discover_isolated=False,
        integration_callback=None,
        context_builder=builder,
    )
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        passed, _, _ = loop.run_until_complete(svc._test_orphan_modules(test_paths))
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    passed_set = {Path(p).resolve().as_posix() for p in passed}

    router = router or wf_db.router or GLOBAL_ROUTER
    conn = router.get_connection("workflows")
    for wf in workflows:
        mods = [
            m
            for m in candidates.get(wf.wid, [])
            if (repo / m).resolve().as_posix() in passed_set
        ]
        if not mods:
            continue
        for mod in mods:
            wf.workflow.append(names[mod])
        conn.execute(
            "UPDATE workflows SET workflow=? WHERE id=?",
            (",".join(wf.workflow), wf.wid),
        )
        updated.append(wf.wid)
    conn.commit()

    return updated


# ----------------------------------------------------------------------
def append_orphan_modules_to_workflows(
    modules: Iterable[str],
    workflows_db: str | Path = "workflows.db",
    *,
    router: DBRouter | None = None,
) -> list[int]:
    """Append modules to existing workflows and commit the updates.

    Parameters
    ----------
    modules:
        Iterable of module file paths. Paths may be absolute or repository
        relative.
    workflows_db:
        Location of the workflow database.

    Returns
    -------
    list[int]
        IDs of workflows that were updated.
    """

    from menace.task_handoff_bot import WorkflowDB
    from db_router import GLOBAL_ROUTER

    repo = Path(resolve_path(os.getenv("SANDBOX_REPO_PATH", ".")))
    steps: list[str] = []
    for m in modules:
        p = Path(m)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(repo)
            except Exception:
                rel = p
        else:
            rel = p
        steps.append(rel.with_suffix("").as_posix().replace("/", "."))

    if not steps:
        return []

    wf_db = WorkflowDB(Path(workflows_db), router=router)
    workflows = wf_db.fetch(limit=1000)
    updated: list[int] = []
    router = router or wf_db.router or GLOBAL_ROUTER
    conn = router.get_connection("workflows")
    for wf in workflows:
        changed = False
        existing = set(wf.workflow)
        for step in steps:
            if step not in existing:
                wf.workflow.append(step)
                existing.add(step)
                changed = True
        if changed:
            conn.execute(
                "UPDATE workflows SET workflow=? WHERE id=?",
                (",".join(wf.workflow), wf.wid),
            )
            updated.append(wf.wid)
    conn.commit()

    return updated


# ----------------------------------------------------------------------
def run_workflow_simulations(
    workflows_db: str | Path = "workflows.db",
    env_presets: List[Dict[str, Any]] | Mapping[str, List[Dict[str, Any]]] | None = None,
    *,
    dynamic_workflows: bool = False,
    module_algorithm: str = "greedy",
    module_threshold: float | None = None,
    module_semantic: bool = False,
    return_details: bool = False,
    tracker: "ROITracker" | None = None,
    foresight_tracker: "ForesightTracker" | None = None,
    router: DBRouter | None = None,
    runner_config: Dict[str, Any] | None = None,
    context_builder: ContextBuilder,
) -> "ROITracker" | tuple["ROITracker", Dict[str, list[Dict[str, Any]]]]:
    """Execute stored workflows under optional environment presets.

    ``module_algorithm`` selects the clustering method used when
    ``dynamic_workflows`` generates workflows from module groups. The
    ``module_threshold`` and ``module_semantic`` options mirror the
    parameters accepted by :func:`discover_module_groups`. When
    ``module_threshold`` is ``None``, the value is derived from the
    ``tracker``'s synergy baseline.
    """
    from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
    from menace.roi_tracker import ROITracker
    from menace.self_debugger_sandbox import SelfDebuggerSandbox
    from menace.self_coding_engine import SelfCodingEngine
    from menace.code_database import CodeDB
    from menace.menace_memory_manager import MenaceMemoryManager
    from sandbox_settings import SandboxSettings
    get_error_logger(context_builder)
    tracker = tracker or ROITracker()
    if module_threshold is None:
        k = getattr(SandboxSettings(), "synergy_dev_multiplier", 1.0)
        module_threshold = tracker.get("synergy") + k * tracker.std("synergy")
    try:
        from menace.environment_generator import _PROFILE_ALIASES, CANONICAL_PROFILES
    except Exception:  # pragma: no cover - environment generator optional
        _PROFILE_ALIASES = {}
        CANONICAL_PROFILES = [
            "high_latency_api",
            "hostile_input",
            "user_misuse",
            "concurrency_spike",
            "schema_drift",
            "flaky_upstream",
        ]

    if env_presets is None:
        if os.getenv("SANDBOX_GENERATE_PRESETS", "1") != "0":
            try:
                from menace.environment_generator import (
                    generate_presets,
                    generate_canonical_presets,
                )

                canonical = generate_canonical_presets()
                flat_canonical = [p for levels in canonical.values() for p in levels.values()]
                env_presets = flat_canonical + generate_presets()
            except Exception:
                env_presets = [{}]
        else:
            env_presets = [{}]

    base_builder = context_builder
    try:
        base_builder.refresh_db_weights()
    except Exception:
        pass

    if isinstance(env_presets, Mapping):
        preset_map: Dict[str, List[Dict[str, Any]]] = {}
        for k, v in env_presets.items():
            if isinstance(v, Mapping):
                flattened = [p for lst in v.values() for p in lst]
            else:
                flattened = list(v)
            preset_map[str(k)] = flattened
        all_presets: List[Dict[str, Any]] = [
            p for lst in preset_map.values() for p in lst
        ]
    else:
        preset_map = {}
        all_presets = list(env_presets)

    required = set(CANONICAL_PROFILES)
    existing = {
        _PROFILE_ALIASES.get(p.get("SCENARIO_NAME"), p.get("SCENARIO_NAME"))
        for p in all_presets
    }
    missing = required - existing
    if missing:
        try:
            from menace.environment_generator import generate_canonical_presets

            canonical_map: Dict[str, Dict[str, Any]] = {}
            for levels in generate_canonical_presets().values():
                for p in levels.values():
                    canonical_map[p["SCENARIO_NAME"]] = p
            for name in missing:
                preset = canonical_map.get(name)
                if not preset:
                    continue
                all_presets.append(preset)
                if isinstance(env_presets, Mapping):
                    preset_map.setdefault(name, []).append(preset)
        except Exception:
            logger.exception('unexpected error')

    wf_db = WorkflowDB(Path(workflows_db), router=router)
    workflows = wf_db.fetch()
    if dynamic_workflows or not workflows:
        from dynamic_module_mapper import discover_module_groups, dotify_groups

        groups = dotify_groups(
            discover_module_groups(
                Path.cwd(),
                algorithm=module_algorithm,
                threshold=module_threshold,
                use_semantic=module_semantic,
            )
        )
        if dynamic_workflows:
            try:
                from menace.module_index_db import ModuleIndexDB

                idx = ModuleIndexDB()
                for mod, gid in idx._map.items():
                    dotted = Path(mod).with_suffix("").as_posix().replace("/", ".")
                    present = any(dotted in m for m in groups.values())
                    if not present:
                        groups.setdefault(str(gid), []).append(dotted)
            except Exception:
                logger.exception('unexpected error')
        workflows = [
            WorkflowRecord(workflow=mods, title=f"workflow_{gid}", wid=i + 1)
            for i, (gid, mods) in enumerate(sorted(groups.items()))
        ]

    if isinstance(env_presets, Mapping):
        try:
            from menace.environment_generator import (
                generate_presets,
                suggest_profiles_for_module,
            )

            modules = {step.split(":")[0] for wf in workflows for step in wf.workflow}
            for module in modules:
                if module in preset_map:
                    continue
                profiles = suggest_profiles_for_module(module)
                if profiles:
                    new_presets = generate_presets(profiles=profiles)
                    preset_map[module] = new_presets
                    all_presets.extend(new_presets)
        except Exception:
            logger.exception('unexpected error')

    def _module_from_step(step: str) -> str:
        if ":" in step:
            return step.split(":", 1)[0]
        step = step.replace("/", ".")
        if "." in step:
            return step.rsplit(".", 1)[0]
        if importlib.util.find_spec(step) is not None:
            return step
        raise ValueError(f"Workflow step '{step}' must include a module path")

    scenario_names: List[str] = []
    for i, p in enumerate(all_presets):
        raw = p.get("SCENARIO_NAME", f"scenario_{i}")
        name = _PROFILE_ALIASES.get(raw, raw)
        p["SCENARIO_NAME"] = name
        if name not in scenario_names:
            scenario_names.append(name)
    for name in required:
        if name not in scenario_names:
            scenario_names.append(name)

    module_names = {_module_from_step(step) for wf in workflows for step in wf.workflow}
    coverage_summary: Dict[str, Dict[str, bool]] = {
        mod: {scen: False for scen in scenario_names} for mod in module_names
    }

    async def _run() -> (
        "ROITracker" | tuple["ROITracker", Dict[str, list[Dict[str, Any]]]]
    ):
        details: Dict[str, list[Dict[str, Any]]] = {}

        tasks: list[tuple[int, asyncio.Task, int, str, Dict[str, Any], str, str]] = []
        index = 0
        synergy_data: Dict[str, Dict[str, list]] = {
            name: {"roi": [], "metrics": []} for name in scenario_names
        }
        combined_results: Dict[str, Dict[str, Any]] = {}

        def _wf_snippet(steps: list[str]) -> str:
            imports: list[str] = []
            calls: list[str] = []
            for idx, step in enumerate(steps):
                alias = f"_wf_{idx}"
                mod: str
                func: str | None = None
                if ":" in step:
                    mod, func = step.split(":", 1)
                elif importlib.util.find_spec(step) is not None:
                    mod = step
                elif "." in step:
                    mod, func = step.rsplit(".", 1)
                elif "/" in step:
                    mod = step.replace("/", ".")
                else:
                    raise ValueError(
                        f"Workflow step '{step}' must include a module path"
                    )
                if importlib.util.find_spec(mod) is None:
                    raise ValueError(
                        f"Module '{mod}' for workflow step '{step}' not found"
                    )
                if func is None:
                    imports.append(f"import {mod} as {alias}")
                    calls.append(f"getattr({alias}, 'main', lambda: None)()")
                else:
                    imports.append(f"from {mod} import {func} as {alias}")
                    calls.append(f"{alias}()")
            if not calls:
                return "\n".join(f"# {s}" for s in steps) + "\npass\n"
            return "\n".join(imports + [""] + calls) + "\n"

        wf_aggregates: Dict[tuple[int, str], Dict[str, Any]] = {}

        for wf in workflows:
            for step in wf.workflow:
                snippet = _wf_snippet([step])
                builder = copy.deepcopy(base_builder)
                debugger = SelfDebuggerSandbox(
                    object(),
                    SelfCodingEngine(
                        CodeDB(), MenaceMemoryManager(), context_builder=builder
                    ),
                    context_builder=builder,
                )
                mod_name = _module_from_step(step)
                for preset in all_presets:
                    scenario = preset.get("SCENARIO_NAME", "")
                    env_input = dict(preset)
                    _radar_track_module_usage(mod_name)
                    for _ in range(3):
                        result = simulate_execution_environment(snippet, env_input)
                        if not result.get("risk_flags_triggered"):
                            break
                        debugger.analyse_and_fix()
                    fut = asyncio.create_task(
                        _section_worker(
                            snippet,
                            env_input,
                            tracker.diminishing(),
                            runner_config,
                        )
                    )
                    tasks.append((index, fut, wf.wid, scenario, preset, step, mod_name))
                    index += 1

        for _, fut, wid, scenario, preset, module, mod_name in sorted(tasks, key=lambda x: x[0]):
            res, updates, cov_map = await fut
            if updates:
                if mod_name in coverage_summary and scenario in coverage_summary[mod_name]:
                    coverage_summary[mod_name][scenario] = True
                funcs = _functions_for_module(cov_map, mod_name)
                _update_coverage(mod_name, scenario, funcs)
            for prev, actual, metrics in updates:
                specific = _scenario_specific_metrics(scenario, metrics)
                if specific:
                    metrics.update(specific)
                scenario_metrics = {f"{k}:{scenario}": v for k, v in metrics.items()}
                pred_roi, _ = tracker.forecast()
                tracker.record_metric_prediction("roi", pred_roi, actual)
                tracker.update(
                    prev,
                    actual,
                    modules=[module, scenario],
                    metrics={**metrics, **scenario_metrics},
                )
            if updates:
                final_roi = updates[-1][1]
                final_metrics = updates[-1][2]
                agg = wf_aggregates.setdefault((wid, scenario), {
                    "roi": 0.0,
                    "metrics_totals": {},
                    "metrics_counts": {},
                })
                agg["roi"] += final_roi
                for m, val in final_metrics.items():
                    agg["metrics_totals"][m] = agg["metrics_totals"].get(m, 0.0) + float(val)
                    agg["metrics_counts"][m] = agg["metrics_counts"].get(m, 0) + 1
            if return_details:
                details.setdefault(str(wid), []).append(
                    {"preset": preset, "module": module, "result": res}
                )

        for wf in workflows:
            for scenario in scenario_names:
                agg = wf_aggregates.get((wf.wid, scenario))
                if not agg:
                    continue
                totals = agg["metrics_totals"]
                counts = agg["metrics_counts"]
                avg_metrics = {
                    m: totals[m] / counts[m] for m in totals if counts.get(m)
                }
                specific = _scenario_specific_metrics(scenario, avg_metrics)
                if specific:
                    avg_metrics.update(specific)
                scenario_metrics = {f"{k}:{scenario}": v for k, v in avg_metrics.items()}
                pred_roi, _ = tracker.forecast()
                tracker.record_metric_prediction("roi", pred_roi, agg["roi"])
                tracker.update(
                    0.0,
                    agg["roi"],
                    modules=[f"workflow_{wf.wid}", scenario],
                    metrics={**avg_metrics, **scenario_metrics},
                )
                synergy_data[scenario]["roi"].append(agg["roi"])
                synergy_data[scenario]["metrics"].append(avg_metrics)

        combined_steps: list[str] = []
        for wf in workflows:
            combined_steps.extend(wf.workflow)
        combined_snippet = _wf_snippet(combined_steps)
        workflow_modules = [f"workflow_{wf.wid}" for wf in workflows]
        for preset in all_presets:
            scenario = preset.get("SCENARIO_NAME", "")
            env_input = dict(preset)
            for m in workflow_modules:
                _radar_track_module_usage(m)
            res, updates, _ = await _section_worker(
                combined_snippet,
                env_input,
                tracker.diminishing(),
                runner_config,
            )
            for prev, actual, metrics in updates:
                specific = _scenario_specific_metrics(scenario, metrics)
                if specific:
                    metrics.update(specific)
                scenario_metrics = {f"{k}:{scenario}": v for k, v in metrics.items()}
                pred_roi, _ = tracker.forecast()
                tracker.record_metric_prediction("roi", pred_roi, actual)
                tracker.update(
                    prev,
                    actual,
                    modules=["all_workflows", scenario],
                    metrics={**metrics, **scenario_metrics},
                )
            if updates:
                combined_results[scenario] = {
                    "roi": updates[-1][1],
                    "metrics": updates[-1][2],
                }
                roi_sum = sum(float(r) for r in synergy_data[scenario]["roi"])
                metric_totals: Dict[str, float] = {}
                metric_counts: Dict[str, int] = {}
                for m_dict in synergy_data[scenario]["metrics"]:
                    for m, val in m_dict.items():
                        metric_totals[m] = metric_totals.get(m, 0.0) + float(val)
                        metric_counts[m] = metric_counts.get(m, 0) + 1
                avg_metrics = {
                    m: metric_totals[m] / metric_counts[m]
                    for m in metric_totals
                    if metric_counts.get(m)
                }
                combined_metrics = combined_results[scenario]["metrics"]
                synergy_metrics = {
                    f"synergy_{k}": combined_metrics.get(k, 0.0)
                    - avg_metrics.get(k, 0.0)
                    for k in set(avg_metrics) | set(combined_metrics)
                }
                synergy_metrics["synergy_roi"] = (
                    combined_results[scenario]["roi"] - roi_sum
                )
                if "synergy_profitability" not in synergy_metrics:
                    synergy_metrics["synergy_profitability"] = synergy_metrics[
                        "synergy_roi"
                    ]
                if "synergy_revenue" not in synergy_metrics:
                    synergy_metrics["synergy_revenue"] = synergy_metrics["synergy_roi"]
                if "synergy_projected_lucrativity" not in synergy_metrics:
                    synergy_metrics["synergy_projected_lucrativity"] = (
                        combined_metrics.get("projected_lucrativity", 0.0)
                        - avg_metrics.get("projected_lucrativity", 0.0)
                    )
                for m in (
                    "maintainability",
                    "code_quality",
                    "network_latency",
                    "throughput",
                    "latency_error_rate",
                    "hostile_failures",
                    "misuse_failures",
                    "concurrency_throughput",
                ):
                    synergy_metrics.setdefault(
                        f"synergy_{m}",
                        combined_metrics.get(m, 0.0) - avg_metrics.get(m, 0.0),
                    )
                if hasattr(tracker, "register_metrics"):
                    tracker.register_metrics(*synergy_metrics.keys())
                tracker.update(
                    roi_sum,
                    combined_results[scenario]["roi"],
                    modules=workflow_modules + [scenario],
                    metrics=synergy_metrics,
                )
            if return_details:
                details.setdefault("_combined", []).append(
                    {"preset": preset, "result": res}
                )

        for mod, scen_map in coverage_summary.items():
            missing = [s for s, done in scen_map.items() if not done]
            if missing:
                logger.warning("module %s missing scenarios: %s", mod, ", ".join(missing))
            else:
                logger.info("module %s covered in all scenarios", mod)
        logger.info("scenario coverage summary: %s", coverage_summary)
        try:
            if any(not all(scen_map.values()) for scen_map in coverage_summary.values()):
                dispatch_alert("sandbox_coverage", {"coverage": coverage_summary})
        except Exception:
            logger.exception("failed to dispatch coverage alert")
        save_coverage_data()
        settings = SandboxSettings()
        verify_scenario_coverage(raise_on_missing=settings.fail_on_missing_scenarios)
        worst_label, _ = tracker.worst_scenario() if hasattr(tracker, "worst_scenario") else (None, 0.0)
        summary = save_scenario_summary(
            synergy_data,
            getattr(tracker, "scenario_roi_deltas", {}),
            worst_label if worst_label else None,
        )
        setattr(tracker, "scenario_summary", summary)
        setattr(tracker, "coverage_summary", coverage_summary)
        return (tracker, details) if return_details else tracker

    return asyncio.run(_run())


# ----------------------------------------------------------------------
def auto_include_modules(
    modules: Iterable[str],
    recursive: bool = False,
    validate: bool = False,
    *,
    router: DBRouter | None = None,
    context_builder: ContextBuilder,
) -> tuple["ROITracker", Dict[str, list[str]]]:
    """Automatically include ``modules`` into the workflow system.

    The ``context_builder`` argument is mandatory and must not be ``None``.

    The helper performs three steps:

    #. Generate simple workflows for each provided module via
       :func:`generate_workflows_for_modules`.
    #. Attempt to integrate those modules into existing workflows using
       :func:`try_integrate_into_workflows` so that related workflows gain the
       new steps.
    #. Execute :func:`run_workflow_simulations` to evaluate the newly
       incorporated workflows.

    When ``recursive`` or ``SANDBOX_RECURSIVE_ORPHANS`` is enabled the helper
    expands the initial module list by following local imports using
    :func:`sandbox_runner.dependency_utils.collect_local_dependencies` and by
    merging orphan dependencies returned from
    :func:`sandbox_runner.discover_recursive_orphans`. Redundancy analysis via
    :func:`orphan_analyzer.classify_module` now occurs *after* optional self
    testing so that modules are executed before being marked as redundant.

    Isolated modules discovered via ``scripts.discover_isolated_modules`` are
    always merged into the candidate set when isolated inclusion is enabled or
    recursive processing is requested.

    Each candidate module—along with any newly discovered paths—is executed via
    :class:`self_test_service.SelfTestService` with ``recursive_orphans=True``,
    ``discover_isolated=True`` and ``auto_include_isolated=True`` when
    validation is requested or the module was newly discovered. Only modules
    that pass these self tests are considered for integration. Modules that fail
    validation or are classified as redundant after testing are recorded in
    ``sandbox_data/orphan_modules.json`` while passing entries are pruned from
    that cache. Modules whose simulated ROI increase is below
    ``SandboxSettings.min_integration_roi`` are discarded prior to workflow
    integration. Redundant modules are integrated only when
    :class:`sandbox_settings.SandboxSettings` sets ``test_redundant_modules``.

    The return value from :func:`run_workflow_simulations` is forwarded to the
    caller alongside a mapping of tested modules and the resulting ROI metrics
    are saved to ``SANDBOX_DATA_DIR/roi_history.json`` for later analysis by
    the self-improvement engine.
    """

    if context_builder is None:
        raise ValueError("context_builder must not be None")

    import os
    from pathlib import Path
    import orphan_analyzer
    import json
    from sandbox_settings import SandboxSettings
    from .dependency_utils import collect_local_dependencies
    from db_router import GLOBAL_ROUTER
    try:
        from . import metrics_exporter as _me
    except Exception:  # pragma: no cover - fallback import
        import metrics_exporter as _me  # type: ignore

    # Retrieve modules flagged by relevancy radar and track retire candidates
    retired_flags: set[str] = set()
    try:  # pragma: no cover - optional dependency
        from relevancy_radar import flagged_modules as _rr_flagged

        for _mod, _flag in _rr_flagged().items():
            if _flag == "retire":
                rel = _mod.replace(".", "/")
                if not rel.endswith(".py"):
                    rel += ".py"
                retired_flags.add(Path(rel).as_posix())
    except Exception:
        logger.exception('unexpected error')

    mod_paths = {Path(m).as_posix() for m in modules}
    isolated_mods: set[str] = set()

    get_error_logger(context_builder)
    data_dir = _env_path("SANDBOX_DATA_DIR", "sandbox_data")
    map_file = data_dir / "module_map.json"
    existing_mods: set[str] = set()
    try:
        mapping = json.loads(map_file.read_text()) if map_file.exists() else {}
        if isinstance(mapping, dict):
            if isinstance(mapping.get("modules"), dict):
                existing_mods = set(mapping["modules"])
            else:
                existing_mods = set(mapping)
    except Exception:
        logger.exception('unexpected error')
    mod_paths.difference_update(existing_mods)
    redundant_mods: dict[str, str] = {}

    settings = SandboxSettings()
    min_roi = getattr(settings, "min_integration_roi", 0.0)

    repo = Path(resolve_path(os.getenv("SANDBOX_REPO_PATH", ".")))

    recursive_orphans = recursive or os.getenv("SANDBOX_RECURSIVE_ORPHANS", "1") not in {"0", "false", "False"}

    traces = None
    if recursive or recursive_orphans:
        try:
            import inspect

            for frame in inspect.stack():  # pragma: no cover - best effort
                ctx = frame.frame.f_locals.get("ctx")
                if ctx is not None and hasattr(ctx, "orphan_traces"):
                    traces = ctx.orphan_traces
                    break
        except Exception:  # pragma: no cover - best effort
            traces = None

        try:
            initial = (
                {Path(m).as_posix(): traces.get(Path(m).as_posix(), {}).get("parents", [])
                 for m in modules}
                if traces is not None
                else None
            )

            def _on_module(rel: str, _path: Path, parents: list[str]) -> None:
                if traces is None or not parents:
                    return
                entry = traces.setdefault(rel, {"parents": []})
                entry["parents"] = list(
                    dict.fromkeys(entry.get("parents", []) + parents)
                )

            def _on_dep(dep_rel: str, _parent_rel: str, chain: list[str]) -> None:
                if traces is None or not chain:
                    return
                entry = traces.setdefault(dep_rel, {"parents": []})
                entry["parents"] = list(
                    dict.fromkeys(entry.get("parents", []) + chain)
                )

            deps = collect_local_dependencies(
                modules,
                initial_parents=initial,
                on_module=_on_module if traces is not None else None,
                on_dependency=_on_dep if traces is not None else None,
                max_depth=getattr(settings, "max_recursion_depth", None),
            )
            mod_paths.update(deps)
            mod_paths.difference_update(existing_mods)
        except Exception:
            logger.exception('unexpected error')

    candidate_paths = set(mod_paths)
    evaluated: set[str] = set()
    if recursive_orphans:
        try:
            import importlib, sys

            od = (
                importlib.reload(sys.modules["sandbox_runner.orphan_discovery"])
                if "sandbox_runner.orphan_discovery" in sys.modules
                else importlib.import_module("sandbox_runner.orphan_discovery")
            )
            mapping = od.discover_recursive_orphans(str(repo))
            for name, info in mapping.items():
                path = Path(name.replace(".", "/")).with_suffix(".py").as_posix()
                if path in existing_mods:
                    continue
                candidate_paths.add(path)
                evaluated.add(path)
                if info.get("redundant"):
                    cls = info.get("classification", "redundant")
                    redundant_mods[path] = cls
                    mod_paths.discard(path)
                else:
                    mod_paths.add(path)
        except Exception:
            logger.exception('unexpected error')

    include_isolated = getattr(settings, "auto_include_isolated", True)
    if include_isolated or recursive:
        try:
            from scripts.discover_isolated_modules import discover_isolated_modules

            isolated = discover_isolated_modules(repo, recursive=True)
            for rel in isolated:
                path = Path(rel)
                rel_posix = path.as_posix()
                if rel_posix in existing_mods:
                    continue
                mod_paths.add(rel_posix)
                isolated_mods.add(rel_posix)
                try:
                    _me.isolated_modules_discovered_total.inc()
                except Exception:
                    logger.exception('unexpected error')
        except Exception:
            logger.exception('unexpected error')

    if retired_flags:
        retired_present = sorted(retired_flags.intersection(mod_paths))
        if retired_present:
            for m in retired_present:
                logger.info("skipping %s due to retirement flag", m)
            mod_paths.difference_update(retired_present)
            candidate_paths.difference_update(retired_present)

    new_paths = set(mod_paths) - candidate_paths

    mods = sorted(mod_paths)
    passed_mods: list[str] = []
    failed_mods: list[str] = []
    heavy_side_effects: dict[str, float] = {}

    builder = context_builder
    for mod in mods:
        path = repo / mod
        need_validate = validate or mod in new_paths
        if need_validate:
            try:
                from self_test_service import SelfTestService

                svc = SelfTestService(
                    pytest_args=mod,
                    include_orphans=False,
                    discover_orphans=False,
                    discover_isolated=True,
                    recursive_orphans=True,
                    recursive_isolated=settings.recursive_isolated,
                    auto_include_isolated=True,
                    include_redundant=settings.test_redundant_modules,
                    disable_auto_integration=True,
                    context_builder=builder,
                )
                res = svc.run_once()
                result = res[0] if isinstance(res, tuple) else res
                if not result.get("failed"):
                    passed_mods.append(mod)
                else:
                    failed_mods.append(mod)
            except Exception:
                failed_mods.append(mod)
        else:
            passed_mods.append(mod)

        try:
            if hasattr(orphan_analyzer, "classify_module"):
                res = orphan_analyzer.classify_module(path, include_meta=True)
                cls, meta = res if isinstance(res, tuple) else (res, {})
                score = float(meta.get("side_effects", 0))
                tracker = _get_baseline_tracker()
                tracker.update(side_effects=score)
                avg = tracker.get("side_effects")
                std = tracker.std("side_effects")
                threshold = avg + getattr(
                    settings, "side_effect_dev_multiplier", 1.0
                ) * std
                if score > threshold:
                    logger.info(
                        "skipping %s due to side effects score %.2f", mod, score
                    )
                    heavy_side_effects[mod] = score
                    if mod in passed_mods:
                        passed_mods.remove(mod)
                    failed_mods.append(mod)
                elif cls != "candidate":
                    redundant_mods[mod] = cls
            elif hasattr(orphan_analyzer, "analyze_redundancy"):
                if orphan_analyzer.analyze_redundancy(path):
                    redundant_mods[mod] = "redundant"
        except Exception:
            logger.exception('unexpected error')

    cache = _env_path("SANDBOX_DATA_DIR", "sandbox_data") / "orphan_modules.json"
    try:
        existing = json.loads(cache.read_text()) if cache.exists() else {}
    except Exception:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    # Remove passing modules from the orphan cache
    for m in passed_mods:
        if m in existing and m not in redundant_mods and m not in failed_mods:
            del existing[m]
    # Record redundant, failed and heavy side-effect modules
    for m in set(redundant_mods) | set(failed_mods) | set(heavy_side_effects):
        info = existing.get(m, {})
        if m in redundant_mods:
            cls = redundant_mods[m]
            info["classification"] = cls
            info["redundant"] = cls != "candidate"
        if m in failed_mods:
            info["failed"] = True
            if "classification" not in info:
                info["classification"] = "failed"
        if m in heavy_side_effects:
            info["reason"] = "heavy_side_effects"
            info["side_effects"] = heavy_side_effects[m]
        existing[m] = info
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(existing, indent=2))
    except Exception:
        logger.exception('unexpected error')

    try:
        from sandbox_runner.orphan_discovery import append_orphan_classifications

        class_entries = {
            m: existing[m]
            for m in set(redundant_mods) | set(failed_mods) | set(heavy_side_effects)
        }
        append_orphan_classifications(repo, class_entries)
    except Exception:
        logger.exception('unexpected error')

    mods = [
        m
        for m in passed_mods
        if settings.test_redundant_modules or m not in redundant_mods
    ]
    tested = {
        "added": list(passed_mods),
        "failed": list(failed_mods),
        "redundant": list(redundant_mods),
    }

    # Evaluate preliminary ROI for each candidate module and exclude those with
    # non-positive projections.
    low_roi: list[str] = []
    try:  # pragma: no cover - best effort
        from menace.roi_tracker import ROITracker

        data_dir = _env_path("SANDBOX_DATA_DIR", "sandbox_data")
        hist_path = data_dir / "roi_history.json"
        tracker = ROITracker()
        if hist_path.exists():
            try:
                tracker.load_history(str(hist_path))
            except Exception:
                logger.exception('unexpected error')
        for m in list(mods):
            roi_val = sum(tracker.module_deltas.get(m, []))
            if roi_val < min_roi:
                low_roi.append(m)
        if low_roi:
            mods = [m for m in mods if m not in low_roi]
            tested.setdefault("low_roi", []).extend(low_roi)
            # Update orphan cache with low ROI reason
            cache = data_dir / "orphan_modules.json"
            try:
                existing = json.loads(cache.read_text()) if cache.exists() else {}
            except Exception:
                existing = {}
            if not isinstance(existing, dict):
                existing = {}
            for m in low_roi:
                info = existing.get(m, {})
                info["reason"] = "low_roi"
                existing[m] = info
                if m in tested.get("added", []):
                    tested["added"].remove(m)
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(existing, indent=2))
            except Exception:
                logger.exception('unexpected error')
    except Exception:
        logger.exception('unexpected error')

    baseline_result = run_workflow_simulations(router=router, context_builder=context_builder)
    baseline_tracker = (
        baseline_result[0] if isinstance(baseline_result, tuple) else baseline_result
    )
    data_dir = _env_path("SANDBOX_DATA_DIR", "sandbox_data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        baseline_tracker.save_history(str(data_dir / "roi_history.json"))
    except Exception:
        logger.exception('unexpected error')
    baseline_roi = sum(float(r) for r in getattr(baseline_tracker, "roi_history", []))

    if not mods:
        return baseline_tracker, tested

    accepted: list[str] = []
    low_roi_mods: list[str] = []
    for mod in mods:
        ids = generate_workflows_for_modules([mod], router=router, context_builder=context_builder)
        result = run_workflow_simulations(router=router, context_builder=context_builder)
        tracker = result[0] if isinstance(result, tuple) else result
        new_roi = sum(float(r) for r in getattr(tracker, "roi_history", []))
        delta = new_roi - baseline_roi
        if delta >= min_roi:
            accepted.append(mod)
            baseline_roi = new_roi
            baseline_tracker = tracker
        else:
            low_roi_mods.append(mod)
            try:
                router = router or GLOBAL_ROUTER
                conn = router.get_connection("workflows")
                for wid in ids:
                    conn.execute("DELETE FROM workflows WHERE id=?", (wid,))
                conn.commit()
            except Exception:
                logger.exception('unexpected error')
    if low_roi_mods:
        tested.setdefault("low_roi", []).extend(low_roi_mods)
        tested["added"] = [m for m in tested.get("added", []) if m not in low_roi_mods]
        cache = data_dir / "orphan_modules.json"
        try:
            existing = json.loads(cache.read_text()) if cache.exists() else {}
        except Exception:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        for m in low_roi_mods:
            info = existing.get(m, {})
            info["reason"] = "low_roi"
            existing[m] = info
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(existing, indent=2))
        except Exception:
            logger.exception('unexpected error')
    mods = accepted
    if not mods:
        return baseline_tracker, tested

    pre_integrate_roi = baseline_roi
    try_integrate_into_workflows(mods, router=router, context_builder=context_builder)
    for mod in mods:
        if mod in isolated_mods:
            try:
                _me.isolated_modules_integrated_total.inc()
            except Exception:
                logger.exception('unexpected error')

    data_dir = _env_path("SANDBOX_DATA_DIR", "sandbox_data")
    map_file = data_dir / "module_map.json"
    try:
        orig_map_text = map_file.read_text() if map_file.exists() else None
        existing = json.loads(orig_map_text) if orig_map_text else {}
    except Exception:
        existing = {}
        orig_map_text = None
    if (
        isinstance(existing, dict)
        and isinstance(existing.get("modules"), dict)
    ):
        module_map = existing["modules"]
        container: dict[str, Any] = existing
    else:
        module_map = existing if isinstance(existing, dict) else {}
        container = module_map
    next_id = (
        max((int(v) for v in module_map.values() if isinstance(v, int)), default=0)
        + 1
    )
    for mod in mods:
        if mod not in module_map:
            module_map[mod] = next_id
            next_id += 1
    try:
        map_file.parent.mkdir(parents=True, exist_ok=True)
        map_file.write_text(json.dumps(container, indent=2))
    except Exception:
        logger.exception('unexpected error')

    result = run_workflow_simulations(router=router, context_builder=context_builder)
    tracker = result[0] if isinstance(result, tuple) else result
    new_roi = sum(float(r) for r in getattr(tracker, "roi_history", []))
    if new_roi < pre_integrate_roi:
        try:
            if orig_map_text is not None:
                map_file.write_text(orig_map_text)
            elif map_file.exists():
                map_file.unlink()
        except Exception:
            logger.exception('unexpected error')
        cache = data_dir / "orphan_modules.json"
        try:
            existing_cache = json.loads(cache.read_text()) if cache.exists() else {}
        except Exception:
            existing_cache = {}
        if not isinstance(existing_cache, dict):
            existing_cache = {}
        for m in mods:
            info = existing_cache.get(m, {})
            info["rejected"] = True
            info.setdefault("classification", "rejected")
            existing_cache[m] = info
            if m in tested.get("added", []):
                tested["added"].remove(m)
        tested["rejected"] = list(mods)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(existing_cache, indent=2))
        except Exception:
            logger.exception('unexpected error')
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            baseline_tracker.save_history(str(data_dir / "roi_history.json"))
        except Exception:
            logger.exception('unexpected error')
        return baseline_tracker, tested

    try:
        tracker.cluster_map.update(module_map)
    except Exception:
        logger.exception('unexpected error')
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        tracker.save_history(str(data_dir / "roi_history.json"))
    except Exception:
        logger.exception('unexpected error')
    return tracker, tested


# ----------------------------------------------------------------------
def discover_and_integrate_orphans(
    repo: Path,
    *,
    router: DBRouter | None = None,
    context_builder: ContextBuilder,
) -> list[str]:
    """Discover orphan modules and integrate them into the workflow system.

    Parameters
    ----------
    repo:
        Root directory of the repository to scan.
    router:
        Optional :class:`db_router.DBRouter` forwarded to
        :func:`auto_include_modules`.
    context_builder:
        :class:`~vector_service.context_builder.ContextBuilder` forwarded to
        :func:`integrate_and_graph_orphans`.

    Returns
    -------
    list[str]
        Repository-relative paths of modules successfully added.
    """

    try:
        _, tested, _, _, _ = integrate_and_graph_orphans(
            repo, logger=logger, router=router, context_builder=context_builder
        )
    except Exception:  # pragma: no cover - best effort
        logger.exception("discover_and_integrate_orphans failed")
        return []

    return tested.get("added", [])


# ----------------------------------------------------------------------
def integrate_new_orphans(
    repo_path: str | Path,
    *,
    router: DBRouter | None = None,
    context_builder: ContextBuilder,
) -> list[str]:
    """Discover and integrate new orphan modules.

    This wrapper exists for backward compatibility and forwards to
    :func:`discover_and_integrate_orphans`.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to scan.
    router:
        Optional :class:`db_router.DBRouter` forwarded to
        :func:`auto_include_modules`.
    context_builder:
        :class:`~vector_service.context_builder.ContextBuilder` forwarded to
        :func:`discover_and_integrate_orphans`.

    Returns
    -------
    list[str]
        Repository-relative paths of modules successfully added.
    """

    return discover_and_integrate_orphans(
        Path(resolve_path(repo_path)), router=router, context_builder=context_builder
    )


# ----------------------------------------------------------------------
def aggregate_synergy_metrics(
    paths: list[str], metric: str = "roi"
) -> list[tuple[str, float]]:
    """Return scenarios sorted by cumulative synergy ``metric``.

    Parameters
    ----------
    paths:
        Paths to run directories or ``roi_history.json`` files.
    metric:
        Metric name without the ``synergy_`` prefix. Defaults to ``"roi"``.
    """

    from menace.roi_tracker import ROITracker

    metric_name = metric if str(metric).startswith("synergy_") else f"synergy_{metric}"

    results: list[tuple[str, float]] = []
    for entry in paths:
        p = Path(entry)
        hist_path = p / "roi_history.json" if p.is_dir() else p
        name = p.name if p.is_dir() else p.stem
        tracker = ROITracker()
        try:
            tracker.load_history(str(hist_path))
        except Exception:
            logger.exception(
                "failed to load history %s", path_for_prompt(hist_path)
            )
            continue
        vals = tracker.metrics_history.get(metric_name)
        if vals is None:
            vals = tracker.synergy_metrics_history.get(metric_name, [])
        else:
            vals = list(vals)
        total = sum(float(v) for v in vals)
        results.append((name, total))

    return sorted(results, key=lambda x: x[1], reverse=True)
