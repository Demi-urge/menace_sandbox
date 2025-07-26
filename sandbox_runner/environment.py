from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import re

try:
    import resource
except Exception:  # pragma: no cover - not available on some platforms
    resource = None  # type: ignore
import shutil
import subprocess
import tempfile
import textwrap
import logging
import multiprocessing
import time
import inspect
import random
import threading
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Callable, get_origin, get_args
from contextlib import asynccontextmanager, suppress
from filelock import FileLock

try:
    from menace.diagnostic_manager import DiagnosticManager, ResolutionRecord
except Exception:  # pragma: no cover - optional dependency
    DiagnosticManager = None  # type: ignore
    ResolutionRecord = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from pyroute2 import IPRoute, NSPopen, netns
except Exception as exc:
    IPRoute = None  # type: ignore
    NSPopen = None  # type: ignore
    netns = None  # type: ignore
    logger.warning("pyroute2 import failed: %s", exc)

try:
    from faker import Faker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Faker = None  # type: ignore

try:
    from hypothesis import strategies as _hyp_strats  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _hyp_strats = None  # type: ignore

_FAKER = Faker() if Faker is not None else None

from .config import SANDBOX_REPO_URL, SANDBOX_REPO_PATH
from .input_history_db import InputHistoryDB
from collections import Counter
import sqlite3

ROOT = Path(__file__).resolve().parents[1]

# path to cleanup log file
_CLEANUP_LOG_PATH = Path(
    os.getenv("SANDBOX_CLEANUP_LOG", str(ROOT / "sandbox_data" / "cleanup.log"))
)
_CLEANUP_LOG_LOCK = threading.Lock()
POOL_LOCK_FILE = Path(
    os.getenv("SANDBOX_POOL_LOCK", str(ROOT / "sandbox_data" / "pool.lock"))
)

_INPUT_HISTORY_DB: InputHistoryDB | None = None


def _get_history_db() -> InputHistoryDB:
    """Return cached :class:`InputHistoryDB` instance."""
    global _INPUT_HISTORY_DB
    if _INPUT_HISTORY_DB is None:
        path = os.getenv(
            "SANDBOX_INPUT_HISTORY", str(ROOT / "sandbox_data" / "input_history.db")
        )
        _INPUT_HISTORY_DB = InputHistoryDB(path)
    return _INPUT_HISTORY_DB


if DiagnosticManager is not None:
    try:
        _DIAGNOSTIC = DiagnosticManager()
    except Exception:  # pragma: no cover - diagnostics optional
        _DIAGNOSTIC = None
else:
    _DIAGNOSTIC = None


def _log_diagnostic(issue: str, success: bool) -> None:
    """Record a resolution attempt with ``DiagnosticManager`` if available."""
    if _DIAGNOSTIC is None:
        return
    try:
        _DIAGNOSTIC.log.add(ResolutionRecord(issue, "retry", success))
        if not success:
            _DIAGNOSTIC.error_bot.handle_error(issue)
    except Exception as exc:
        logger.exception("diagnostic logging failed: %s", exc)


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

    # optional Bandit integration
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code_str)
            tmp_path = tmp.name
        proc = subprocess.run(
            ["bandit", "-f", "json", "-q", tmp_path],
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            data = json.loads(proc.stdout)
            issues = [
                {
                    "line": i.get("line_number"),
                    "severity": i.get("issue_severity"),
                    "text": i.get("issue_text"),
                }
                for i in data.get("results", [])
            ]
            if issues:
                result["bandit"] = issues
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("bandit failed: %s", exc)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            logger.exception("temporary file removal failed")
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
    logger.warning("docker import failed: %s", exc)
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore
    APIError = Exception  # type: ignore
    _DOCKER_CLIENT = None
else:
    try:
        _DOCKER_CLIENT = docker.from_env()
    except DockerException as exc:  # pragma: no cover - docker may be unavailable
        logger.warning("docker client init failed: %s", exc)
        _DOCKER_CLIENT = None

_CONTAINER_POOL_SIZE = int(os.getenv("SANDBOX_CONTAINER_POOL_SIZE", "2"))
_CONTAINER_IDLE_TIMEOUT = float(os.getenv("SANDBOX_CONTAINER_IDLE_TIMEOUT", "300"))
_POOL_CLEANUP_INTERVAL = float(os.getenv("SANDBOX_POOL_CLEANUP_INTERVAL", "60"))
_WORKER_CHECK_INTERVAL = float(os.getenv("SANDBOX_WORKER_CHECK_INTERVAL", "30"))
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
        except Exception:
            pass
    if thread is not None:
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass
    if loop is not None:
        try:
            loop.close()
        except Exception:
            pass
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
_POOL_METRICS_FILE = Path(
    os.getenv(
        "SANDBOX_POOL_METRICS_FILE", str(ROOT / "sandbox_data" / "pool_failures.json")
    )
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

_ACTIVE_CONTAINERS_FILE = Path(
    os.getenv(
        "SANDBOX_ACTIVE_CONTAINERS",
        str(ROOT / "sandbox_data" / "active_containers.json"),
    )
)

_ACTIVE_OVERLAYS_FILE = Path(
    os.getenv(
        "SANDBOX_ACTIVE_OVERLAYS",
        str(ROOT / "sandbox_data" / "active_overlays.json"),
    )
)

_FAILED_OVERLAYS_FILE = Path(
    os.getenv(
        "SANDBOX_FAILED_OVERLAYS",
        str(ROOT / "sandbox_data" / "failed_overlays.json"),
    )
)

FAILED_CLEANUP_FILE = Path(
    os.getenv(
        "SANDBOX_FAILED_CLEANUP",
        str(ROOT / "sandbox_data" / "failed_cleanup.json"),
    )
)

# timestamp of last automatic purge
_LAST_AUTOPURGE_FILE = Path(
    os.getenv("SANDBOX_LAST_AUTOPURGE", str(ROOT / "sandbox_data" / "last_autopurge"))
)

# age threshold for automatic purge invocation
_SANDBOX_AUTOPURGE_THRESHOLD = 0.0
_LAST_AUTOPURGE_TS = 0.0

# file tracking persistent cleanup statistics
_CLEANUP_STATS_FILE = Path(
    os.getenv(
        "SANDBOX_CLEANUP_STATS",
        str(ROOT / "sandbox_data" / "cleanup_stats.json"),
    )
)

# duration after which stray overlay directories are purged
# defined later once _parse_timespan is available
_OVERLAY_MAX_AGE = 0.0

# threshold for logging persistent cleanup failures
# defined later once _parse_timespan is available
_FAILED_CLEANUP_ALERT_AGE = 0.0

_POOL_FILE_LOCK = FileLock(str(POOL_LOCK_FILE))
_PURGE_FILE_LOCK = FileLock(str(POOL_LOCK_FILE) + ".purge")

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
            pass
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


def _read_active_containers() -> List[str]:
    """Return list of active container IDs from file."""
    with _ACTIVE_CONTAINERS_LOCK:
        return _read_active_containers_unlocked()


def _read_active_containers_unlocked() -> List[str]:
    """Read active container IDs without acquiring the lock."""
    try:
        if _ACTIVE_CONTAINERS_FILE.exists():
            data = json.loads(_ACTIVE_CONTAINERS_FILE.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning(
            "failed reading active containers %s: %s", _ACTIVE_CONTAINERS_FILE, exc
        )
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


def _read_active_overlays() -> List[str]:
    """Return list of active overlay directories from file."""
    with _ACTIVE_OVERLAYS_LOCK:
        return _read_active_overlays_unlocked()


def _read_active_overlays_unlocked() -> List[str]:
    """Read active overlays without acquiring the lock."""
    try:
        if _ACTIVE_OVERLAYS_FILE.exists():
            data = json.loads(_ACTIVE_OVERLAYS_FILE.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning(
            "failed reading active overlays %s: %s", _ACTIVE_OVERLAYS_FILE, exc
        )
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
        except Exception:
            pass
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
    try:
        if _FAILED_OVERLAYS_FILE.exists():
            data = json.loads(_FAILED_OVERLAYS_FILE.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning(
            "failed reading failed overlays %s: %s", _FAILED_OVERLAYS_FILE, exc
        )
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


def _read_failed_cleanup() -> Dict[str, float]:
    """Return mapping of items that failed to clean up and their timestamps."""
    try:
        if FAILED_CLEANUP_FILE.exists():
            data = json.loads(FAILED_CLEANUP_FILE.read_text())
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed reading failed cleanup %s: %s", FAILED_CLEANUP_FILE, exc)
    return {}


def _write_failed_cleanup(entries: Dict[str, float]) -> None:
    """Persist ``entries`` to the failed cleanup file."""
    try:
        _atomic_write_json(FAILED_CLEANUP_FILE, entries)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed writing failed cleanup %s: %s", FAILED_CLEANUP_FILE, exc)


def _read_cleanup_stats() -> Dict[str, int]:
    """Return persistent cleanup statistics."""
    try:
        if _CLEANUP_STATS_FILE.exists():
            data = json.loads(_CLEANUP_STATS_FILE.read_text())
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
    except Exception as exc:  # pragma: no cover - fs errors
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
    try:
        if _LAST_AUTOPURGE_FILE.exists():
            return float(_LAST_AUTOPURGE_FILE.read_text())
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed reading last autopurge %s: %s", _LAST_AUTOPURGE_FILE, exc)
    return 0.0


def _write_last_autopurge(ts: float) -> None:
    """Persist ``ts`` as the last automatic purge time."""
    try:
        _LAST_AUTOPURGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_AUTOPURGE_FILE.write_text(str(ts))
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed writing last autopurge %s: %s", _LAST_AUTOPURGE_FILE, exc)


def _record_failed_cleanup(item: str) -> None:
    """Record ``item`` as failed to clean up with current timestamp."""
    data = _read_failed_cleanup()
    data[item] = time.time()
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
        proc = subprocess.run(
            [sys.executable, "-c", script, path, str(base), str(attempts)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return True
    except Exception as exc:
        logger.debug("rmtree helper failed: %s", exc)

    try:
        proc = subprocess.run(
            ["cmd", "/c", "rmdir", "/s", "/q", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.returncode == 0
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
            except Exception:
                pass
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
                    except Exception:
                        pass
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
        try:
            tmp_dirs: set[str] = set()
            proc = subprocess.run(
                ["pgrep", "-fa", "qemu-system"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            for line in proc.stdout.splitlines():
                parts = line.strip().split(maxsplit=1)
                if not parts:
                    continue
                pid = parts[0]
                cmdline = parts[1] if len(parts) > 1 else ""
                try:
                    logger.info("terminating stale qemu process %s", pid)
                except Exception:
                    pass
                for arg in cmdline.split():
                    if "overlay.qcow2" in arg:
                        if arg.startswith("file="):
                            arg = arg.split("=", 1)[1]
                        tmp_dirs.add(str(Path(arg).parent))
                res = subprocess.run(
                    ["kill", "-9", pid],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
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
        except Exception as exc:
            logger.debug("qemu process cleanup failed: %s", exc)

    if os.name == "nt":  # pragma: no cover - windows process cleanup
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/T", "/IM", "qemu-system*"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            for line in proc.stdout.splitlines():
                if "SUCCESS:" in line:
                    removed_vms += 1
                    _log_cleanup_event("windows_qemu", "vm_process", True)
        except Exception as exc:
            logger.debug("taskkill failed: %s", exc)

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
            except Exception:
                pass
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


def _prune_volumes() -> int:
    """Remove Docker volumes created by the sandbox."""
    if not _PRUNE_VOLUMES:
        return 0
    removed = 0
    labeled: set[str] = set()
    try:
        proc = subprocess.run(
            [
                "docker",
                "volume",
                "ls",
                "-q",
                "--filter",
                f"label={_POOL_LABEL}=1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for vol in proc.stdout.splitlines():
                vol = vol.strip()
                if not vol:
                    continue
                try:
                    logger.info("removing stale sandbox volume %s", vol)
                except Exception:
                    pass
                subprocess.run(
                    ["docker", "volume", "rm", "-f", vol],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                removed += 1
                labeled.add(vol)
                _CLEANUP_METRICS["volume"] += 1
    except Exception as exc:
        logger.debug("leftover volume cleanup failed: %s", exc)

    threshold = time.time() - _CONTAINER_MAX_LIFETIME
    try:
        proc = subprocess.run(
            ["docker", "volume", "ls", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for vol in proc.stdout.splitlines():
                vol = vol.strip()
                if not vol or vol in labeled:
                    continue
                info = subprocess.run(
                    ["docker", "volume", "inspect", vol],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
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
                    except Exception:
                        pass
                    subprocess.run(
                        ["docker", "volume", "rm", "-f", vol],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    removed += 1
                    _CLEANUP_METRICS["volume"] += 1
    except Exception as exc:
        logger.debug("unlabeled volume cleanup failed: %s", exc)

    return removed


def _prune_networks() -> int:
    """Remove Docker networks created by the sandbox."""
    if not _PRUNE_NETWORKS:
        return 0
    removed = 0
    labeled: set[str] = set()
    try:
        proc = subprocess.run(
            [
                "docker",
                "network",
                "ls",
                "-q",
                "--filter",
                f"label={_POOL_LABEL}=1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for net in proc.stdout.splitlines():
                net = net.strip()
                if not net:
                    continue
                try:
                    logger.info("removing stale sandbox network %s", net)
                except Exception:
                    pass
                subprocess.run(
                    ["docker", "network", "rm", "-f", net],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                removed += 1
                labeled.add(net)
                _CLEANUP_METRICS["network"] += 1
    except Exception as exc:
        logger.debug("leftover network cleanup failed: %s", exc)

    threshold = time.time() - _CONTAINER_MAX_LIFETIME
    try:
        proc = subprocess.run(
            ["docker", "network", "ls", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for net in proc.stdout.splitlines():
                net = net.strip()
                if not net or net in labeled:
                    continue
                info = subprocess.run(
                    ["docker", "network", "inspect", net],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
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
                    except Exception:
                        pass
                    subprocess.run(
                        ["docker", "network", "rm", "-f", net],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    removed += 1
                    _CLEANUP_METRICS["network"] += 1
    except Exception as exc:
        logger.debug("unlabeled network cleanup failed: %s", exc)

    return removed


def purge_leftovers() -> None:
    """Remove stale sandbox containers and leftover QEMU overlay files."""
    global _STALE_CONTAINERS_REMOVED
    with _PURGE_FILE_LOCK:
        reconcile_active_containers()
        removed_containers = 0
        try:
            ids = _read_active_containers()
            remaining_ids = []
            for cid in ids:
                try:
                    logger.info("removing recorded sandbox container %s", cid)
                except Exception:
                    pass
                subprocess.run(
                    ["docker", "rm", "-f", cid],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                exists = False
                try:
                    proc = subprocess.run(
                        ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        exists = True
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
                except Exception:
                    pass
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
            proc = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"label={_POOL_LABEL}=1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                for cid in proc.stdout.splitlines():
                    cid = cid.strip()
                    if cid:
                        try:
                            logger.info("removing stale sandbox container %s", cid)
                        except Exception:
                            pass
                        proc_rm = subprocess.run(
                            ["docker", "rm", "-f", cid],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                        _log_cleanup_event(cid, "shutdown", proc_rm.returncode == 0)
                        _remove_failed_cleanup(cid)
                        removed_containers += 1
        except Exception as exc:
            logger.debug("leftover container cleanup failed: %s", exc)

        try:
            threshold = time.time() - _CONTAINER_MAX_LIFETIME
            proc = subprocess.run(
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
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    parts = line.split("\t", 2)
                    if len(parts) < 3:
                        continue
                    cid, created_at, cmd = parts
                    ts_str = " ".join(created_at.split()[:3])
                    try:
                        created_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S %z").timestamp()
                    except Exception:
                        continue
                    if created_ts <= threshold and "sandbox_runner.py" in cmd:
                        try:
                            logger.info("removing stale sandbox container %s", cid)
                        except Exception:
                            pass
                        proc_rm = subprocess.run(
                            ["docker", "rm", "-f", cid],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                        _log_cleanup_event(cid, "shutdown", proc_rm.returncode == 0)
                        _remove_failed_cleanup(cid)
                        removed_containers += 1
        except Exception as exc:
            logger.debug("unlabeled container cleanup failed: %s", exc)

        removed_vms = _purge_stale_vms()

        _prune_volumes()
        _prune_networks()

        _STALE_CONTAINERS_REMOVED += removed_containers

    global _LAST_AUTOPURGE_TS
    _LAST_AUTOPURGE_TS = time.time()
    _write_last_autopurge(_LAST_AUTOPURGE_TS)

    report_failed_cleanup(alert=True)


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
    reconnect = False
    if _DOCKER_CLIENT is None:
        reconnect = True
    else:
        try:
            _DOCKER_CLIENT.ping()
        except DockerException as exc:  # pragma: no cover - ping may fail
            reconnect = True
            try:
                logger.warning("docker client ping failed: %s", exc)
            except Exception:
                pass
    if not reconnect:
        return
    try:
        logger.info("reconnecting docker client")
    except Exception:
        pass
    try:
        _DOCKER_CLIENT = docker.from_env()
        _DOCKER_CLIENT.ping()
        logger.info("docker client reconnected")
    except DockerException as exc:  # pragma: no cover - docker may be down
        _DOCKER_CLIENT = None
        logger.error("docker client reconnection failed: %s", exc)


def _ensure_pool_size_async(image: str) -> None:
    """Warm up pool for ``image`` asynchronously."""
    if _DOCKER_CLIENT is None:
        return
    with _POOL_LOCK:
        pool = _CONTAINER_POOLS.setdefault(image, [])
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
        except Exception:
            pass
        global _ACTIVE_OVERLAY_LIMIT_REACHED
        _ACTIVE_OVERLAY_LIMIT_REACHED += 1
    t = _WARMUP_TASKS.get(image)
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
            _WARMUP_TASKS.pop(image, None)

    task = _schedule_coroutine(_worker())
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
            _record_active_container(container.id)
            _CONSECUTIVE_CREATE_FAILURES[image] = 0
            _log_pool_metrics(image)
            with _POOL_LOCK:
                _CONTAINER_DIRS[container.id] = td
                _CONTAINER_LAST_USED[container.id] = time.time()
                _CONTAINER_CREATED[container.id] = time.time()
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
            attempt += 1
        assert last_exc is not None
        raise last_exc


def _verify_container(container: Any) -> bool:
    """Return ``True`` if ``container`` is healthy and running."""
    try:
        container.reload()
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
                except Exception:  # pragma: no cover - metrics failures
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
            except Exception:  # pragma: no cover - metrics failures
                logger.exception("failed to update gauge cleanup_duration_gauge")
    return result


async def collect_metrics_async(
    prev_roi: float,
    roi: float,
    resources: Dict[str, float] | None,
) -> Dict[str, float]:
    """Async wrapper for :func:`collect_metrics`."""
    return collect_metrics(prev_roi, roi, resources)


def _stop_and_remove(container: Any, retries: int = 3, base_delay: float = 0.1) -> bool:
    """Stop and remove ``container`` with retries.

    Returns ``True`` when the container no longer exists after attempts.
    """
    global _CLEANUP_FAILURES, _FORCE_KILLS
    cid = getattr(container, "id", "")
    for attempt in range(retries):
        try:
            container.stop(timeout=0)
            break
        except Exception as exc:
            if attempt == retries - 1:
                logger.error("failed to stop container %s: %s", cid, exc)
            else:
                time.sleep(base_delay * (2**attempt))
    for attempt in range(retries):
        try:
            container.remove(force=True)
            break
        except Exception as exc:
            if attempt == retries - 1:
                logger.error("failed to remove container %s: %s", cid, exc)
            else:
                time.sleep(base_delay * (2**attempt))

    exists = False
    if cid:
        try:
            proc = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                exists = True
        except Exception as exc:  # pragma: no cover - unexpected runtime issues
            logger.debug("container existence check failed for %s: %s", cid, exc)

    if cid and exists:
        try:
            proc = subprocess.run(
                ["docker", "rm", "-f", cid],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
            exists = False
            try:
                confirm = subprocess.run(
                    ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if confirm.returncode == 0 and confirm.stdout.strip():
                    exists = True
            except Exception as exc:
                logger.debug("container existence re-check failed for %s: %s", cid, exc)
        except Exception as exc:
            _CLEANUP_FAILURES += 1
            logger.error("docker rm fallback failed for container %s: %s", cid, exc)

    if cid and exists:
        try:
            subprocess.run(
                ["docker", "kill", cid],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            subprocess.run(
                ["docker", "rm", "-f", cid],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            exists = False
            try:
                confirm = subprocess.run(
                    ["docker", "ps", "-aq", "--filter", f"id={cid}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if confirm.returncode == 0 and confirm.stdout.strip():
                    exists = True
            except Exception as exc:
                logger.debug("container existence re-check failed for %s: %s", cid, exc)
            _FORCE_KILLS += 1
        except Exception as exc:
            _CLEANUP_FAILURES += 1
            logger.error("docker kill escalation failed for container %s: %s", cid, exc)

    if cid and not exists:
        _remove_active_container(cid)
        _remove_failed_cleanup(cid)
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
) -> Dict[str, float]:
    """Return failed cleanup entries older than ``threshold``.

    When ``alert`` is ``True``, log errors and send a diagnostic record
    if any entries exceed the threshold.
    """
    if threshold is None:
        threshold = _FAILED_CLEANUP_ALERT_AGE
    data = _read_failed_cleanup()
    now = time.time()
    stale = {item: ts for item, ts in data.items() if now - ts >= threshold}
    if alert and stale:
        try:
            logger.error("failed cleanup items: %s", list(stale.keys()))
        except Exception:
            pass
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
                logger.warning("size check failed for %s: %s", path, exc)
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


def _cleanup_idle_containers() -> tuple[int, int]:
    """Remove idle or unhealthy containers.

    Returns a tuple ``(cleaned, replaced)`` where ``cleaned`` is the number of
    idle containers removed and ``replaced`` is the number of unhealthy
    containers purged.
    """
    if _DOCKER_CLIENT is None:
        return 0
    cleaned = 0
    replaced = 0
    now = time.time()
    with _POOL_LOCK:
        pools_snapshot = {img: list(pool) for img, pool in _CONTAINER_POOLS.items()}
    for image, pool in pools_snapshot.items():
        for c in list(pool):
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
            elif not _verify_container(c):
                reason = "unhealthy"
            if reason:
                with _POOL_LOCK:
                    actual_pool = _CONTAINER_POOLS.get(image, [])
                    if c in actual_pool:
                        actual_pool.remove(c)
                success = _stop_and_remove(c)
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
                with _POOL_LOCK:
                    _CONTAINER_LAST_USED.pop(c.id, None)
                    _CONTAINER_CREATED.pop(c.id, None)
                if reason == "idle":
                    cleaned += 1
                else:
                    replaced += 1
                _CLEANUP_METRICS[reason] += 1
                _ensure_pool_size_async(image)
    return cleaned, replaced


def _reap_orphan_containers() -> int:
    """Remove containers labeled with :data:`_POOL_LABEL` not in the pool."""
    if _DOCKER_CLIENT is None:
        return 0
    try:
        containers = _DOCKER_CLIENT.containers.list(
            all=True, filters={"label": f"{_POOL_LABEL}=1"}
        )
    except Exception as exc:  # pragma: no cover - docker may be unavailable
        logger.warning("orphan container listing failed: %s", exc)
        return 0
    with _POOL_LOCK:
        active = {c.id for pool in _CONTAINER_POOLS.values() for c in pool}
    removed = 0
    for c in list(containers):
        if c.id in active:
            continue
        try:
            _verify_container(c)
        except Exception:
            pass
        success = _stop_and_remove(c)
        _log_cleanup_event(c.id, "orphan", success)
        removed += 1
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
    return removed


def reconcile_active_containers() -> None:
    """Remove untracked containers labeled with :data:`_POOL_LABEL`."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={_POOL_LABEL}=1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return
        ids = [cid.strip() for cid in proc.stdout.splitlines() if cid.strip()]
    except Exception as exc:  # pragma: no cover - docker may be unavailable
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
        except Exception:
            pass
        subprocess.run(
            ["docker", "rm", "-f", cid],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _remove_active_container(cid)
        with _POOL_LOCK:
            td = _CONTAINER_DIRS.pop(cid, None)
            _CONTAINER_LAST_USED.pop(cid, None)
            _CONTAINER_CREATED.pop(cid, None)
        if td:
            try:
                shutil.rmtree(td)
            except Exception:
                if os.name == "nt" and _rmtree_windows(td):
                    pass
                else:
                    logger.exception(
                        "temporary directory removal failed for %s", td
                    )


def retry_failed_cleanup() -> tuple[int, int]:
    """Attempt to delete items recorded in :data:`FAILED_CLEANUP_FILE`."""
    data = _read_failed_cleanup()
    successes = 0
    failures = 0
    for item in list(data.keys()):
        is_path = os.path.sep in item or os.path.exists(item)
        if is_path:
            try:
                shutil.rmtree(item)
                _remove_failed_cleanup(item)
                successes += 1
                continue
            except Exception:
                if os.name == "nt" and _rmtree_windows(item):
                    _remove_failed_cleanup(item)
                    successes += 1
                    continue
                failures += 1
                continue
        try:
            subprocess.run(
                ["docker", "rm", "-f", item],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            proc = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"id={item}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and not proc.stdout.strip():
                _remove_failed_cleanup(item)
                successes += 1
            else:
                failures += 1
        except Exception:
            failures += 1

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
            except Exception:
                pass
            try:
                subprocess.run(
                    ["docker", "system", "prune", "-f", "--volumes"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
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
        except Exception:
            pass
        _log_diagnostic("cleanup_retry_failure", False)
        _CONSECUTIVE_CLEANUP_FAILURES += 1
        if _CONSECUTIVE_CLEANUP_FAILURES > _CLEANUP_ALERT_THRESHOLD:
            try:
                logger.error(
                    "cleanup retries failing %s times consecutively",
                    _CONSECUTIVE_CLEANUP_FAILURES,
                )
            except Exception:
                pass
            _log_diagnostic("persistent_cleanup_failure", False)
    else:
        _CONSECUTIVE_CLEANUP_FAILURES = 0

    return successes, failures


async def _cleanup_worker() -> None:
    """Background task to clean idle containers."""
    total_cleaned = 0
    total_replaced = 0
    try:
        while True:
            ensure_docker_client()
            reconcile_active_containers()
            await asyncio.sleep(_POOL_CLEANUP_INTERVAL)
            start = time.monotonic()
            try:
                retry_failed_cleanup()
                cleaned, replaced = _cleanup_idle_containers()
                total_cleaned += cleaned
                total_replaced += replaced
                vm_removed = _purge_stale_vms(record_runtime=True)
                _prune_volumes()
                _prune_networks()
                report_failed_cleanup(alert=True)
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
                try:
                    from . import metrics_exporter as _me
                except Exception:
                    try:  # pragma: no cover - package may not be available
                        import metrics_exporter as _me  # type: ignore
                    except Exception:
                        _me = None  # type: ignore
                gauge = getattr(_me, "cleanup_duration_gauge", None) if _me else None
                if gauge is not None:
                    try:
                        gauge.labels(worker="cleanup").set(duration)
                    except Exception:  # pragma: no cover - metrics failures
                        logger.exception("failed to update cleanup duration")
                global _LAST_CLEANUP_TS
                _LAST_CLEANUP_TS = time.monotonic()
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        logger.debug("cleanup worker cancelled")
        raise


async def _reaper_worker() -> None:
    """Background task to reap orphan containers."""
    total_removed = 0
    try:
        while True:
            await asyncio.sleep(_POOL_CLEANUP_INTERVAL)
            start = time.monotonic()
            try:
                removed = _reap_orphan_containers()
                total_removed += removed
                if removed:
                    logger.info(
                        "reaped %d orphan containers (total %d)", removed, total_removed
                    )
            except Exception:
                logger.exception("orphan container cleanup failed")
            finally:
                duration = time.monotonic() - start
                _CLEANUP_DURATIONS["reaper"] = duration
                try:
                    from . import metrics_exporter as _me
                except Exception:
                    try:  # pragma: no cover - package may not be available
                        import metrics_exporter as _me  # type: ignore
                    except Exception:
                        _me = None  # type: ignore
                gauge = getattr(_me, "cleanup_duration_gauge", None) if _me else None
                if gauge is not None:
                    try:
                        gauge.labels(worker="reaper").set(duration)
                    except Exception:  # pragma: no cover - metrics failures
                        logger.exception("failed to update cleanup duration")
                global _LAST_REAPER_TS
                _LAST_REAPER_TS = time.monotonic()
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        logger.debug("reaper worker cancelled")
        raise


def _cleanup_pools() -> None:
    """Stop and remove pooled containers and stale ones."""
    global _CLEANUP_TASK, _REAPER_TASK
    _POOL_FILE_LOCK.acquire()
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

        try:
            proc = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"label={_POOL_LABEL}=1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                for cid in proc.stdout.splitlines():
                    cid = cid.strip()
                    if cid:
                        try:
                            logger.info("removing stale sandbox container %s", cid)
                        except Exception:
                            pass
                        subprocess.run(
                            ["docker", "rm", "-f", cid],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
            else:
                pass
        except Exception:
            pass
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
            pass
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
            pass
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
        except Exception as exc:  # pragma: no cover - docker missing
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
                pass
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
    if _EVENT_THREAD is None:
        return
    if _EVENT_STOP is not None:
        _EVENT_STOP.set()
    try:
        _EVENT_THREAD.join(timeout=1.0)
    except Exception:
        pass
    _EVENT_THREAD = None
    _EVENT_STOP = None


def ensure_cleanup_worker() -> None:
    """Ensure background cleanup worker task is active."""
    global _CLEANUP_TASK, _REAPER_TASK, _LAST_CLEANUP_TS, _LAST_REAPER_TS
    if _DOCKER_CLIENT is None:
        return
    if _EVENT_THREAD is None or not _EVENT_THREAD.is_alive():
        start_container_event_listener()
    task = _CLEANUP_TASK
    if task is None:
        _CLEANUP_TASK = _schedule_coroutine(_cleanup_worker())
        if _CLEANUP_TASK is None:
            _run_cleanup_sync()
            return
        _LAST_CLEANUP_TS = time.monotonic()
        if _REAPER_TASK is None:
            _REAPER_TASK = _schedule_coroutine(_reaper_worker())
            if _REAPER_TASK is None:
                _run_cleanup_sync()
                return
            _LAST_REAPER_TS = time.monotonic()
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
        pass
    if cancelled or exc is not None:
        _CLEANUP_TASK = _schedule_coroutine(_cleanup_worker())
        if _CLEANUP_TASK is None:
            _run_cleanup_sync()
            return
        _LAST_CLEANUP_TS = time.monotonic()
    task = _REAPER_TASK
    if task is None:
        _REAPER_TASK = _schedule_coroutine(_reaper_worker())
        if _REAPER_TASK is None:
            _run_cleanup_sync()
            return
        _LAST_REAPER_TS = time.monotonic()
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
        pass
    if done or cancelled or exc is not None:
        _REAPER_TASK = _schedule_coroutine(_reaper_worker())
        if _REAPER_TASK is None:
            _run_cleanup_sync()
            return
        _LAST_REAPER_TS = time.monotonic()


def watchdog_check() -> None:
    """Verify background workers are alive and restart if needed."""

    global _CLEANUP_TASK, _REAPER_TASK, _LAST_CLEANUP_TS, _LAST_REAPER_TS

    if _DOCKER_CLIENT is None:
        return

    prev_cleanup = _CLEANUP_TASK
    prev_reaper = _REAPER_TASK
    prev_event = _EVENT_THREAD

    ensure_cleanup_worker()

    now = time.monotonic()
    limit = 2 * _POOL_CLEANUP_INTERVAL
    if now - _LAST_CLEANUP_TS > limit:
        logger.warning("cleanup worker stalled; restarting")
        try:
            if _CLEANUP_TASK is not None:
                _CLEANUP_TASK.cancel()
        finally:
            _CLEANUP_TASK = _schedule_coroutine(_cleanup_worker())
            if _CLEANUP_TASK is None:
                _run_cleanup_sync()
            else:
                _LAST_CLEANUP_TS = time.monotonic()
    if now - _LAST_REAPER_TS > limit:
        logger.warning("reaper worker stalled; restarting")
        try:
            if _REAPER_TASK is not None:
                _REAPER_TASK.cancel()
        finally:
            _REAPER_TASK = _schedule_coroutine(_reaper_worker())
            if _REAPER_TASK is None:
                _run_cleanup_sync()
            else:
                _LAST_REAPER_TS = time.monotonic()

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

if time.time() - _LAST_AUTOPURGE_TS >= _SANDBOX_AUTOPURGE_THRESHOLD:
    purge_leftovers()
    retry_failed_cleanup()

if _DOCKER_CLIENT is not None:
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

atexit.register(_release_pool_lock)
atexit.register(_cleanup_pools)
atexit.register(stop_background_loop)


def register_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""

    def _handler(signum, frame) -> None:  # pragma: no cover - signal path
        try:
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

    def _execute_locally(err_msg: str | None = None) -> Dict[str, float]:
        """Fallback local execution with basic metrics."""
        with tempfile.TemporaryDirectory(prefix="sim_local_") as td:
            path = Path(td) / "snippet.py"
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
                    path = Path(td) / "snippet.py"
                    path.write_text(code_str, encoding="utf-8")

                    image = env.get("CONTAINER_IMAGE")
                    if not image:
                        image = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
                        os_type = env.get("OS_TYPE")
                        if os_type:
                            image = os.getenv(
                                f"SANDBOX_CONTAINER_IMAGE_{os_type.upper()}", image
                            )

                    volumes = {td: {"bind": "/code", "mode": "rw"}}
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
                        ["python", "/code/snippet.py"],
                        **kwargs,
                    )
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
                logger.exception("container execution failed: %s", exc)
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
                path = Path(td) / "snippet.py"
                path.write_text(code_str, encoding="utf-8")

                timeout = int(env.get("TIMEOUT", 300))
                try:
                    result = container.exec_run(
                        ["python", "/code/snippet.py"],
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
            logger.exception("container execution failed: %s", exc)
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
            logger.exception("container execution failed: %s", exc)

    if runtime_metrics:
        env_result["runtime_metrics"] = runtime_metrics

    logger.debug("environment simulation result: %s", env_result)
    return env_result


# ----------------------------------------------------------------------
def generate_sandbox_report(analysis_result: Dict[str, Any], output_path: str) -> None:
    """Write ``analysis_result`` to ``output_path`` as JSON with timestamp."""
    logger.debug("writing sandbox report to %s", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = dict(analysis_result)
    data["timestamp"] = datetime.utcnow().isoformat()
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    logger.debug("sandbox report written: %s", output_path)


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
            logger.exception("failed to remove cgroup directory %s", path)
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
    """Return a normalized set of failure modes from ``value``."""
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

    if not parts:
        return snippet

    return "\n".join(parts) + "\n" + snippet


async def _section_worker(
    snippet: str, env_input: Dict[str, Any], threshold: float
) -> tuple[Dict[str, Any], list[tuple[float, float, Dict[str, float]]]]:
    """Execute ``snippet`` with resource limits and return results."""

    if env_input:
        try:
            _get_history_db().add(env_input)
        except Exception:
            logger.exception("failed to record input history")

    def _run_snippet() -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="run_") as td:
            path = Path(td) / "snippet.py"
            modes = _parse_failure_modes(env_input.get("FAILURE_MODES"))
            snip = _inject_failure_modes(snippet, modes)
            path.write_text(snip, encoding="utf-8")
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_input.items()})
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

    updates: list[tuple[float, float, Dict[str, float]]] = []
    prev = 0.0
    attempt = 0
    delay = 0.5
    retried = False
    while True:
        try:
            result = await _run()
        except Exception as exc:  # pragma: no cover - runtime failures
            logger.exception("section execution failed: %s", exc)
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
        if result.get("exit_code", 0) < 0:
            _log_diagnostic(str(result.get("stderr", "error")), False)
            if attempt >= 2:
                return result, updates
            attempt += 1
            retried = True
            await asyncio.sleep(delay)
            delay *= 2
            continue

        actual = 1.0 if result.get("exit_code") == 0 else 0.0
        metrics = {
            "exit_code": float(result.get("exit_code", 0)),
            "stdout_len": float(len(result.get("stdout", ""))),
            "stderr_len": float(len(result.get("stderr", ""))),
            "cpu": float(result.get("cpu", 0.0)),
            "memory": float(result.get("memory", 0.0)),
            "disk_io": float(result.get("disk_io", 0.0)),
            "net_io": float(result.get("net_io", 0.0)),
            "gpu_usage": float(result.get("gpu_usage", 0.0)),
            "netem_latency_ms": float(result.get("netem_latency_ms", 0.0)),
            "netem_jitter_ms": float(result.get("netem_jitter_ms", 0.0)),
            "netem_packet_loss": float(result.get("netem_packet_loss", 0.0)),
            "netem_packet_duplication": float(
                result.get("netem_packet_duplication", 0.0)
            ),
        }
        if SANDBOX_EXTRA_METRICS:
            metrics.update(SANDBOX_EXTRA_METRICS)
        updates.append((prev, actual, metrics))
        if abs(actual - prev) <= threshold:
            if retried:
                _log_diagnostic("section_worker_retry", True)
            return result, updates
        prev = actual


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


SANDBOX_EXTRA_METRICS: Dict[str, float] = _load_metrics_file(
    os.getenv("SANDBOX_METRICS_FILE", str(ROOT / "sandbox_metrics.yaml"))
)

_preset_env = os.getenv("SANDBOX_ENV_PRESETS", "[]")
try:
    SANDBOX_ENV_PRESETS: List[Dict[str, Any]] = json.loads(_preset_env)
    if isinstance(SANDBOX_ENV_PRESETS, dict):
        SANDBOX_ENV_PRESETS = [SANDBOX_ENV_PRESETS]
    SANDBOX_ENV_PRESETS = [dict(p) for p in SANDBOX_ENV_PRESETS]
except Exception:
    SANDBOX_ENV_PRESETS = [{}]
if not SANDBOX_ENV_PRESETS:
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
        db = _get_history_db()
        with sqlite3.connect(db.path) as conn:
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
            result[key] = Counter(vals).most_common(1)[0][0]
    return result


def _random_strategy(
    count: int, conf: Dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
    conf = conf or {}
    modes = conf.get("modes", ["default", "alt", "stress"])
    level_range = conf.get("level_range", [1, 5])
    flags = conf.get("flags", ["A", "B", "C"])
    flag_prob = float(conf.get("flag_prob", 0.3))
    stubs: List[Dict[str, Any]] = []
    for _ in range(count):
        stub = {
            "mode": random.choice(modes),
            "level": random.randint(int(level_range[0]), int(level_range[1])),
        }
        if random.random() < flag_prob and flags:
            stub["flag"] = random.choice(flags)
        stubs.append(stub)
    return stubs


def _smart_value(name: str, hint: Any) -> Any:
    """Return a realistic value for ``name`` with type ``hint``."""
    val = None
    if _FAKER is not None:
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
    if val is None and _hyp_strats is not None and hint is not inspect._empty:
        try:
            val = _hyp_strats.from_type(hint).example()
        except Exception:
            val = None
    return val


def _stub_from_signature(
    func: Callable[..., Any], *, smart: bool = False
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
            val = _smart_value(name, hint)
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
) -> List[Dict[str, Any]]:
    """Return example input dictionaries.

    ``SANDBOX_INPUT_STUBS`` overrides all other behaviour. When unset the
    generator consults ``providers`` discovered via ``SANDBOX_STUB_PLUGINS``.
    The built-in strategies ``templates``, ``history``, ``random``, ``smart`` and
    ``synthetic`` can be selected via ``strategy`` or the
    ``SANDBOX_STUB_STRATEGY`` environment variable. The ``smart`` strategy
    attempts to generate realistic values using ``faker`` or ``hypothesis`` when
    available. The ``synthetic`` strategy mirrors ``smart`` but is intended for
    language model based stub providers.
    """

    if SANDBOX_INPUT_STUBS:
        stubs = [dict(s) for s in SANDBOX_INPUT_STUBS]
        providers = providers or discover_stub_providers()
        for prov in providers:
            try:
                new = prov(stubs, {"strategy": "env", "target": target})
                if new:
                    stubs = [dict(s) for s in new if isinstance(s, dict)]
            except Exception:
                logger.exception(
                    "stub provider %s failed", getattr(prov, "__name__", "?")
                )
        return stubs

    num = 2 if count is None else max(0, count)

    providers = providers or discover_stub_providers()
    stubs: List[Dict[str, Any]] | None = None

    history = _load_history(os.getenv("SANDBOX_INPUT_HISTORY"))
    strat = strategy or os.getenv("SANDBOX_STUB_STRATEGY", "templates")
    templates: List[Dict[str, Any]] | None = None

    if history:
        stubs = [dict(random.choice(history)) for _ in range(num)]
    else:
        if strat in {"smart", "synthetic"} and target is not None:
            base = _stub_from_signature(target, smart=True)
            stubs = [dict(base) for _ in range(num)]

        if strat in {"templates", "history"}:
            templates = _load_templates(
                os.getenv(
                    "SANDBOX_INPUT_TEMPLATES_FILE",
                    str(ROOT / "sandbox_data" / "input_stub_templates.json"),
                )
            )
            if templates:
                stubs = [dict(random.choice(templates)) for _ in range(num)]
            else:
                agg = aggregate_history_stubs()
                if agg:
                    stubs = [dict(agg) for _ in range(num)]

        if stubs is None:
            if target is not None:
                base = _stub_from_signature(
                    target, smart=strat in {"smart", "synthetic"}
                )
                stubs = [dict(base) for _ in range(num)]
            else:
                conf_env = os.getenv("SANDBOX_STUB_RANDOM_CONFIG", "")
                try:
                    conf = json.loads(conf_env) if conf_env else {}
                except Exception:
                    conf = {}
                stubs = _random_strategy(num, conf) or [{}]

    if strat == "history" or (templates is not None and not templates):
        try:
            from . import generative_stub_provider as gsp  # local import

            stubs = gsp.generate_stubs(stubs, {"strategy": "history", "target": target})
        except Exception:
            logger.exception("history stub generation failed")

    for prov in providers:
        try:
            new = prov(stubs, {"strategy": strat, "target": target})
            if new:
                stubs = [dict(s) for s in new if isinstance(s, dict)]
        except Exception:
            logger.exception("stub provider %s failed", getattr(prov, "__name__", "?"))

    return stubs


# ----------------------------------------------------------------------
def run_repo_section_simulations(
    repo_path: str,
    input_stubs: List[Dict[str, Any]] | None = None,
    env_presets: List[Dict[str, Any]] | None = None,
    *,
    return_details: bool = False,
) -> "ROITracker" | tuple["ROITracker", Dict[str, Dict[str, list[Dict[str, Any]]]]]:
    """Analyse sections and simulate execution environment per section."""
    from menace.roi_tracker import ROITracker
    from menace.self_debugger_sandbox import SelfDebuggerSandbox
    from menace.self_coding_engine import SelfCodingEngine
    from menace.code_database import CodeDB
    from menace.menace_memory_manager import MenaceMemoryManager
    from sandbox_runner.metrics_plugins import (
        discover_metrics_plugins,
        collect_plugin_metrics,
    )

    if input_stubs is None:
        input_stubs = generate_input_stubs()
    if env_presets is None:
        if os.getenv("SANDBOX_GENERATE_PRESETS", "1") != "0":
            try:
                from menace.environment_generator import generate_presets

                env_presets = generate_presets()
            except Exception:
                env_presets = [{}]
        else:
            env_presets = [{}]

    async def _run() -> (
        "ROITracker" | tuple["ROITracker", Dict[str, Dict[str, list[Dict[str, Any]]]]]
    ):
        from sandbox_runner import scan_repo_sections

        logger.info("scanning repository sections in %s", repo_path)
        sections = scan_repo_sections(repo_path)
        tracker = ROITracker()
        plugins = discover_metrics_plugins(os.environ)
        scenario_names = []
        for i, preset in enumerate(env_presets):
            name = preset.get("SCENARIO_NAME", f"scenario_{i}")
            scenario_names.append(name)
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
            max(float(p.get("CPU_LIMIT", 1)) for p in env_presets)
            if env_presets
            else 1.0
        )
        max_mem = (
            max(_parse_size(p.get("MEMORY_LIMIT", 0)) for p in env_presets)
            if env_presets
            else 0
        )
        max_gpu = (
            max(int(p.get("GPU_LIMIT", 0)) for p in env_presets) if env_presets else 0
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
                debugger = SelfDebuggerSandbox(
                    object(), SelfCodingEngine(CodeDB(), MenaceMemoryManager())
                )
                try:
                    for sec_name, lines in sec_map.items():
                        code_str = "\n".join(lines)
                        for p_idx, preset in enumerate(env_presets):
                            scenario = scenario_names[p_idx]
                            logger.info(
                                "simulate %s:%s under scenario %s",
                                module,
                                sec_name,
                                scenario,
                            )
                            for stub in input_stubs:
                                env_input = dict(preset)
                                env_input.update(stub)

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
                                ]:
                                    try:
                                        return await _section_worker(
                                            code_str,
                                            env_input,
                                            tracker.diminishing(),
                                        )
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
            for (_, _fut, module, sec_name, scenario, preset, stub), (
                res,
                updates,
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
                    scenario_metrics = {
                        f"{k}:{scenario}": v for k, v in metrics.items()
                    }
                    pred_roi, _ = tracker.forecast()
                    tracker.record_metric_prediction("roi", pred_roi, actual)
                    tracker.update(
                        prev,
                        actual,
                        modules=[f"{module}:{sec_name}", scenario],
                        metrics={**metrics, **scenario_metrics},
                    )
                if updates:
                    synergy_data[scenario]["roi"].append(updates[-1][1])
                    synergy_data[scenario]["metrics"].append(updates[-1][2])
                if return_details:
                    details.setdefault(module, {}).setdefault(sec_name, []).append(
                        {"preset": preset, "stub": stub, "result": res}
                    )
                if res.get("exit_code") not in (0, None):
                    all_diminished = False

        await _gather_tasks()

        if all_diminished:
            combined: List[str] = []
            for sec_map in sections.values():
                for lines in sec_map.values():
                    combined.extend(lines)
            all_modules = list(sections)
            for p_idx, preset in enumerate(env_presets):
                scenario = scenario_names[p_idx]
                for stub in input_stubs:
                    env_input = dict(preset)
                    env_input.update(stub)
                    logger.info("combined run for scenario %s", scenario)
                    res, updates = await _section_worker(
                        "\n".join(combined),
                        env_input,
                        tracker.diminishing(),
                    )
                    for prev, actual, metrics in updates:
                        extra = collect_plugin_metrics(plugins, prev, actual, metrics)
                        if extra:
                            metrics.update(extra)
                        scenario_metrics = {
                            f"{k}:{scenario}": v for k, v in metrics.items()
                        }
                        pred_roi, _ = tracker.forecast()
                        tracker.record_metric_prediction("roi", pred_roi, actual)
                        tracker.update(
                            prev,
                            actual,
                            modules=all_modules + [scenario],
                            metrics={**metrics, **scenario_metrics},
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
                        if hasattr(tracker, "scenario_synergy"):
                            tracker.scenario_synergy.setdefault(scenario, []).append(
                                synergy_metrics
                            )
                    if return_details:
                        details.setdefault("_combined", {}).setdefault(
                            "all", []
                        ).append({"preset": preset, "stub": stub, "result": res})

        if hasattr(tracker, "scenario_synergy"):
            tracker.scenario_synergy = scenario_synergy
        return (tracker, details) if return_details else tracker

    return asyncio.run(_run())


# ----------------------------------------------------------------------
def simulate_full_environment(preset: Dict[str, Any]) -> "ROITracker":
    """Execute an isolated sandbox run using ``preset`` environment vars."""

    tmp_dir = tempfile.mkdtemp(prefix="full_env_")
    diagnostics: Dict[str, str] = {}
    try:
        repo_path = SANDBOX_REPO_PATH
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

            code = (
                "import subprocess, os\n"
                "subprocess.run(['python', 'sandbox_runner.py'], cwd='"
                + container_repo
                + "')\n"
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
                        "docker execution failed: %s; cmd: docker run <image> python sandbox_runner.py",
                        exc,
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
                                f"python {vm_repo}/sandbox_runner.py",
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
                        pass
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
                ["python", "sandbox_runner.py"],
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
def run_workflow_simulations(
    workflows_db: str | Path = "workflows.db",
    env_presets: List[Dict[str, Any]] | None = None,
    *,
    return_details: bool = False,
    tracker: "ROITracker" | None = None,
) -> "ROITracker" | tuple["ROITracker", Dict[str, list[Dict[str, Any]]]]:
    """Execute stored workflows under optional environment presets."""
    from menace.task_handoff_bot import WorkflowDB
    from menace.roi_tracker import ROITracker
    from menace.self_debugger_sandbox import SelfDebuggerSandbox
    from menace.self_coding_engine import SelfCodingEngine
    from menace.code_database import CodeDB
    from menace.menace_memory_manager import MenaceMemoryManager

    if env_presets is None:
        if os.getenv("SANDBOX_GENERATE_PRESETS", "1") != "0":
            try:
                from menace.environment_generator import generate_presets

                env_presets = generate_presets()
            except Exception:
                env_presets = [{}]
        else:
            env_presets = [{}]

    tracker = tracker or ROITracker()
    scenario_names = [
        p.get("SCENARIO_NAME", f"scenario_{i}") for i, p in enumerate(env_presets)
    ]

    wf_db = WorkflowDB(Path(workflows_db))
    workflows = wf_db.fetch()

    async def _run() -> (
        "ROITracker" | tuple["ROITracker", Dict[str, list[Dict[str, Any]]]]
    ):
        details: Dict[str, list[Dict[str, Any]]] = {}

        tasks: list[tuple[int, asyncio.Task, int, str, Dict[str, Any]]] = []
        index = 0
        synergy_data: Dict[str, Dict[str, list]] = {
            name: {"roi": [], "metrics": []} for name in scenario_names
        }
        combined_results: Dict[str, Dict[str, Any]] = {}

        def _wf_snippet(steps: list[str]) -> str:
            imports: list[str] = []
            calls: list[str] = []
            for idx, step in enumerate(steps):
                mod = ""
                func = ""
                if ":" in step:
                    mod, func = step.split(":", 1)
                elif "." in step:
                    mod, func = step.rsplit(".", 1)
                else:
                    mod, func = "simple_functions", step
                alias = f"_wf_{idx}"
                imports.append(f"from {mod} import {func} as {alias}")
                calls.append(f"{alias}()")
            if not calls:
                return "\n".join(f"# {s}" for s in steps) + "\npass\n"
            return "\n".join(imports + [""] + calls) + "\n"

        for wf in workflows:
            snippet = _wf_snippet(wf.workflow)
            debugger = SelfDebuggerSandbox(
                object(), SelfCodingEngine(CodeDB(), MenaceMemoryManager())
            )
            for p_idx, preset in enumerate(env_presets):
                scenario = scenario_names[p_idx]
                env_input = dict(preset)
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
                    )
                )
                tasks.append((index, fut, wf.wid, scenario, preset))
                index += 1

        for _, fut, wid, scenario, preset in sorted(tasks, key=lambda x: x[0]):
            res, updates = await fut
            for prev, actual, metrics in updates:
                scenario_metrics = {f"{k}:{scenario}": v for k, v in metrics.items()}
                pred_roi, _ = tracker.forecast()
                tracker.record_metric_prediction("roi", pred_roi, actual)
                tracker.update(
                    prev,
                    actual,
                    modules=[f"workflow_{wid}", scenario],
                    metrics={**metrics, **scenario_metrics},
                )
            if updates:
                synergy_data[scenario]["roi"].append(updates[-1][1])
                synergy_data[scenario]["metrics"].append(updates[-1][2])
            if return_details:
                details.setdefault(str(wid), []).append(
                    {"preset": preset, "result": res}
                )

        combined_steps: list[str] = []
        for wf in workflows:
            combined_steps.extend(wf.workflow)
        combined_snippet = _wf_snippet(combined_steps)
        workflow_modules = [f"workflow_{wf.wid}" for wf in workflows]
        for p_idx, preset in enumerate(env_presets):
            scenario = scenario_names[p_idx]
            env_input = dict(preset)
            res, updates = await _section_worker(
                combined_snippet,
                env_input,
                tracker.diminishing(),
            )
            for prev, actual, metrics in updates:
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

        return (tracker, details) if return_details else tracker

    return asyncio.run(_run())


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
            logger.exception("failed to load history %s", hist_path)
            continue
        vals = tracker.metrics_history.get(metric_name)
        if vals is None:
            vals = tracker.synergy_metrics_history.get(metric_name, [])
        else:
            vals = list(vals)
        total = sum(float(v) for v in vals)
        results.append((name, total))

    return sorted(results, key=lambda x: x[1], reverse=True)
