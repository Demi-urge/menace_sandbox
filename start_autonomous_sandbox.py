"""Entry point for launching the autonomous sandbox.

This wrapper bootstraps the environment and model paths automatically before
starting the sandbox. It captures startup exceptions and allows the log level
to be configured via ``SandboxSettings`` or overridden on the command line
without requiring any manual post-launch edits.
"""

from __future__ import annotations

import os
from pathlib import Path

# Must be set before HF/tokenizers imports and before multiprocessing forks.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import importlib

# Resolve the repository root from this file's location instead of assuming a
# hard-coded home directory. The sandbox may live outside ``~/menace_sandbox``
# (e.g., in CI or ephemeral containers), so anchor the import root to the
# checked-out path to avoid missing modules like ``sandbox`` or
# ``bootstrap_metrics``.
ROOT = str(Path(__file__).resolve().parent)

# Ensure ONLY the resolved root is used
sys.path = [p for p in sys.path if "menace_sandbox" not in p or p == ROOT]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _existing_pythonpath:
    _pythonpath_entries = [
        path for path in _existing_pythonpath.split(os.pathsep) if path and path != ROOT
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join([ROOT, *_pythonpath_entries])
else:
    os.environ["PYTHONPATH"] = ROOT

import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

# Auto-start watchdog
_WATCHDOG_SCRIPT = Path(__file__).resolve().parent / "start_watchdog.py"


def _heartbeat_watchdog_running() -> bool:
    import psutil

    for process in psutil.process_iter(attrs=["cmdline"]):
        try:
            cmdline = process.info.get("cmdline")
            if cmdline and "start_watchdog.py" in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return False


if not _heartbeat_watchdog_running():
    subprocess.Popen([sys.executable, str(_WATCHDOG_SCRIPT)])
    print("🐶 [AUTO] Watchdog started.")
else:
    print("🐶 [AUTO] Watchdog already running.")

print("🧭 Beginning import-root normalization...", flush=True)

# Normalize the import roots so that all menace_sandbox modules resolve from a
# single location instead of mixing the repository root with a nested package
# copy. A duplicated path causes metadata to be written and read from different
# directories, which leads to stale embedding checks even after rebuilds.
_ROOT = ROOT

_NESTED = os.path.join(_ROOT, "menace_sandbox")
if _NESTED in sys.path:
    sys.path.remove(_NESTED)

# Ensure the repository and package roots are available before importing any
# project modules that rely on ``menace_sandbox`` as an installed package.
_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    _str_path = str(_path)
    if "menace_sandbox" in _str_path and _str_path != _ROOT:
        continue
    if _str_path not in sys.path:
        sys.path.insert(0, _str_path)

print("🧭 Import roots normalized.", flush=True)

import logging

from logging_utils import get_logger, setup_logging, set_correlation_id, log_record

LOGGER = logging.getLogger(__name__)

print("🔎 sys.path (early):", sys.path, flush=True)
LOGGER.info(
    "Import guard sys.path snapshot",
    extra=log_record(
        event="import-guard-syspath",
        sys_path=sys.path[:5],
        sys_path_total=len(sys.path),
    ),
)
try:
    _meta_planning = importlib.import_module("self_improvement.meta_planning")
    print(
        f"🔎 self_improvement.meta_planning.__file__: {_meta_planning.__file__}",
        flush=True,
    )
    LOGGER.info(
        "Import guard resolved self_improvement.meta_planning",
        extra=log_record(
            event="import-guard-meta-planning",
            module_file=getattr(_meta_planning, "__file__", None),
        ),
    )
except Exception as exc:
    print(
        f"⚠️ Failed to import self_improvement.meta_planning early: {exc}",
        flush=True,
    )
    LOGGER.exception(
        "Import guard failed to import self_improvement.meta_planning",
        extra=log_record(
            event="import-guard-meta-planning-error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        ),
    )

_duplicate_meta_planning_paths = []
for _path in list(sys.path):
    if not _path:
        continue
    _candidate = Path(_path) / "self_improvement" / "meta_planning.py"
    if _candidate.is_file() and str(Path(_path).resolve()) != str(Path(ROOT).resolve()):
        _duplicate_meta_planning_paths.append(str(_candidate))
        sys.path.remove(_path)

if _duplicate_meta_planning_paths:
    print(
        "⚠️ Removed duplicate self_improvement.meta_planning module paths: "
        f"{_duplicate_meta_planning_paths}",
        flush=True,
    )

# --- AUTO-START BOOTSTRAP WATCHDOG ---
WATCHDOG_PATH = os.path.join(_HERE, "sandbox", "bootstrap_watchdog.py")


def _watchdog_running() -> bool:
    import psutil

    for process in psutil.process_iter(attrs=["cmdline"]):
        try:
            cmdline = process.info.get("cmdline")
            if cmdline and "bootstrap_watchdog.py" in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return False


if not _watchdog_running():
    print("🐾 [AUTO] Starting bootstrap watchdog...")
    subprocess.Popen(
        [sys.executable, WATCHDOG_PATH, "--interval", "2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
else:
    print("🐾 [AUTO] Watchdog already running.")
# --------------------------------------

from sandbox.preseed_bootstrap import initialize_bootstrap_context
from sandbox_runner.bootstrap import bootstrap_environment

# Force bootstrap readiness before any downstream imports touch GPTMemoryManager
print("🛠️ Initializing bootstrap context...", flush=True)
initialize_bootstrap_context()
print("✅ Bootstrap context ready.", flush=True)

print("🧰 Bootstrapping environment helpers...", flush=True)
bootstrap_environment()
print("✅ Environment helpers ready.", flush=True)

import argparse
import faulthandler
import json
import random
import signal
import threading
import time
import traceback
import uuid

from coding_bot_interface import (
    _bootstrap_dependency_broker,
    advertise_bootstrap_placeholder,
)
from bootstrap_manager import bootstrap_manager
from system_binary_check import assert_required_system_binaries

_BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL = advertise_bootstrap_placeholder(
    dependency_broker=_bootstrap_dependency_broker()
)

from bootstrap_timeout_policy import (
    enforce_bootstrap_timeout_policy,
    compute_prepare_pipeline_component_budgets,
    load_component_timeout_floors,
    load_escalated_timeout_floors,
    broadcast_timeout_floors,
    guard_bootstrap_wait_env,
    get_bootstrap_guard_context,
    read_bootstrap_heartbeat,
    build_progress_signal_hook,
    derive_bootstrap_timeout_env,
    load_adaptive_stage_windows,
    load_component_runtime_samples,
    collect_timeout_telemetry,
    _BOOTSTRAP_TIMEOUT_MINIMUMS,
    wait_for_bootstrap_quiet_period,
    load_last_global_bootstrap_window,
)
from bootstrap_readiness import (
    CORE_COMPONENTS,
    OPTIONAL_COMPONENTS,
    build_stage_deadlines,
    lagging_optional_components,
    minimal_online,
    start_bootstrap_heartbeat_keepalive,
    stop_bootstrap_heartbeat_keepalive,
    stage_for_step,
)
from coding_bot_interface import get_prepare_pipeline_coordinator
from bootstrap_metrics import (
    BOOTSTRAP_DURATION_STORE,
    BUDGET_MAX_SCALE,
    BUDGET_BUFFER_MULTIPLIER,
    calibrate_overall_timeout,
    calibrate_step_budgets,
    compute_stats,
    load_duration_store,
    record_durations,
)

if "--health-check" in sys.argv[1:]:
    if not os.getenv("SANDBOX_DEPENDENCY_MODE"):
        os.environ["SANDBOX_DEPENDENCY_MODE"] = "minimal"
    # Disable long-running monitoring loops during the lightweight health
    # probe so the command terminates promptly even when background services
    # would normally bootstrap DataBot.
    os.environ.setdefault("MENACE_SANDBOX_MODE", "health_check")
    os.environ.setdefault("MENACE_DISABLE_MONITORING", "1")

_ADAPTIVE_TIMEOUT_FLOORS = derive_bootstrap_timeout_env()
DEFAULT_BOOTSTRAP_TIMEOUTS: Mapping[str, str] = {
    "MENACE_BOOTSTRAP_WAIT_SECS": str(
        _ADAPTIVE_TIMEOUT_FLOORS.get("MENACE_BOOTSTRAP_WAIT_SECS", _BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_WAIT_SECS"])
    ),
    "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS": str(
        _ADAPTIVE_TIMEOUT_FLOORS.get(
            "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", _BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"]
        )
    ),
    "BOOTSTRAP_STEP_TIMEOUT": str(
        _ADAPTIVE_TIMEOUT_FLOORS.get("BOOTSTRAP_STEP_TIMEOUT", _BOOTSTRAP_TIMEOUT_MINIMUMS["BOOTSTRAP_STEP_TIMEOUT"])
    ),
    "BOOTSTRAP_VECTOR_STEP_TIMEOUT": str(
        _ADAPTIVE_TIMEOUT_FLOORS.get(
            "BOOTSTRAP_VECTOR_STEP_TIMEOUT", _BOOTSTRAP_TIMEOUT_MINIMUMS["BOOTSTRAP_VECTOR_STEP_TIMEOUT"]
        )
    ),
}

BOOTSTRAP_WAIT_GUARD = guard_bootstrap_wait_env(floors=_ADAPTIVE_TIMEOUT_FLOORS)

for _timeout_env, _timeout_default in DEFAULT_BOOTSTRAP_TIMEOUTS.items():
    os.environ.setdefault(_timeout_env, _timeout_default)

os.environ.setdefault("MENACE_BOOTSTRAP_STAGGER_SECS", "30")
os.environ.setdefault("MENACE_BOOTSTRAP_STAGGER_JITTER_SECS", "30")

STEP_BUDGET_FLOOR = _BOOTSTRAP_TIMEOUT_MINIMUMS["BOOTSTRAP_STEP_TIMEOUT"]
VECTOR_STEP_BUDGET_FLOOR = max(360.0, _BOOTSTRAP_TIMEOUT_MINIMUMS["BOOTSTRAP_VECTOR_STEP_TIMEOUT"])


def _derive_step_budget(
    *,
    step: str,
    env_var: str | None,
    default: float,
    vector_heavy: bool = False,
) -> float:
    """Return a step budget respecting shared timeout minimums."""

    raw_value = os.getenv(env_var) if env_var else None
    try:
        parsed = float(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        parsed = None

    floor = VECTOR_STEP_BUDGET_FLOOR if vector_heavy else STEP_BUDGET_FLOOR
    requested = parsed if parsed is not None else default
    effective = max(requested, floor)
    logging.getLogger(__name__).info(
        "Bootstrap step budget resolved for %s: requested=%.2fs floor=%.2fs effective=%.2fs env=%s",
        step,
        requested,
        floor,
        effective,
        env_var or "default",
    )
    return effective


BOOTSTRAP_TIMEOUT_POLICY = enforce_bootstrap_timeout_policy(logger=logging.getLogger(__name__))

WATCHER_ROOTS_ENV = "MENACE_BOOTSTRAP_WATCH_ROOTS"
WATCHER_EXCLUDES_ENV = "MENACE_BOOTSTRAP_WATCH_EXCLUDES"
from sandbox_settings import SandboxSettings
from dependency_health import DependencyMode, resolve_dependency_mode
from sandbox.preseed_bootstrap import (
    BOOTSTRAP_PROGRESS,
    BOOTSTRAP_ONLINE_STATE,
    initialize_bootstrap_context,
)
from sandbox_runner.bootstrap import (
    auto_configure_env,
    bootstrap_environment,
    launch_sandbox,
    sandbox_health,
    shutdown_autonomous_sandbox,
)
from lock_utils import SandboxLock, Timeout
try:  # pragma: no cover - allow package relative import
    from metrics_exporter import (
        sandbox_restart_total,
        sandbox_crashes_total,
        sandbox_last_failure_ts,
)
except Exception:  # pragma: no cover - fallback when run as a module
    from .metrics_exporter import (  # type: ignore
        sandbox_restart_total,
        sandbox_crashes_total,
        sandbox_last_failure_ts,
    )

from dynamic_path_router import resolve_path
from shared_event_bus import event_bus as shared_event_bus
from workflow_evolution_manager import WorkflowEvolutionManager
import workflow_graph
from self_improvement.workflow_discovery import discover_workflow_specs
from self_improvement.component_workflow_synthesis import discover_component_workflows
from sandbox_orchestrator import SandboxOrchestrator
from context_builder_util import create_context_builder
from self_improvement.orphan_handling import (
    integrate_orphans,
    integrate_orphans_sync,
    post_round_orphan_scan,
)

_META_PLANNING_MODULE: Any | None = None
_META_PLANNING_LOADED = False
_RECORD_WORKFLOW_ITERATION: Callable[..., Mapping[str, Any]] | None = None
_WORKFLOW_CONTROLLER_STATUS: Callable[..., Any] | None = None


def _load_meta_planning_module(logger: logging.Logger) -> Any | None:
    global _META_PLANNING_MODULE, _META_PLANNING_LOADED

    if _META_PLANNING_LOADED:
        return _META_PLANNING_MODULE

    _META_PLANNING_LOADED = True
    spec = importlib.util.find_spec("self_improvement.meta_planning")
    if spec is None or spec.loader is None:
        logger.warning(
            "Meta planning module unavailable; disabling workflow controller status.",
        )
        _META_PLANNING_MODULE = None
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning(
            "Meta planning module failed to load; disabling workflow controller status. (%s)",
            exc,
        )
        _META_PLANNING_MODULE = None
        return None

    sys.modules["self_improvement.meta_planning"] = module
    _META_PLANNING_MODULE = module
    return module


def _noop_record_workflow_iteration(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
    return {}


def _noop_workflow_controller_status(*_args: Any, **_kwargs: Any) -> None:
    return None


def _load_meta_planning_hooks() -> None:
    global _RECORD_WORKFLOW_ITERATION, _WORKFLOW_CONTROLLER_STATUS

    if _RECORD_WORKFLOW_ITERATION is not None and _WORKFLOW_CONTROLLER_STATUS is not None:
        return

    meta_planning = _load_meta_planning_module(LOGGER)
    if meta_planning is None:
        LOGGER.warning(
            "Meta planning record_workflow_iteration unavailable; using no-op.",
        )
        _RECORD_WORKFLOW_ITERATION = _noop_record_workflow_iteration
        _WORKFLOW_CONTROLLER_STATUS = _noop_workflow_controller_status
        return

    record_workflow = getattr(meta_planning, "record_workflow_iteration", None)
    if record_workflow is None:
        LOGGER.warning(
            "Meta planning module missing record_workflow_iteration; using no-op.",
        )
        record_workflow = _noop_record_workflow_iteration

    controller_status = getattr(meta_planning, "workflow_controller_status", None)
    if controller_status is None:
        LOGGER.warning(
            "Meta planning module missing workflow_controller_status; using no-op.",
        )
        controller_status = _noop_workflow_controller_status

    _RECORD_WORKFLOW_ITERATION = record_workflow
    _WORKFLOW_CONTROLLER_STATUS = controller_status


def record_workflow_iteration(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    _load_meta_planning_hooks()
    return _RECORD_WORKFLOW_ITERATION(*args, **kwargs)  # type: ignore[operator]


def workflow_controller_status(*args: Any, **kwargs: Any) -> Any:
    _load_meta_planning_hooks()
    return _WORKFLOW_CONTROLLER_STATUS(*args, **kwargs)  # type: ignore[operator]

try:  # pragma: no cover - optional dependency
    from task_handoff_bot import WorkflowDB  # type: ignore
except Exception:  # pragma: no cover - allow sandbox startup without WorkflowDB
    WorkflowDB = None  # type: ignore

SHUTDOWN_EVENT = threading.Event()
SIGNAL_SHUTDOWN_REQUESTED = threading.Event()


# Centralized helper for emoji-forward logging so critical launch phases are
# easy to scan in dense startup logs.
def _emoji_step(
    logger: logging.Logger,
    emoji: str,
    message: str,
    *,
    event: str,
    **context: Any,
) -> None:
    logger.info(
        f"{emoji} {message}",
        extra=log_record(event=event, emoji=emoji, **context),
    )


# --- BOOTSTRAP INITIALISATION FIX ---
LOGGER.info("Starting Menace bootstrap sequence...")
initialize_bootstrap_context()
bootstrap_environment()
start_bootstrap_heartbeat_keepalive(logger=LOGGER)
# ------------------------------------

BOOTSTRAP_ARTIFACT_PATH = Path("sandbox_data/bootstrap_artifacts.json")
BOOTSTRAP_SENTINEL_PATH = Path("maintenance-logs/bootstrap_status.json")
BOOTSTRAP_LOCK_PATH = Path(
    os.getenv("MENACE_BOOTSTRAP_LOCK_FILE", "/tmp/menace_bootstrap.lock")
)
BOOTSTRAP_LOCK_TIMEOUT = float(os.getenv("MENACE_BOOTSTRAP_LOCK_TIMEOUT", "300"))
VECTOR_HEAVY_STEPS: set[str] = {
    "embedder_preload",
    "prepare_pipeline",
    "seed_final_context",
    "push_final_context",
    "promote_pipeline",
}

DEFAULT_STEP_BUDGET = _derive_step_budget(
    step="default",
    env_var="BOOTSTRAP_STEP_BUDGET",
    default=STEP_BUDGET_FLOOR,
)
BOOTSTRAP_STEP_BUDGETS: Mapping[str, float] = {
    "embedder_preload": _derive_step_budget(
        step="embedder_preload",
        env_var="BOOTSTRAP_EMBEDDER_BUDGET",
        default=VECTOR_STEP_BUDGET_FLOOR,
        vector_heavy=True,
    ),
    "context_builder": _derive_step_budget(
        step="context_builder", env_var=None, default=STEP_BUDGET_FLOOR
    ),
    "bot_registry": _derive_step_budget(
        step="bot_registry", env_var=None, default=STEP_BUDGET_FLOOR
    ),
    "data_bot": _derive_step_budget(
        step="data_bot", env_var=None, default=STEP_BUDGET_FLOOR
    ),
    "self_coding_engine": _derive_step_budget(
        step="self_coding_engine", env_var=None, default=STEP_BUDGET_FLOOR
    ),
    "prepare_pipeline": _derive_step_budget(
        step="prepare_pipeline",
        env_var=None,
        default=VECTOR_STEP_BUDGET_FLOOR,
        vector_heavy=True,
    ),
    "threshold_persistence": _derive_step_budget(
        step="threshold_persistence", env_var=None, default=STEP_BUDGET_FLOOR
    ),
    "internalize_coding_bot": _derive_step_budget(
        step="internalize_coding_bot", env_var=None, default=STEP_BUDGET_FLOOR
    ),
    "promote_pipeline": _derive_step_budget(
        step="promote_pipeline",
        env_var="BOOTSTRAP_PROMOTE_PIPELINE_BUDGET",
        default=VECTOR_STEP_BUDGET_FLOOR,
        vector_heavy=True,
    ),
    "seed_final_context": _derive_step_budget(
        step="seed_final_context",
        env_var=None,
        default=VECTOR_STEP_BUDGET_FLOOR,
        vector_heavy=True,
    ),
    "push_final_context": _derive_step_budget(
        step="push_final_context",
        env_var=None,
        default=VECTOR_STEP_BUDGET_FLOOR,
        vector_heavy=True,
    ),
    "bootstrap_complete": _derive_step_budget(
        step="bootstrap_complete", env_var=None, default=STEP_BUDGET_FLOOR
    ),
}
ADAPTIVE_LONG_STAGE_GRACE: Mapping[str, tuple[float, int]] = {
    "internalize_coding_bot": (30.0, 3),
    "promote_pipeline": (20.0, 2),
    "seed_final_context": (15.0, 2),
}
BOOTSTRAP_BACKOFF_BASE = float(os.getenv("BOOTSTRAP_BACKOFF_BASE", "20"))
BOOTSTRAP_BACKOFF_MAX = float(os.getenv("BOOTSTRAP_BACKOFF_MAX", "300"))
BOOTSTRAP_MAX_RETRIES = int(os.getenv("BOOTSTRAP_MAX_RETRIES", "2"))


def _step_budget_floors(step_budgets: Mapping[str, float]) -> dict[str, float]:
    floors = {step: STEP_BUDGET_FLOOR for step in step_budgets}
    for step in VECTOR_HEAVY_STEPS:
        floors[step] = max(VECTOR_STEP_BUDGET_FLOOR, floors.get(step, 0.0))
    return floors


def _calibrate_bootstrap_step_budgets(logger: logging.Logger) -> tuple[dict[str, float], dict[str, Any]]:
    """Load persisted metrics and scale step budgets accordingly."""

    store = load_duration_store()
    stats = compute_stats(store.get("bootstrap_steps", {}))
    floors = _step_budget_floors(BOOTSTRAP_STEP_BUDGETS)
    calibrated, debug = calibrate_step_budgets(
        base_budgets=BOOTSTRAP_STEP_BUDGETS,
        stats=stats,
        floors=floors,
    )
    if debug.get("adjusted"):
        logger.info(
            "bootstrap step budgets calibrated from historical timings",
            extra=log_record(
                event="bootstrap-step-budgets-calibrated",
                adjustments=debug.get("adjusted"),
                buffer=debug.get("buffer", BUDGET_BUFFER_MULTIPLIER),
                scale_cap=debug.get("scale_cap", BUDGET_MAX_SCALE),
                source=str(BOOTSTRAP_DURATION_STORE),
            ),
        )
    else:
        logger.info(
            "bootstrap step budgets unchanged; insufficient historical timings",
            extra=log_record(event="bootstrap-step-budgets-default", source=str(BOOTSTRAP_DURATION_STORE)),
        )
    return calibrated, debug


def _persist_step_durations(step_durations: Mapping[str, float], logger: logging.Logger) -> None:
    if not step_durations:
        return

    store = record_durations(durations=step_durations, category="bootstrap_steps", logger=logger)
    stats = compute_stats(store.get("bootstrap_steps", {}))
    logger.info(
        "bootstrap step durations persisted",
        extra=log_record(
            event="bootstrap-step-durations-persisted",
            steps=list(sorted(step_durations)),
            stats={key: {k: round(v, 2) for k, v in value.items()} for key, value in stats.items()},
            store=str(BOOTSTRAP_DURATION_STORE),
        ),
    )


def _normalize_log_level(value: str | int | None) -> int:
    """Return a numeric logging level from user-provided *value*."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return getattr(logging, value.upper(), logging.INFO)
    return logging.INFO


def handle_sigint(sig: int, frame: Any) -> None:
    if SHUTDOWN_EVENT.is_set():
        return
    if SIGNAL_SHUTDOWN_REQUESTED.is_set():
        return
    SIGNAL_SHUTDOWN_REQUESTED.set()
    try:
        signal_name = signal.Signals(sig).name
    except ValueError:
        signal_name = f"signal {sig}"
    print(
        f"[META] Caught {signal_name} — requesting safe shutdown "
        f"(shutdown_in_progress={SHUTDOWN_EVENT.is_set()})"
    )
    SHUTDOWN_EVENT.set()


def cleanup_and_exit(exit_code: int = 0) -> None:
    stop_bootstrap_heartbeat_keepalive()
    try:
        shutdown_autonomous_sandbox()
    except Exception:
        LOGGER.exception(
            "sandbox shutdown failed during signal handling",
            extra=log_record(event="shutdown-error"),
        )
    sys.exit(exit_code)


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)


def _read_json_file(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text()
        return json.loads(raw)
    except Exception:
        return {}


def _extract_context_hint(bootstrap_context: Mapping[str, Any]) -> str | None:
    context_builder = bootstrap_context.get("context_builder")
    for attr in ("context_path", "context_file", "db_path", "path"):
        value = getattr(context_builder, attr, None)
        if isinstance(value, (str, Path)):
            path_value = Path(value)
            if path_value.exists():
                return str(path_value)
    return None


def _persist_bootstrap_artifacts(
    *,
    last_step: str,
    bootstrap_context: Mapping[str, Any],
    completed_steps: Sequence[str] | None = None,
) -> None:
    artifact = {
        "status": "complete",
        "timestamp": time.time(),
        "last_step": last_step,
        "completed_steps": list(completed_steps or []),
    }
    context_hint = _extract_context_hint(bootstrap_context)
    if context_hint:
        artifact["context_hint"] = context_hint

    BOOTSTRAP_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        BOOTSTRAP_ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    except Exception:
        LOGGER.debug("failed to persist bootstrap artifact", exc_info=True)


def _load_bootstrap_artifacts() -> Mapping[str, Any]:
    if not BOOTSTRAP_ARTIFACT_PATH.exists():
        return {}
    return _read_json_file(BOOTSTRAP_ARTIFACT_PATH)


def _reset_bootstrap_sentinel() -> None:
    if BOOTSTRAP_SENTINEL_PATH.exists():
        try:
            BOOTSTRAP_SENTINEL_PATH.unlink()
        except Exception:
            LOGGER.debug("failed to clear bootstrap sentinel", exc_info=True)


def _determine_bootstrap_stagger() -> float:
    """Return delay (seconds) applied to stagger bootstrap start times."""

    base = float(os.getenv("MENACE_BOOTSTRAP_STAGGER_SECS", "0") or 0)
    jitter = float(os.getenv("MENACE_BOOTSTRAP_STAGGER_JITTER_SECS", "0") or 0)
    if jitter > 0:
        base += random.uniform(0, jitter)
    return max(base, 0.0)


def _apply_bootstrap_guard(logger: logging.Logger) -> tuple[float, float]:
    """Pause when another bootstrap is active or host load is elevated."""

    guard_delay, budget_scale = wait_for_bootstrap_quiet_period(logger)
    floors = load_escalated_timeout_floors()
    resolved_timeouts = guard_bootstrap_wait_env(floors=floors)

    for env_var in ("BOOTSTRAP_STEP_TIMEOUT", "BOOTSTRAP_VECTOR_STEP_TIMEOUT"):
        floor = floors.get(env_var, _BOOTSTRAP_TIMEOUT_MINIMUMS.get(env_var, 0.0))
        raw_value = os.getenv(env_var)
        try:
            parsed = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            parsed = None

        resolved = max(parsed if parsed is not None else floor, floor)
        resolved_timeouts[env_var] = resolved

    if budget_scale > 1.0:
        resolved_timeouts = {k: v * budget_scale for k, v in resolved_timeouts.items()}

    for env_var, value in resolved_timeouts.items():
        try:
            os.environ[env_var] = str(float(value))
        except (TypeError, ValueError):
            continue

    logger.info(
        "bootstrap guard timeouts exported",
        extra=log_record(
            event="bootstrap-guard-timeouts",
            guard_scale=budget_scale,
            resolved_timeout_floors={
                key: round(value, 2) for key, value in resolved_timeouts.items()
            },
        ),
    )
    return guard_delay, budget_scale


def _acquire_bootstrap_lock(logger: logging.Logger) -> tuple[SandboxLock, Any]:
    """Serialize bootstrap attempts across processes using a file lock."""

    BOOTSTRAP_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = SandboxLock(str(BOOTSTRAP_LOCK_PATH), timeout=BOOTSTRAP_LOCK_TIMEOUT)
    logger.info(
        "waiting for bootstrap lock before starting heavy initialization",
        extra=log_record(
            event="bootstrap-lock-wait",
            lock_path=str(BOOTSTRAP_LOCK_PATH),
            timeout=BOOTSTRAP_LOCK_TIMEOUT,
        ),
    )
    guard = lock.acquire()
    logger.info(
        "bootstrap lock acquired",
        extra=log_record(
            event="bootstrap-lock-acquired",
            lock_path=str(BOOTSTRAP_LOCK_PATH),
            timeout=BOOTSTRAP_LOCK_TIMEOUT,
        ),
    )
    _emoji_step(
        logger,
        "🔒",
        "Bootstrap lock acquired",
        event="step-bootstrap-lock",
        lock_path=str(BOOTSTRAP_LOCK_PATH),
    )
    return lock, guard


def _compute_watcher_scope() -> tuple[Path, list[Path]]:
    repo_root = resolve_path(".")
    exclusions: list[Path] = [
        repo_root / "sandbox_data",
        repo_root / "checkpoints",
        repo_root / "checkpoint",
        repo_root / ".venv",
        repo_root / "venv",
        repo_root / ".virtualenv",
        repo_root / ".direnv",
    ]

    data_dir = os.getenv("SANDBOX_DATA_DIR")
    if data_dir:
        try:
            exclusions.append(resolve_path(data_dir))
        except Exception:
            LOGGER.debug("unable to resolve SANDBOX_DATA_DIR=%s", data_dir, exc_info=True)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in exclusions:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    return repo_root, deduped


def _log_watcher_scope_hint(logger: logging.Logger) -> None:
    """Advise operators to scope filesystem watchers to the repo root only."""

    repo_root, excluded = _compute_watcher_scope()
    os.environ.setdefault(WATCHER_ROOTS_ENV, str(repo_root))
    if excluded:
        os.environ.setdefault(
            WATCHER_EXCLUDES_ENV, os.pathsep.join(str(path) for path in excluded)
        )
    logger.info(
        "validate editor/sync watchers before bootstrap to avoid I/O storms",
        extra=log_record(
            event="bootstrap-watcher-scope",
            repo_root=str(repo_root),
            excluded_paths=[str(path) for path in excluded],
            guidance=(
                "configure editor or sync watchers to only monitor the active repository root "
                "and explicitly ignore sandbox_data, checkpoint directories, and any virtual "
                "environment folders to reduce startup I/O"
            ),
        ),
    )


def _detect_heavy_watchers(logger: logging.Logger) -> None:
    """Inspect platform watcher state for heavyweight paths and emit warnings.

    The bootstrap sequence is sensitive to broad filesystem watchers that scan
    large artifact trees (``sandbox_data`` checkpoints, virtual environments,
    etc.). This probe opportunistically uses ``lsof`` on Unix-like systems to
    surface processes watching those paths so operators can pause them before
    heavy initialization begins.
    """

    repo_root, excluded = _compute_watcher_scope()
    heavy_paths = [path for path in excluded if path.exists()]
    if not heavy_paths:
        return

    watcher_hits: dict[Path, set[str]] = {path: set() for path in heavy_paths}

    if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
        try:
            cmd = ["lsof", "-Fpn"] + [str(path) for path in heavy_paths]
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            logger.debug("lsof unavailable; skipping watcher inspection")
            return
        except Exception:
            logger.debug("failed to inspect watchers via lsof", exc_info=True)
            return

        current_pid: str | None = None
        for line in (result.stdout or "").splitlines():
            if not line:
                continue
            if line.startswith("p"):
                current_pid = line[1:]
            elif line.startswith("n") and current_pid:
                path_str = line[1:]
                for heavy_path in heavy_paths:
                    try:
                        if Path(path_str).is_relative_to(heavy_path):
                            watcher_hits.setdefault(heavy_path, set()).add(current_pid)
                    except Exception:
                        continue

    noisy_paths = {path: pids for path, pids in watcher_hits.items() if pids}
    if noisy_paths:
        logger.warning(
            "potentially noisy filesystem watchers detected; bootstrap may stall",
            extra=log_record(
                event="bootstrap-watchers-detected",
                repo_root=str(repo_root),
                heavy_paths=[str(path) for path in noisy_paths],
                processes={str(path): sorted(pids) for path, pids in noisy_paths.items()},
                guidance=(
                    "pause background sync or editor watchers touching sandbox_data, checkpoints, "
                    "and virtualenv folders, or reconfigure them to ignore these directories "
                    "before retrying bootstrap"
                ),
            ),
        )
        print(
            "[WARN] Detected active watchers on heavyweight paths (sandbox_data/checkpoints/venv). "
            "Pause sync tools or update ignore rules before continuing to reduce I/O contention.",
            flush=True,
        )


def _scan_for_parallel_bootstraps(logger: logging.Logger) -> list[str]:
    """Return any other bootstrap processes currently running on the host."""

    conflicts: list[str] = []
    try:
        result = subprocess.run(
            [
                "pgrep",
                "-af",
                "bootstrap_self_coding|start_autonomous_sandbox|sandbox_runner",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        logger.warning(
            "pgrep is unavailable; skipping parallel bootstrap detection",
            exc_info=True,
            extra=log_record(event="bootstrap-pgrep-scan-failed"),
        )
        return conflicts

    for line in (result.stdout or "").splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue

        if pid == os.getpid():
            continue

        desc = parts[1] if len(parts) > 1 else ""
        conflicts.append(f"pid {pid}: {desc}".strip())

    if conflicts:
        logger.error(
            "another bootstrap process is already active; aborting startup",
            extra=log_record(
                event="bootstrap-conflict-detected",
                processes=conflicts,
                resolution=(
                    "stop the other bootstrap (pgrep -af "
                    "'bootstrap_self_coding|start_autonomous_sandbox|sandbox_runner')"
                ),
            ),
        )

    return conflicts


def _preflight_bootstrap_conflicts(logger: logging.Logger) -> None:
    """Abort or delay startup when another bootstrap appears to be running."""

    conflicts = _scan_for_parallel_bootstraps(logger)
    if not conflicts:
        return

    delay = float(os.getenv("MENACE_BOOTSTRAP_CONFLICT_DELAY", "0") or 0)
    if delay > 0:
        logger.warning(
            "bootstrap conflict detected; delaying before re-checking",
            extra=log_record(
                event="bootstrap-conflict-delay",
                delay=delay,
                processes=conflicts,
            ),
        )
        time.sleep(delay)
        conflicts = _scan_for_parallel_bootstraps(logger)

    if conflicts:
        print(
            "[ERROR] Another sandbox bootstrap appears to be running. "
            "Please stop the other process before retrying.",
            flush=True,
        )
        sys.exit(1)


def _record_bootstrap_timeout(
    *,
    last_step: str,
    elapsed: float,
    attempt: int,
    step_budget: float,
    stage_timeout: Mapping[str, Any] | None = None,
) -> float:
    backoff = min(BOOTSTRAP_BACKOFF_BASE * (2 ** max(attempt - 1, 0)), BOOTSTRAP_BACKOFF_MAX)
    payload = {
        "status": "timeout",
        "last_step": last_step,
        "timestamp": time.time(),
        "elapsed": elapsed,
        "attempt": attempt,
        "next_allowed": time.time() + backoff,
        "step_budget": step_budget,
    }
    if stage_timeout:
        payload["stage_timeout"] = dict(stage_timeout)
    BOOTSTRAP_SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        BOOTSTRAP_SENTINEL_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except Exception:
        LOGGER.debug("failed to persist bootstrap sentinel", exc_info=True)
    return backoff


def _stage_timeout_context(
    *,
    stage: str,
    stage_entry: Mapping[str, Any] | None,
    stage_elapsed: float,
    stage_deadline: float | None,
    stage_optional: bool,
    stage_enforced: bool,
) -> dict[str, Any]:
    policy_fields = {}
    for key in ("budget", "scaled_budget", "floor", "scale"):
        if isinstance(stage_entry, Mapping) and key in stage_entry:
            policy_fields[key] = stage_entry.get(key)

    context = {
        "stage": stage,
        "elapsed": round(stage_elapsed, 2),
        "deadline": stage_deadline,
        "optional": stage_optional,
        "enforced": stage_enforced,
    }
    if policy_fields:
        context["stage_policy"] = policy_fields
    return context


def _check_bootstrap_sentinel() -> float:
    if not BOOTSTRAP_SENTINEL_PATH.exists():
        return 0.0
    sentinel = _read_json_file(BOOTSTRAP_SENTINEL_PATH)
    next_allowed = float(sentinel.get("next_allowed", 0))
    wait_time = max(next_allowed - time.time(), 0.0)
    return wait_time


def _monitor_bootstrap_thread(
    *,
    bootstrap_thread: threading.Thread,
    bootstrap_stop_event: threading.Event,
    bootstrap_start: float,
    step_budgets: Mapping[str, float],
    adaptive_grace: Mapping[str, tuple[float, int]],
    completed_steps: set[str] | None = None,
    vector_heavy_steps: set[str] | None = None,
    stage_policy: Mapping[str, Any] | None = None,
    stage_signal: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[
    bool,
    str,
    float,
    float,
    set[str],
    set[str],
    dict[str, float],
    dict[str, Any] | None,
]:
    """Track bootstrap progress and enforce per-step budgets.

    Returns a tuple of ``(timed_out, last_step, elapsed, budget_used, observed_steps, lagging_optional, step_durations, stage_timeout_context)``.
    """

    completed = completed_steps or set()
    last_step = BOOTSTRAP_PROGRESS.get("last_step", "unknown")
    step_start_times: dict[str, float] = {last_step: bootstrap_start}
    step_durations: dict[str, float] = {}
    grace_extensions: dict[str, int] = {}
    optional_lagging: set[str] = set()
    stage_start_times: dict[str, float] = {}
    stage_overruns: set[str] = set()
    stage_timeout_context: dict[str, Any] | None = None
    optional_timeout_notes: dict[str, dict[str, object]] = {}
    core_soft_overruns: set[str] = set()
    core_online_announced = False
    stage_progress_sent: dict[str, float] = {}

    while bootstrap_thread.is_alive():
        if SHUTDOWN_EVENT.is_set():
            cleanup_and_exit()

        current_step = BOOTSTRAP_PROGRESS.get("last_step", last_step)
        if current_step not in step_start_times:
            step_start_times[current_step] = time.monotonic()
            if last_step not in step_durations:
                step_durations[last_step] = step_start_times[current_step] - step_start_times[last_step]
        last_step = current_step

        elapsed = time.monotonic() - step_start_times[current_step]
        budget = step_budgets.get(current_step, DEFAULT_STEP_BUDGET)
        budget_floor = (
            VECTOR_STEP_BUDGET_FLOOR
            if vector_heavy_steps and current_step in vector_heavy_steps
            else STEP_BUDGET_FLOOR
        )
        budget = max(budget, budget_floor)
        if current_step in completed:
            budget += 10.0

        grace = adaptive_grace.get(current_step)
        extension_count = grace_extensions.get(current_step, 0)
        if grace and elapsed > budget and extension_count < grace[1]:
            grace_extensions[current_step] = extension_count + 1
            budget += grace[0] * grace_extensions[current_step]
            LOGGER.info(
                "bootstrap step %s exceeded budget; applying adaptive grace (%d/%d)",
                current_step,
                grace_extensions[current_step],
                grace[1],
            )

        stage = stage_for_step(current_step) or "unknown"
        now = time.monotonic()
        if stage not in stage_start_times:
            stage_start_times[stage] = now
        stage_elapsed = time.monotonic() - stage_start_times[stage]
        stage_entry = stage_policy.get(stage, {}) if isinstance(stage_policy, Mapping) else {}
        stage_deadline = stage_entry.get("deadline") if isinstance(stage_entry, Mapping) else None
        stage_soft_budget = (
            stage_entry.get("soft_budget") if isinstance(stage_entry, Mapping) else None
        )
        stage_enforced = bool(stage_entry.get("enforced")) if isinstance(stage_entry, Mapping) else False
        stage_optional = bool(stage_entry.get("optional")) if isinstance(stage_entry, Mapping) else False
        soft_degrade = bool(stage_entry.get("soft_degrade")) if isinstance(stage_entry, Mapping) else False
        soft_target = stage_soft_budget or stage_deadline

        online_state_snapshot = dict(BOOTSTRAP_ONLINE_STATE)

        def _mark_component_state(state: str) -> None:
            components = online_state_snapshot.get("components", {})
            if not isinstance(components, Mapping):
                components = {}
            updated_components = dict(components)
            if updated_components.get(stage) == state:
                return
            updated_components[stage] = state
            online_state_snapshot["components"] = updated_components
            BOOTSTRAP_ONLINE_STATE["components"] = updated_components

        if (
            soft_target is not None
            and stage_entry.get("core_gate")
            and stage_elapsed > soft_target
        ):
            _mark_component_state("degraded")
            if stage not in core_soft_overruns:
                core_soft_overruns.add(stage)
                LOGGER.info(
                    "core gate exceeded target; promoting to degraded-online",
                    extra=log_record(
                        event="bootstrap-core-degraded", stage=stage, elapsed=round(stage_elapsed, 2)
                    ),
                )
                if stage_signal:
                    try:
                        stage_signal(
                            {
                                "event": "bootstrap-core-degraded",
                                "stage": stage,
                                "elapsed": round(stage_elapsed, 2),
                                "soft_target": soft_target,
                            }
                        )
                    except Exception:
                        LOGGER.debug("core degraded signal failed", exc_info=True)

        core_online, lagging_core, degraded_core, degraded_online = minimal_online(
            online_state_snapshot
        )
        degraded_core = set(degraded_core) | set(core_soft_overruns)
        coordinator_ready = False
        coordinator_meta: Mapping[str, object] = {}
        coordinator = get_prepare_pipeline_coordinator()
        if coordinator is not None:
            coordinator_ready, coordinator_meta = coordinator.quorum_met()
            if coordinator_ready and not core_online:
                core_online = True
                degraded_online = True
        online_state_snapshot.update(
            {
                "core_ready": core_online,
                "core_lagging": sorted(lagging_core),
                "core_degraded": sorted(degraded_core),
                "core_degraded_online": degraded_online,
                "pending_optional": list(coordinator_meta.get("pending_optional", ())),
                "pending_background": list(
                    coordinator_meta.get("pending_background", ())
                ),
                "optional_timeout_notes": dict(optional_timeout_notes),
            }
        )
        optional_warming = lagging_optional_components(online_state_snapshot)
        optional_warming.update(optional_lagging)
        optional_warming.update(optional_timeout_notes)
        online_state_snapshot["optional_warming"] = sorted(optional_warming)
        if core_soft_overruns:
            online_state_snapshot["core_warming"] = sorted(core_soft_overruns)
        BOOTSTRAP_ONLINE_STATE.update(online_state_snapshot)
        if not core_online_announced and core_online:
            LOGGER.info(
                "core bootstrap quorum reached; optional services warming",
                extra=log_record(
                    event="bootstrap-online-minimum",
                    online_state=online_state_snapshot,
                ),
            )
            print(
                "[BOOTSTRAP] minimal services online; continuing to warm optional stages",
                flush=True,
            )
            lagging = lagging_optional_components(online_state_snapshot)
            if coordinator_meta.get("pending_optional"):
                lagging.update(coordinator_meta.get("pending_optional", ()))
            if coordinator_meta.get("pending_background"):
                lagging.update(coordinator_meta.get("pending_background", ()))
            lagging.update(optional_timeout_notes)
            lagging.update(optional_lagging)
            if lagging:
                LOGGER.info(
                    "optional components still warming",
                    extra=log_record(
                        event="bootstrap-online-partial",
                        lagging_optional=sorted(lagging),
                        online_state=online_state_snapshot,
                        pending_optional=coordinator_meta.get("pending_optional"),
                        pending_background=coordinator_meta.get("pending_background"),
                    ),
                )
            if degraded_core:
                LOGGER.info(
                    "core components degraded but permitting background warmup",
                    extra=log_record(
                        event="bootstrap-online-degraded",
                        degraded_core=sorted(degraded_core),
                        online_state=online_state_snapshot,
                        pending_optional=coordinator_meta.get("pending_optional"),
                        pending_background=coordinator_meta.get("pending_background"),
                    ),
                )
            core_online_announced = True

        if stage_signal:
            last_signal = stage_progress_sent.get(stage)
            if last_signal is None or now - last_signal > 5:
                try:
                    stage_signal(
                        {
                            "event": "bootstrap-stage-watchdog",
                            "stage": stage,
                            "elapsed": round(stage_elapsed, 2),
                            "deadline": stage_deadline,
                            "soft_budget": stage_soft_budget,
                            "optional": stage_optional,
                            "enforced": stage_enforced,
                            "core_online": core_online,
                            "degraded_online": degraded_online,
                            "coordinator_ready": coordinator_ready,
                            "pending_optional": list(
                                coordinator_meta.get("pending_optional", ())
                            ),
                            "pending_background": list(
                                coordinator_meta.get("pending_background", ())
                            ),
                        }
                    )
                    stage_progress_sent[stage] = now
                except Exception:
                    LOGGER.debug("stage progress signal failed", exc_info=True)
        soft_target = stage_soft_budget or stage_deadline

        if (
            stage_deadline is not None
            and stage_elapsed > stage_deadline
            and stage_enforced
        ):
            stage_timeout_context = _stage_timeout_context(
                stage=stage,
                stage_entry=stage_entry,
                stage_elapsed=stage_elapsed,
                stage_deadline=stage_deadline,
                stage_optional=stage_optional,
                stage_enforced=stage_enforced,
            )
        elif (
            soft_target is not None
            and stage_elapsed > soft_target
            and stage_optional
            and core_online_announced
        ):
            if stage not in stage_overruns:
                stage_overruns.add(stage)
                optional_lagging.add(stage)
                optional_timeout_notes.setdefault(
                    stage,
                    {
                        "elapsed": round(stage_elapsed, 2),
                        "deadline": soft_target,
                        "optional": True,
                        "soft_degrade": True,
                    },
                )
                LOGGER.warning(
                    "optional stage exceeded soft deadline after core readiness",
                    extra=log_record(
                        event="bootstrap-optional-stage-overrun",
                        stage=stage,
                        elapsed=round(stage_elapsed, 2),
                        deadline=soft_target,
                        online_state=online_state_snapshot,
                    ),
                )
                if stage_signal:
                    try:
                        stage_signal(
                            {
                                "event": "bootstrap-optional-stage-overrun",
                                "stage": stage,
                                "elapsed": round(stage_elapsed, 2),
                                "deadline": soft_target,
                                "core_online": core_online,
                                "degraded_online": degraded_online,
                                "optional": True,
                            }
                        )
                    except Exception:
                        LOGGER.debug("optional stage overrun signal failed", exc_info=True)

        if stage_timeout_context and soft_degrade:
            stage_timeout_context = dict(stage_timeout_context)
            stage_timeout_context["soft_degrade"] = True
            _mark_component_state("degraded")
            core_soft_overruns.add(stage)
            optional_lagging.add(stage)
            optional_timeout_notes.setdefault(
                stage,
                {
                    "elapsed": round(stage_elapsed, 2),
                    "deadline": stage_deadline,
                    "optional": stage_optional,
                    "soft_degrade": True,
                },
            )

        if stage_timeout_context and degraded_online:
            stage_timeout_context = dict(stage_timeout_context)
            stage_timeout_context.setdefault("degraded_online", True)
            if degraded_core:
                stage_timeout_context.setdefault(
                    "degraded_core", sorted(degraded_core)
                )

        if stage_timeout_context and stage_enforced and not degraded_online:
            if soft_degrade:
                optional_lagging.add(stage)
                if stage_signal:
                    try:
                        stage_signal(
                            {
                                "event": "bootstrap-stage-soft-overrun",
                                "stage": stage,
                                "elapsed": round(stage_elapsed, 2),
                                "deadline": stage_deadline,
                                "optional": stage_optional,
                            }
                        )
                    except Exception:
                        LOGGER.debug("stage overrun signal failed", exc_info=True)
            else:
                step_durations.setdefault(current_step, elapsed)
                return (
                    True,
                    current_step,
                    elapsed,
                    budget,
                    set(step_start_times),
                    optional_lagging,
                    dict(step_durations),
                    stage_timeout_context,
                )
 
        if elapsed > budget:
            if stage_entry.get("optional"):
                if stage not in optional_lagging:
                    optional_lagging.add(stage)
                    optional_timeout_notes.setdefault(
                        stage,
                        {
                            "elapsed": round(elapsed, 2),
                            "budget": round(budget, 2),
                            "optional": True,
                            "soft_degrade": True,
                        },
                    )
                    LOGGER.warning(
                        "optional bootstrap stage exceeded soft deadline",  # advisory only
                        extra=log_record(
                            event="bootstrap-optional-overrun",
                            stage=stage,
                            elapsed=round(elapsed, 2),
                            budget=round(budget, 2),
                            online_state=online_state_snapshot,
                        ),
                    )
                    print(
                        f"[BOOTSTRAP] optional stage {stage} is lagging ({elapsed:.1f}s)",
                        flush=True,
                    )
                budget += 30.0
            else:
                stage_timeout_context = stage_timeout_context or _stage_timeout_context(
                    stage=stage,
                    stage_entry=stage_entry,
                    stage_elapsed=stage_elapsed,
                    stage_deadline=stage_deadline,
                    stage_optional=stage_optional,
                    stage_enforced=stage_enforced,
                )
                if not degraded_online and not soft_degrade:
                    step_durations.setdefault(current_step, elapsed)
                    return (
                        True,
                        current_step,
                        elapsed,
                        budget,
                        set(step_start_times),
                        optional_lagging,
                        dict(step_durations),
                        stage_timeout_context,
                    )

        bootstrap_thread.join(1.0)

    elapsed = time.monotonic() - bootstrap_start
    budget = step_budgets.get(last_step, DEFAULT_STEP_BUDGET)
    if vector_heavy_steps and last_step in vector_heavy_steps:
        budget = max(budget, VECTOR_STEP_BUDGET_FLOOR)
    else:
        budget = max(budget, STEP_BUDGET_FLOOR)
    step_durations.setdefault(last_step, elapsed)
    optional_lagging.update(optional_timeout_notes)
    return (
        False,
        last_step,
        elapsed,
        budget,
        set(step_start_times),
        optional_lagging,
        dict(step_durations),
        stage_timeout_context,
    )


def _detect_heavy_bootstrap_hints(args: argparse.Namespace | None = None) -> tuple[bool, dict[str, Any]]:
    """Surface signals that bootstrap may be vector-heavy or slow."""

    hints: dict[str, Any] = {}
    if args and getattr(args, "heavy_bootstrap", False):
        hints["flag"] = "cli"
    env_heavy = os.getenv("BOOTSTRAP_HEAVY_BOOTSTRAP", "")
    if env_heavy.lower() in {"1", "true", "yes"}:
        hints["env"] = env_heavy
    vector_env = os.getenv("VECTOR_SERVICE_HEAVY", "")
    if vector_env:
        hints["vector_env"] = vector_env
    vector_wait = os.getenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS")
    standard_wait = os.getenv("MENACE_BOOTSTRAP_WAIT_SECS")
    if vector_wait and vector_wait != standard_wait:
        hints["vector_wait_secs"] = vector_wait
    vector_timeout = os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT")
    standard_timeout = os.getenv("BOOTSTRAP_STEP_TIMEOUT")
    if vector_timeout and vector_timeout != standard_timeout:
        hints["vector_step_timeout"] = vector_timeout

    return bool(hints), hints


def _derive_pipeline_complexity(settings: SandboxSettings) -> dict[str, object]:
    """Best-effort pipeline complexity signals for adaptive budgets."""

    complexity: dict[str, object] = {}

    def _maybe_assign(key: str, value: object) -> None:
        if value:
            complexity[key] = value

    _maybe_assign(
        "vectorizers",
        getattr(settings, "vectorizers", None)
        or getattr(settings, "vectorizer_configs", None)
        or getattr(settings, "vectorizer_endpoints", None),
    )
    _maybe_assign(
        "retrievers",
        getattr(settings, "retrievers", None)
        or getattr(settings, "retriever_configs", None)
        or getattr(settings, "retriever_endpoints", None),
    )

    index_candidates = getattr(settings, "db_indexes", None) or getattr(settings, "index_paths", None)
    if index_candidates:
        _maybe_assign("db_indexes", index_candidates)
        total_bytes = 0
        for candidate in (
            index_candidates
            if isinstance(index_candidates, (list, tuple, set, frozenset))
            else [index_candidates]
        ):
            try:
                total_bytes += Path(candidate).stat().st_size
            except Exception:
                continue
        if total_bytes:
            complexity["db_index_bytes"] = total_bytes

    _maybe_assign(
        "background_loops",
        getattr(settings, "background_loops", None)
        or getattr(settings, "monitoring_loops", None)
        or getattr(settings, "background_tasks", None),
    )

    pipeline_config = getattr(settings, "pipeline_config", None)
    if isinstance(pipeline_config, Mapping):
        complexity["pipeline_config_sections"] = list(pipeline_config)

    orchestrator_components = getattr(settings, "orchestrator_components", None)
    if orchestrator_components:
        complexity["orchestrator_components"] = orchestrator_components

    return complexity


def _resolve_soft_deadline_flag(args: argparse.Namespace | None = None) -> bool:
    env_soft = os.getenv("BOOTSTRAP_SOFT_DEADLINE", "").lower()
    soft_env = env_soft in {"1", "true", "yes"}
    cli_soft = bool(getattr(args, "soft_bootstrap_deadline", False))
    return soft_env or cli_soft


def _aggregate_stage_deadlines(
    stage_deadlines: Mapping[str, Mapping[str, Any]],
    *,
    buffer: float,
    host_telemetry: Mapping[str, object] | None = None,
    guard_context: Mapping[str, object] | None = None,
) -> tuple[float | None, dict[str, object]]:
    """Return a buffered aggregate deadline across enforced stages."""

    enforced_deadlines: list[float] = []
    for entry in stage_deadlines.values():
        if not isinstance(entry, Mapping) or not entry.get("enforced"):
            continue
        deadline = entry.get("deadline")
        if deadline is None:
            continue
        try:
            enforced_deadlines.append(float(deadline))
        except (TypeError, ValueError):
            continue

    guard_context = dict(guard_context or get_bootstrap_guard_context() or {})
    host_telemetry = dict(host_telemetry or {})
    guard_scale = 1.0
    try:
        guard_scale = max(float(guard_context.get("budget_scale", 1.0) or 1.0), 1.0)
    except Exception:
        guard_scale = 1.0
    host_load = host_telemetry.get("host_load")
    try:
        host_load = float(host_load) if host_load is not None else None
    except (TypeError, ValueError):
        host_load = None
    historical_window, historical_window_inputs = load_last_global_bootstrap_window()
    historical_window_inputs = historical_window_inputs or {}
    telemetry_window = host_telemetry.get("global_window") or host_telemetry.get(
        "bootstrap_window"
    )
    try:
        telemetry_window = float(telemetry_window) if telemetry_window is not None else None
    except (TypeError, ValueError):
        telemetry_window = None

    if not enforced_deadlines:
        return None, {
            "buffer": buffer,
            "guard_scale": guard_scale,
            "host_load": host_load,
            "historical_window": historical_window,
            "historical_window_inputs": historical_window_inputs,
        }

    aggregate_deadline = sum(enforced_deadlines)
    host_load_buffer = 1.0
    if host_load is not None:
        host_load_buffer = 1.0 + min(host_load, 2.0) * 0.15
    effective_buffer = max(buffer, guard_scale, host_load_buffer, 1.0)

    buffered_deadline = aggregate_deadline * effective_buffer
    for candidate in (telemetry_window, historical_window):
        if candidate:
            buffered_deadline = max(buffered_deadline, float(candidate))

    return buffered_deadline, {
        "buffer": buffer,
        "effective_buffer": effective_buffer,
        "guard_scale": guard_scale,
        "host_load": host_load,
        "historical_window": historical_window,
        "historical_window_inputs": historical_window_inputs,
        "telemetry_window": telemetry_window,
        "enforced_total": aggregate_deadline,
    }


def _resolve_bootstrap_deadline_policy(
    *,
    baseline_timeout: float,
    heavy_detected: bool,
    soft_deadline: bool,
    hints: Mapping[str, Any],
    pipeline_complexity: Mapping[str, object] | None = None,
    host_telemetry: Mapping[str, object] | None = None,
    load_average: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Determine the effective bootstrap deadline policy."""

    heavy_scale = float(os.getenv("BOOTSTRAP_HEAVY_TIMEOUT_SCALE", "1.5"))
    telemetry = collect_timeout_telemetry()
    runtime_samples = load_component_runtime_samples(telemetry)
    component_floors = load_component_timeout_floors()
    component_budgets = compute_prepare_pipeline_component_budgets(
        component_floors=component_floors,
        pipeline_complexity=pipeline_complexity,
        host_telemetry=host_telemetry,
        load_average=load_average,
    )
    stage_windows = load_adaptive_stage_windows(component_budgets=component_budgets)
    stage_deadlines = build_stage_deadlines(
        baseline_timeout,
        heavy_detected=heavy_detected,
        soft_deadline=soft_deadline,
        heavy_scale=heavy_scale,
        component_budgets=component_budgets,
        component_floors=component_floors,
        stage_windows=stage_windows,
        stage_runtime=runtime_samples,
    )
    LOGGER.info(
        "adaptive stage deadlines computed",
        extra={
            "event": "bootstrap-stage-deadlines",
            "stage_deadlines": stage_deadlines,
            "stage_windows": stage_windows,
            "runtime_samples": runtime_samples,
        },
    )
    aggregate_buffer = float(
        os.getenv("BOOTSTRAP_DEADLINE_BUFFER", str(BUDGET_BUFFER_MULTIPLIER))
    )
    aggregate_deadline, aggregate_meta = _aggregate_stage_deadlines(
        stage_deadlines,
        buffer=aggregate_buffer,
        host_telemetry=host_telemetry,
        guard_context=get_bootstrap_guard_context(),
    )
    telemetry_context = {
        "host_load": host_telemetry.get("host_load") if host_telemetry else None,
        "historical_window": aggregate_meta.get("historical_window"),
        "aggregate_deadline": aggregate_deadline,
        "guard_scale": aggregate_meta.get("guard_scale"),
    }
    for entry in stage_deadlines.values():
        if isinstance(entry, dict):
            entry.setdefault("telemetry", {}).update(telemetry_context)
    max_stage_deadline: float | None = None
    enforced_deadlines = [
        entry.get("deadline")
        for entry in stage_deadlines.values()
        if entry.get("enforced") and entry.get("deadline") is not None
    ]
    if enforced_deadlines:
        max_stage_deadline = max(float(value) for value in enforced_deadlines)
    policy: dict[str, Any] = {
        "baseline_timeout": baseline_timeout,
        "heavy_detected": heavy_detected,
        "soft_deadline": soft_deadline,
        "hints": dict(hints),
        "heavy_scale": heavy_scale,
        "component_budgets": component_budgets,
        "component_floors": component_floors,
        "stage_deadlines": stage_deadlines,
        "aggregate_deadline": aggregate_deadline,
        "deadline_buffer": aggregate_buffer,
        "max_stage_deadline": max_stage_deadline,
        "aggregate_buffer_meta": aggregate_meta,
        "adjusted_timeout": aggregate_deadline,
        "pipeline_complexity": pipeline_complexity,
        "host_telemetry": host_telemetry,
        "load_average": load_average,
    }

    if soft_deadline and heavy_detected:
        policy["mode"] = "soft-heavy"
    elif soft_deadline:
        policy["mode"] = "soft"
    elif heavy_detected:
        policy["mode"] = "scaled"
    else:
        policy["mode"] = "baseline"

    return stage_deadlines, policy


def _emit_meta_trace(logger: logging.Logger, message: str, **details: Any) -> None:
    """Log and print a dense meta-planning breadcrumb for immediate visibility."""

    payload = log_record(event="meta-trace", **details)
    logger.info(message, extra=payload)
    summary_bits = ", ".join(f"{k}={v}" for k, v in sorted(details.items()))
    print(f"[META-TRACE] {message} :: {summary_bits}", flush=True)


def _resolve_dependency_mode(settings: SandboxSettings) -> DependencyMode:
    """Resolve the effective dependency handling policy for *settings*."""

    configured: str | None = getattr(settings, "dependency_mode", None)
    return resolve_dependency_mode(configured)


def _dependency_failure_messages(
    dependency_health: Mapping[str, Any] | None,
    *,
    dependency_mode: DependencyMode,
) -> list[str]:
    """Return user-facing failure reasons derived from dependency metadata."""

    if not isinstance(dependency_health, Mapping):
        return []

    missing: Sequence[Mapping[str, Any]] = tuple(
        item
        for item in dependency_health.get("missing", [])
        if isinstance(item, Mapping)
    )

    if not missing:
        return []

    required = [item for item in missing if not item.get("optional", False)]
    optional = [item for item in missing if item.get("optional", False)]

    failures: list[str] = []
    if required:
        failures.append(
            "missing required dependencies: "
            + ", ".join(sorted(str(item.get("name", "unknown")) for item in required))
        )
    if dependency_mode is not DependencyMode.MINIMAL and optional:
        failures.append(
            "missing optional dependencies in strict mode: "
            + ", ".join(sorted(str(item.get("name", "unknown")) for item in optional))
        )
    return failures


def _evaluate_health(
    health: Mapping[str, Any],
    *,
    dependency_mode: DependencyMode,
) -> tuple[bool, list[str]]:
    """Determine whether ``health`` represents a successful health check."""

    failures: list[str] = []

    if not health.get("databases_accessible", True):
        db_errors = health.get("database_errors")
        if isinstance(db_errors, Mapping) and db_errors:
            details = ", ".join(
                f"{name}: {error}"
                for name, error in sorted(db_errors.items())
            )
            failures.append(f"databases inaccessible ({details})")
        else:
            failures.append("databases inaccessible")

    failures.extend(
        _dependency_failure_messages(
            health.get("dependency_health"),
            dependency_mode=dependency_mode,
        )
    )

    return not failures, failures


def _emit_health_report(
    health: Mapping[str, Any],
    *,
    healthy: bool,
    failures: Sequence[str],
) -> None:
    """Write a structured health report to standard output."""

    payload = {
        "status": "pass" if healthy else "fail",
        "failures": list(failures),
        "health": health,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _load_workflow_records(
    settings: SandboxSettings,
    *,
    discovered_specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[Mapping[str, Any]]:
    """Load workflow specs from configuration and the WorkflowDB if available."""

    records: list[Mapping[str, Any]] = []

    configured = getattr(settings, "workflow_specs", None) or getattr(
        settings, "meta_workflow_specs", None
    )
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        records.extend([spec for spec in configured if isinstance(spec, Mapping)])

    if discovered_specs:
        records.extend([spec for spec in discovered_specs if isinstance(spec, Mapping)])

    if WorkflowDB is not None:
        try:
            wf_db = WorkflowDB(Path(settings.workflows_db))
            records.extend(wf_db.fetch_workflows(limit=200))
        except Exception:  # pragma: no cover - best effort hydration
            LOGGER.exception("failed to hydrate workflow records from WorkflowDB")

    return records


def _discover_repo_workflows(
    *, logger: logging.Logger, base_path: str | Path | None = None
) -> list[Mapping[str, Any]]:
    """Best-effort discovery of workflow-like modules and bots.

    This routine inspects the repository for ``workflow_*.py`` modules as well as
    bot modules ending in ``_bot.py``. Each finding is converted into a minimal
    workflow specification that downstream loaders can hydrate without manual
    configuration.
    """

    specs: list[Mapping[str, Any]] = []
    root = Path(base_path or resolve_path("."))

    try:
        specs.extend(discover_workflow_specs(base_path=root, logger=logger))
    except Exception:
        logger.exception(
            "workflow module discovery failed", extra=log_record(event="workflow-scan")
        )

    try:
        from bot_discovery import _iter_bot_modules

        for mod_path in _iter_bot_modules(root):
            module_name = ".".join(mod_path.relative_to(root).with_suffix("").parts)
            specs.append(
                {
                    "workflow": [module_name],
                    "workflow_id": module_name,
                    "task_sequence": [module_name],
                    "source": "bot_discovery",
                }
            )
    except Exception:
        logger.exception(
            "bot workflow discovery failed", extra=log_record(event="bot-scan")
        )

    return specs


def _decompose_menace_components(
    *,
    settings: SandboxSettings,
    logger: logging.Logger,
    workflow_evolver: WorkflowEvolutionManager,
) -> tuple[list[Mapping[str, Any]], dict[str, Callable[[], Any]]]:
    """Derive workflow specs and callables from the Menace monolith.

    The decomposition step inspects workflow-oriented modules in the repo and
    converts them into executable callables via the ``WorkflowEvolutionManager``
    so the meta-planning loop can mutate and wire them dynamically instead of
    depending solely on pre-seeded bootstrap entries.
    """

    repo_root = getattr(settings, "repo_root", None) or resolve_path(".")
    derived_specs: list[Mapping[str, Any]] = []
    derived_callables: dict[str, Callable[[], Any]] = {}

    try:
        derived_specs.extend(
            discover_workflow_specs(base_path=repo_root, logger=logger)
        )
    except Exception:
        logger.exception(
            "failed to decompose repo workflows",
            extra=log_record(event="menace-decomposition"),
        )

    for spec in derived_specs:
        seq = spec.get("workflow") or spec.get("task_sequence") or []
        workflow_id = str(
            spec.get("workflow_id")
            or spec.get("metadata", {}).get("workflow_id")
            or ""
        ).strip()
        if not workflow_id or not seq:
            continue
        try:
            seq_list = (
                list(seq)
                if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes))
                else [seq]
            )
            derived_callables[workflow_id] = workflow_evolver.build_callable(
                "-".join(str(step) for step in seq_list)
            )
        except Exception:
            logger.exception(
                "failed to build callable for decomposed workflow",
                extra=log_record(workflow_id=workflow_id),
            )

    return derived_specs, derived_callables


def _build_self_improvement_workflows(
    bootstrap_context: Mapping[str, Any],
    settings: SandboxSettings,
    workflow_evolver: WorkflowEvolutionManager,
    *,
    logger: logging.Logger,
    discovered_specs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Callable[[], Any]], workflow_graph.WorkflowGraph]:
    """Return workflow callables and relationship graph for meta planning."""

    workflows: dict[str, Callable[[], Any]] = {}
    graph = workflow_graph.WorkflowGraph()

    def _tag_node(workflow_id: str, **metadata: Any) -> None:
        try:
            if getattr(workflow_graph, "_HAS_NX", False):
                if graph.graph.has_node(workflow_id):
                    graph.graph.nodes[workflow_id].update(metadata)
            else:
                nodes = graph.graph.setdefault("nodes", {})
                nodes.setdefault(workflow_id, {}).update(metadata)
        except Exception:
            logger.debug(
                "failed to tag workflow node", extra=log_record(workflow_id=workflow_id)
            )

    all_discovered = list(discovered_specs or [])
    repo_root = getattr(settings, "repo_root", None)
    all_discovered.extend(
        _discover_repo_workflows(logger=logger, base_path=repo_root)
    )

    try:
        component_specs = discover_component_workflows(
            base_path=repo_root,
            logger=logger,
        )
        all_discovered.extend(component_specs)
    except Exception:
        logger.exception(
            "component workflow synthesis failed",
            extra=log_record(event="component-synthesis-error"),
        )

    derived_specs, derived_callables = _decompose_menace_components(
        settings=settings, logger=logger, workflow_evolver=workflow_evolver
    )
    all_discovered.extend(derived_specs)

    for name in (
        "manager",
        "pipeline",
        "engine",
        "registry",
        "data_bot",
        "context_builder",
    ):
        value = bootstrap_context.get(name)
        if value is None:
            continue

        workflow_id = f"preseeded_{name}"
        workflows[workflow_id] = (lambda v=value: v)
        graph.add_workflow(workflow_id)
        _tag_node(workflow_id, source="bootstrap", order=0)

    records = _load_workflow_records(settings, discovered_specs=all_discovered)
    if derived_callables:
        workflows.update(derived_callables)
        for wf_id in derived_callables:
            graph.add_workflow(wf_id)
            _tag_node(wf_id, source="monolith-decomposition", order=1)

    for record in records:
        seq = record.get("workflow") or record.get("task_sequence") or []
        workflow_id = str(
            record.get("id")
            or record.get("wid")
            or record.get("workflow_id")
            or record.get("name")
            or ""
        ).strip()
        if not workflow_id:
            continue

        try:
            seq_list = (
                list(seq)
                if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes))
                else [seq]
            )
            seq_str = "-".join(str(step) for step in seq_list)
            workflows[workflow_id] = workflow_evolver.build_callable(seq_str)
            graph.add_workflow(workflow_id)
            _tag_node(
                workflow_id,
                source=record.get("source", "record"),
                order=len(seq_list),
            )
            if len(seq_list) > 1:
                for order, (src, dst) in enumerate(zip(seq_list, seq_list[1:]), start=1):
                    try:
                        graph.add_dependency(
                            str(src),
                            str(dst),
                            dependency_type="sequence",
                            order=order,
                        )
                    except Exception:
                        logger.debug(
                            "failed to wire workflow dependency",
                            extra=log_record(src=src, dst=dst, workflow_id=workflow_id),
                        )
        except Exception:  # pragma: no cover - defensive hydration
            logger.exception(
                "failed to hydrate workflow callable",
                extra=log_record(workflow_id=workflow_id),
            )

    context_builder = bootstrap_context.get("context_builder")
    if context_builder and hasattr(context_builder, "refresh_db_weights"):
        workflows.setdefault(
            "refresh_context", lambda cb=context_builder: cb.refresh_db_weights()
        )
        graph.add_workflow("refresh_context")
        _tag_node("refresh_context", source="bootstrap", order=0)

    include_orphans = bool(
        getattr(settings, "include_orphans", False)
        and not getattr(settings, "disable_orphans", False)
    )
    recursive_orphans = bool(getattr(settings, "recursive_orphan_scan", False))
    if include_orphans:
        workflows.setdefault(
            "integrate_orphans",
            lambda recursive=recursive_orphans: integrate_orphans(recursive=recursive),
        )
        graph.add_workflow("integrate_orphans")
        _tag_node("integrate_orphans", source="orphan_handling", order=0)

    if include_orphans and recursive_orphans:
        workflows.setdefault(
            "recursive_orphan_scan",
            lambda: post_round_orphan_scan(recursive=True),
        )
        graph.add_workflow("recursive_orphan_scan")
        _tag_node("recursive_orphan_scan", source="orphan_handling", order=1)

    logger.info(
        "registered %d workflows for meta planning",
        len(workflows),
        extra=log_record(workflow_count=len(workflows)),
    )

    return workflows, graph


def _roi_baseline_available() -> bool:
    """Return ``True`` when historical ROI signals exist on disk."""

    history_path = Path(
        os.environ.get(
            "WORKFLOW_ROI_HISTORY_PATH",
            resolve_path("workflow_roi_history.json"),
        )
    )

    if not history_path.exists():
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: history file missing; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-missing",
                history_path=str(history_path),
            ),
        )
        return False

    try:
        raw = history_path.read_text()
    except Exception:
        LOGGER.exception(
            "prelaunch ROI baseline unavailable: unable to read history file; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-read-error",
                history_path=str(history_path),
            ),
        )
        return False

    if not raw.strip():
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: history file is empty; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-empty",
                history_path=str(history_path),
            ),
        )
        return False

    try:
        data = json.loads(raw)
    except Exception:
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: history file contains invalid JSON; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-invalid-json",
                history_path=str(history_path),
            ),
        )
        return False

    if not isinstance(data, Mapping):
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: unexpected history format; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-invalid-format",
                history_path=str(history_path),
                data_type=type(data).__name__,
            ),
        )
        return False

    valid_entries = 0
    for _, values in data.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue

        for value in values:
            try:
                float(value)
            except Exception:
                continue
            valid_entries += 1
            break

    if not valid_entries:
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: no valid ROI entries found; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-empty-data",
                history_path=str(history_path),
            ),
        )
        return False

    return True


def _run_prelaunch_improvement_cycles(
    workflows: Mapping[str, Callable[[], Any]],
    planner_cls: type | None,
    settings: SandboxSettings,
    logger: logging.Logger,
    *,
    bootstrap_mode: bool = False,
) -> tuple[bool, bool]:
    """Iterate each workflow through ROI-gated improvement before launch."""

    logger.info(
        "✅ starting prelaunch ROI coordination",  # emoji for quick scanning
        extra=log_record(
            event="prelaunch-begin",
            workflow_count=len(workflows),
            planner_available=planner_cls is not None,
            bootstrap_mode=bootstrap_mode,
        ),
    )

    if not workflows:
        logger.error(
            "❌ no workflows available for ROI coordination; aborting sandbox launch",
            extra=log_record(event="meta-coordinator-missing-workflows"),
        )
        raise RuntimeError("no workflows available for ROI coordination")
    else:
        logger.info(
            "✅ workflows detected for ROI coordination",
            extra=log_record(
                event="meta-coordinator-workflows-present",
                workflow_count=len(workflows),
            ),
        )

    if planner_cls is None:
        logger.error(
            "❌ meta planner unavailable; cannot coordinate ROI stagnation",
            extra=log_record(event="meta-coordinator-missing"),
        )
        raise RuntimeError("meta planner unavailable for ROI coordination")
    else:
        logger.info(
            "✅ meta planner located for ROI coordination",
            extra=log_record(event="meta-coordinator-planner-present"),
        )

    system_ready = True
    roi_backoff = False
    per_workflow_ready: dict[str, bool] = {}

    for workflow_id, callable_fn in workflows.items():
        logger.info(
            "ℹ️ coordinating workflow for prelaunch ROI gate",
            extra=log_record(
                event="prelaunch-workflow-start",
                workflow_id=workflow_id,
                continuous_monitor=True,
            ),
        )
        ready, backoff = _coordinate_workflows_until_stagnation(
            {workflow_id: callable_fn},
            planner_cls=planner_cls,
            settings=settings,
            logger=logger,
            continuous_monitor=True,
            cycle_budget=3,
        )
        per_workflow_ready[workflow_id] = ready
        roi_backoff = roi_backoff or backoff
        logger.info(
            "✅ workflow ROI gate completed" if ready else "❌ workflow ROI gate incomplete",
            extra=log_record(
                event="prelaunch-workflow-result",
                workflow_id=workflow_id,
                ready=ready,
            roi_backoff=backoff,
        ),
    )
        if not ready:
            system_ready = False
            logger.warning(
                "❌ workflow stalled before launch",
                extra=log_record(workflow_id=workflow_id, event="prelaunch-stall"),
            )
        elif backoff:
            logger.warning(
                "❌ workflow hit ROI backoff during prelaunch",
                extra=log_record(
                    event="prelaunch-workflow-backoff",
                    workflow_id=workflow_id,
                    roi_backoff=True,
                ),
            )
        else:
            logger.info(
                "✅ workflow cleared ROI gate without backoff",
                extra=log_record(
                    event="prelaunch-workflow-clear",
                    workflow_id=workflow_id,
                    roi_backoff=False,
                ),
            )

    if system_ready and not roi_backoff:
        logger.info(
            "ℹ️ validating combined workflows for ROI stagnation",
            extra=log_record(
                event="prelaunch-system-check",
                workflow_count=len(workflows),
                continuous_monitor=True,
            ),
        )
        system_ready, system_backoff = _coordinate_workflows_until_stagnation(
            workflows,
            planner_cls=planner_cls,
            settings=settings,
            logger=logger,
            continuous_monitor=True,
        )
        roi_backoff = roi_backoff or system_backoff
        logger.info(
            "✅ combined ROI gate reached" if system_ready else "❌ combined ROI gate incomplete",
            extra=log_record(
                event="prelaunch-system-result",
                workflow_count=len(workflows),
                ready=system_ready,
                roi_backoff=system_backoff,
            ),
        )
    else:
        logger.info(
            "ℹ️ skipping combined ROI gate because a workflow stalled or backoff triggered",
            extra=log_record(
                event="prelaunch-system-skip",
                system_ready=system_ready,
                roi_backoff=roi_backoff,
            ),
        )

    snapshot = workflow_controller_status()
    if snapshot:
        logger.info(
            "ℹ️ workflow controller status snapshot",
            extra=log_record(event="prelaunch-controller-status", controllers=snapshot),
        )

    ready = system_ready and all(per_workflow_ready.values())

    logger.info(
        "✅ per-workflow ROI gates cleared" if ready else "❌ one or more workflows blocked",
        extra=log_record(
            event="prelaunch-per-workflow-status",
            ready=ready,
            roi_backoff=roi_backoff,
            workflow_count=len(per_workflow_ready),
            blocked_workflows=[k for k, v in per_workflow_ready.items() if not v],
        ),
    )

    if (
        bootstrap_mode
        and workflows
        and planner_cls is not None
        and not roi_backoff
        and not ready
    ):
        logger.info(
            "✅ bypassing diminishing returns gate during bootstrap; ROI baseline unavailable",
            extra=log_record(
                event="meta-coordinator-bootstrap-bypass",
                workflow_count=len(workflows),
            ),
        )
        ready = True
        logger.info(
            "✅ bootstrap bypass activated; launching despite missing ROI baseline",
            extra=log_record(event="meta-coordinator-bootstrap-bypass-applied"),
        )

    logger.info(
        "✅ prelaunch ROI coordination complete" if ready else "❌ prelaunch ROI coordination incomplete",
        extra=log_record(
            event="prelaunch-complete",
            ready=ready,
            roi_backoff=roi_backoff,
            bootstrap_mode=bootstrap_mode,
        ),
    )
    _emoji_step(
        logger,
        "🚦",
        "Prelaunch ROI gate evaluated",
        event="step-prelaunch-roi",
        ready=ready,
        roi_backoff=roi_backoff,
        bootstrap_mode=bootstrap_mode,
    )

    return ready, roi_backoff


def _coordinate_workflows_until_stagnation(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    planner_cls: type | None,
    settings: SandboxSettings,
    logger: logging.Logger,
    continuous_monitor: bool = False,
    cycle_budget: int | None = None,
) -> tuple[bool, bool]:
    """Iterate workflows through the meta planner until ROI gains stagnate."""

    roi_settings = getattr(settings, "roi", None)
    threshold = float(
        getattr(roi_settings, "stagnation_threshold", 0.0)
        if roi_settings is not None
        else 0.0
    )
    streak_required = max(
        1,
        int(
            getattr(roi_settings, "stagnation_cycles", 1)
            if roi_settings is not None
            else 1
        ),
    )

    logger.info(
        "🔍 attempting to initialize meta planner for ROI coordination",
        extra=log_record(event="meta-coordinator-init-begin"),
    )
    print(
        "[META-TRACE] initializing meta planner class=%s continuous_monitor=%s cycle_budget=%s"
        % (
            getattr(planner_cls, "__name__", str(planner_cls)),
            continuous_monitor,
            cycle_budget,
        ),
        flush=True,
    )
    try:
        planner = planner_cls(context_builder=create_context_builder())
        logger.info(
            "✅ meta planner initialized for ROI coordination",
            extra=log_record(event="meta-coordinator-init-success"),
        )
        _emit_meta_trace(
            logger,
            "meta planner instantiated",
            planner_class=getattr(planner_cls, "__name__", str(planner_cls)),
            continuous_monitor=continuous_monitor,
            cycle_budget=cycle_budget,
            workflow_count=len(workflows),
        )
        print(
            "[META-TRACE] meta planner instantiated with workflows=%d context_builder_ready=%s"
            % (len(workflows), hasattr(planner, "context_builder")),
            flush=True,
        )
    except Exception:
        logger.exception(
            "❌ failed to initialize meta planner for ROI coordination",
            extra=log_record(event="meta-coordinator-init-error"),
        )
        raise

    for name, value in {
        "mutation_rate": settings.meta_mutation_rate,
        "roi_weight": settings.meta_roi_weight,
        "domain_transition_penalty": settings.meta_domain_penalty,
    }.items():
        if hasattr(planner, name):
            setattr(planner, name, value)
            logger.info(
                "✅ applied planner setting",  # emoji for quick scanning
                extra=log_record(
                    event="meta-coordinator-setting-applied",
                    setting=name,
                    value=value,
                ),
            )
            print(
                "[META-TRACE] planner attribute %s set to %s" % (name, value),
                flush=True,
            )
            _emit_meta_trace(
                logger,
                "planner attribute updated",
                setting=name,
                value=value,
                planner_class=getattr(planner_cls, "__name__", str(planner_cls)),
            )
        else:
            logger.debug(
                "ℹ️ planner setting skipped; attribute missing",
                extra=log_record(event="meta-coordinator-setting-skipped", setting=name),
            )
            _emit_meta_trace(
                logger,
                "planner attribute missing; skipped update",
                setting=name,
                planner_class=getattr(planner_cls, "__name__", str(planner_cls)),
            )
            print(
                "[META-TRACE] planner missing attribute %s; leaving default" % name,
                flush=True,
            )

    diminishing: set[str] = set()
    roi_backoff_triggered = False
    budget = cycle_budget or max(len(workflows) * streak_required * 2, 3)
    print(
        "[META-TRACE] planner cycle budget established at %d (workflows=%d streak_required=%d threshold=%.4f)"
        % (budget, len(workflows), streak_required, threshold),
        flush=True,
    )

    for cycle in range(budget):
        if SHUTDOWN_EVENT.is_set():
            cleanup_and_exit()
        _emit_meta_trace(
            logger,
            "meta planner coordination cycle start",
            cycle=cycle,
            budget=budget,
            workflow_count=len(workflows),
            diminishing=len(diminishing),
        )
        try:
            records = planner.discover_and_persist(workflows)
            logger.info(
                "meta planner cycle executed",  # dense trace per cycle
                extra=log_record(
                    event="meta-coordinator-cycle",
                    cycle=cycle,
                    budget=budget,
                    record_count=len(records) if records else 0,
                    diminishing=len(diminishing),
                    workflows=list(workflows.keys()),
                ),
            )
            print(
                "[META-TRACE] planner cycle %d completed; records=%d diminishing=%d"
                % (cycle, len(records) if records else 0, len(diminishing)),
                flush=True,
            )
            print(
                "[META-TRACE] planner cycle %d outputs=%s" % (cycle, records),
                flush=True,
            )
        except Exception:
            logger.exception(
                "❌ meta planner coordination failed",
                extra=log_record(event="meta-coordinator-error", cycle=cycle),
            )
            break

        if not records:
            logger.info(
                "✅ meta planner returned no records; assuming diminishing returns",
                extra=log_record(event="meta-coordinator-empty", cycle=cycle),
            )
            _emit_meta_trace(
                logger,
                "meta planner returned no records",
                cycle=cycle,
                diminishing=len(diminishing),
                workflow_count=len(workflows),
            )
            break

        for rec in records:
            chain = rec.get("chain", [])
            chain_id = "->".join(chain) if chain else rec.get("workflow_id", "unknown")
            roi_gain = float(rec.get("roi_gain", 0.0))
            stats: dict[str, float] = {}
            if getattr(planner, "roi_db", None) is not None and isinstance(chain_id, str):
                try:
                    stats = planner.roi_db.fetch_chain_stats(chain_id)  # type: ignore[operator]
                except Exception:
                    logger.debug(
                        "roi stats lookup failed", extra=log_record(workflow_id=chain_id)
                    )

            roi_delta = float(stats.get("delta_roi", roi_gain))
            streak = int(stats.get("non_positive_streak", 0))
            controller_state = record_workflow_iteration(
                chain_id,
                roi_gain=roi_gain,
                roi_delta=roi_delta,
                threshold=threshold,
                patience=streak_required,
            )
            stagnated = roi_delta <= threshold and streak >= streak_required
            if controller_state.get("status") == "halted":
                stagnated = True
                logger.info(
                    "✅ workflow controller halted improvements",
                    extra=log_record(
                        workflow_id=chain_id,
                        roi_delta=controller_state.get("last_delta", roi_gain),
                        threshold=controller_state.get("threshold", threshold),
                        event="meta-controller-halt",
                    ),
                )

            logger.info(
                "meta planning progress",
                extra=log_record(
                    workflow_id=chain_id,
                    roi_gain=roi_gain,
                    roi_delta=roi_delta,
                    stagnation_threshold=threshold,
                    non_positive_streak=streak,
                    stagnation_met=stagnated,
                    controller_status=controller_state,
                    diminishing_complete=len(diminishing),
                    diminishing_target=len(workflows),
                    cycle=cycle,
                ),
            )
            _emit_meta_trace(
                logger,
                "meta planning record processed",
                workflow_id=chain_id,
                roi_gain=roi_gain,
                roi_delta=roi_delta,
                stagnated=stagnated,
                streak=streak,
                cycle=cycle,
                diminishing=len(diminishing),
            )
            print(
                "[META-TRACE] record processed; chain=%s roi_gain=%.4f roi_delta=%.4f stagnated=%s streak=%d"
                % (chain_id, roi_gain, roi_delta, stagnated, streak),
                flush=True,
            )

            if stagnated and isinstance(chain_id, str):
                diminishing.add(chain_id)

            if continuous_monitor and roi_delta <= threshold:
                roi_backoff_triggered = True
                logger.warning(
                    "roi backoff triggered during coordination",
                    extra=log_record(
                        workflow_id=chain_id,
                        roi_delta=roi_delta,
                        stagnation_threshold=threshold,
                        non_positive_streak=streak,
                        event="roi-backoff",
                    ),
                )
                _emit_meta_trace(
                    logger,
                    "roi backoff triggered",
                    workflow_id=chain_id,
                    roi_delta=roi_delta,
                    threshold=threshold,
                    streak=streak,
                    cycle=cycle,
                )
                break

        if roi_backoff_triggered or len(diminishing) >= len(workflows):
            print(
                "[META-TRACE] planner terminating early; roi_backoff=%s diminishing=%d/%d"
                % (roi_backoff_triggered, len(diminishing), len(workflows)),
                flush=True,
            )
            break

    ready = len(diminishing) >= len(workflows)
    if not ready:
        logger.warning(
            "diminishing returns not reached for all workflows",
            extra=log_record(
                achieved=len(diminishing),
                total=len(workflows),
                event="meta-coordinator-incomplete",
            ),
        )
        print(
            "[META-TRACE] diminishing returns incomplete; achieved=%d total=%d"
            % (len(diminishing), len(workflows)),
            flush=True,
        )
    else:
        print(
            "[META-TRACE] diminishing returns reached for all workflows; ready=%s backoff=%s"
            % (ready, roi_backoff_triggered),
            flush=True,
        )

    _emit_meta_trace(
        logger,
        "meta coordination completed",
        ready=ready,
        roi_backoff_triggered=roi_backoff_triggered,
        diminishing=len(diminishing),
        workflow_count=len(workflows),
        cycles_budget=budget,
    )
    print(
        "[META-TRACE] meta coordination complete; ready=%s roi_backoff=%s diminishing=%d/%d"
        % (ready, roi_backoff_triggered, len(diminishing), len(workflows)),
        flush=True,
    )

    return ready, roi_backoff_triggered


def main(argv: list[str] | None = None) -> None:
    """Launch the sandbox with optional log level configuration.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. If ``None`` the arguments will
        be pulled from :data:`sys.argv`.
    """

    print("[start_autonomous_sandbox] main() entry", flush=True)

    system_binaries_validated = False
    assert_required_system_binaries()
    system_binaries_validated = True

    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--health-check" in argv_list and not os.getenv("SANDBOX_DEPENDENCY_MODE"):
        os.environ["SANDBOX_DEPENDENCY_MODE"] = "minimal"

    settings = SandboxSettings()
    # Automatically configure the environment before proceeding so the caller
    # does not need to pre-populate configuration files or model paths.
    auto_configure_env(settings)
    # Reload settings to pick up any values written by ``auto_configure_env``.
    settings = SandboxSettings()

    bootstrap_timeout_default = 300.0
    env_bootstrap_timeout = os.getenv("BOOTSTRAP_CONTEXT_TIMEOUT")
    if env_bootstrap_timeout:
        try:
            bootstrap_timeout_default = float(env_bootstrap_timeout)
        except ValueError:
            print(
                "[WARN] BOOTSTRAP_CONTEXT_TIMEOUT is not a number; using 300s default",
                flush=True,
            )

    parser = argparse.ArgumentParser(description="Launch the autonomous sandbox")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=settings.sandbox_log_level,
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run sandbox health checks and exit",
    )
    parser.add_argument(
        "--monitor-roi-backoff",
        action="store_true",
        help="Continuously monitor ROI backoff and pause launch when triggered",
    )
    parser.add_argument(
        "--heavy-bootstrap",
        action="store_true",
        help="Hint that heavy vector/bootstrap work is expected so timeouts may scale",
    )
    parser.add_argument(
        "--soft-bootstrap-deadline",
        action="store_true",
        help="Treat the bootstrap deadline as advisory when heavy work is detected",
    )
    parser.add_argument(
        "--bootstrap-timeout",
        type=float,
        default=bootstrap_timeout_default,
        help=(
            "Maximum seconds to wait for initialize_bootstrap_context before failing; "
            "per-step watchdogs now inherit shared timeout policy floors (>=240s or "
            ">=360s for vector-heavy stages)"
        ),
    )
    parser.add_argument(
        "--include-orphans",
        dest="include_orphans",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include orphan modules during testing and planning",
    )
    parser.add_argument(
        "--recursive-orphan-scan",
        dest="recursive_orphan_scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable recursive orphan discovery and integration",
    )
    args = parser.parse_args(argv_list)

    print(
        f"[DEBUG] Parsed args: {args}; health_check={getattr(args, 'health_check', None)}",
        flush=True,
    )

    if args.include_orphans is not None:
        os.environ["SANDBOX_INCLUDE_ORPHANS"] = "1" if args.include_orphans else "0"
        settings.include_orphans = bool(args.include_orphans)
    if args.recursive_orphan_scan is not None:
        os.environ["SANDBOX_RECURSIVE_ORPHANS"] = (
            "1" if args.recursive_orphan_scan else "0"
        )
        settings.recursive_orphan_scan = bool(args.recursive_orphan_scan)

    log_level = _normalize_log_level(args.log_level)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging(level=log_level)
    else:
        root_logger.setLevel(log_level)
        for handler in list(root_logger.handlers):
            handler.setLevel(log_level)

    bootstrap_logger = logging.getLogger("sandbox.preseed_bootstrap")
    bootstrap_logger.setLevel(log_level)
    cid = f"sas-{uuid.uuid4()}"
    set_correlation_id(cid)
    logger = get_logger(__name__)
    _emoji_step(
        logger,
        "🧰",
        "Prepare the sandbox environment and defaults",
        event="step-prepare-environment",
        sandbox_root=ROOT,
        dependency_mode=os.getenv("SANDBOX_DEPENDENCY_MODE"),
    )
    _emoji_step(
        logger,
        "📜",
        "Parsed CLI flags for sandbox launch",
        event="step-parse-cli",
        log_level=log_level,
        include_orphans=args.include_orphans,
        recursive_orphan_scan=args.recursive_orphan_scan,
        monitor_roi_backoff=args.monitor_roi_backoff,
    )
    _emoji_step(
        logger,
        "🌿",
        "Applied sandbox environment toggles",
        event="step-apply-env-flags",
        include_orphans=settings.include_orphans,
        recursive_orphan_scan=settings.recursive_orphan_scan,
        data_dir=os.getenv("SANDBOX_DATA_DIR"),
    )
    logger.info(
        "bootstrap timeout configuration",
        extra=log_record(
            event="bootstrap-timeouts",
            MENACE_BOOTSTRAP_WAIT_SECS=os.getenv("MENACE_BOOTSTRAP_WAIT_SECS"),
            BOOTSTRAP_STEP_TIMEOUT=os.getenv("BOOTSTRAP_STEP_TIMEOUT"),
        ),
    )
    sandbox_restart_total.labels(service="start_autonomous", reason="launch").inc()
    logger.info("sandbox start", extra=log_record(event="start"))

    if system_binaries_validated:
        _emoji_step(
            logger,
            "🧪",
            "System binaries validated (scripts/check_system_binaries.py)",
            event="step-validate-binaries",
        )

    if not args.health_check:
        _preflight_bootstrap_conflicts(logger)

    calibrated_step_budgets, budget_debug = _calibrate_bootstrap_step_budgets(logger)
    calibrated_bootstrap_timeout, timeout_debug = calibrate_overall_timeout(
        base_timeout=args.bootstrap_timeout,
        calibrated_budgets=calibrated_step_budgets,
        max_scale=BUDGET_MAX_SCALE,
    )
    guard_delay, guard_scale = _apply_bootstrap_guard(logger)
    if guard_scale > 1.0:
        calibrated_step_budgets = {
            step: value * guard_scale for step, value in calibrated_step_budgets.items()
        }
        calibrated_bootstrap_timeout *= guard_scale
    if guard_delay > 0:
        logger.info(
            "bootstrap guard delay applied",
            extra=log_record(
                event="bootstrap-guard-applied",
                delay=round(guard_delay, 2),
                adjusted_step_timeout=os.getenv("BOOTSTRAP_STEP_TIMEOUT"),
                budget_scale=guard_scale,
            ),
        )
        print(
            "[BOOTSTRAP] contention detected; bootstrap start staggered and budgets scaled for peers",
            flush=True,
        )
    logger.info(
        "calibrated bootstrap budgets applied",
        extra=log_record(
            event="bootstrap-calibrated-budgets",
            budgets={step: round(value, 2) for step, value in calibrated_step_budgets.items()},
            adjustments=budget_debug.get("adjusted"),
            timeout_baseline=args.bootstrap_timeout,
            timeout_adjusted=calibrated_bootstrap_timeout,
            timeout_context=timeout_debug,
            buffer=BUDGET_BUFFER_MULTIPLIER,
            scale_cap=BUDGET_MAX_SCALE,
            store=str(BOOTSTRAP_DURATION_STORE),
        ),
    )
    _emoji_step(
        logger,
        "⏱️",
        "Calibrated bootstrap budgets and deadlines",
        event="step-calibrate-budgets",
        bootstrap_timeout=round(calibrated_bootstrap_timeout, 2),
        guard_scale=guard_scale,
    )

    heavy_bootstrap_detected, deadline_hints = _detect_heavy_bootstrap_hints(args)
    soft_bootstrap_deadline = _resolve_soft_deadline_flag(args)
    pipeline_complexity = _derive_pipeline_complexity(settings)
    host_telemetry = read_bootstrap_heartbeat() or {}
    try:
        load_average = os.getloadavg()[0]
    except (AttributeError, OSError):
        load_average = None
    stage_deadline_policy, bootstrap_deadline_policy = _resolve_bootstrap_deadline_policy(
        baseline_timeout=calibrated_bootstrap_timeout,
        heavy_detected=heavy_bootstrap_detected,
        soft_deadline=soft_bootstrap_deadline,
        hints=deadline_hints,
        pipeline_complexity=pipeline_complexity,
        host_telemetry=host_telemetry,
        load_average=load_average,
    )
    progress_signal = build_progress_signal_hook(namespace="bootstrap-phases")
    effective_bootstrap_timeout = bootstrap_deadline_policy.get("adjusted_timeout")
    logger.info(
        "bootstrap deadline policy resolved",
        extra=log_record(
            event="bootstrap-deadline-policy",
            policy=bootstrap_deadline_policy,
        ),
    )

    ready_to_launch = True
    roi_backoff_triggered = False
    failure_reasons: list[str] = []
    bootstrap_lock: SandboxLock | None = None
    bootstrap_lock_guard: Any | None = None

    try:
        if not args.health_check:
            last_pre_meta_trace_step = "entering non-health-check bootstrap block"
            try:
                try:
                    _log_watcher_scope_hint(logger)
                    _detect_heavy_watchers(logger)
                    print(
                        "[DEBUG] About to call initialize_bootstrap_context()",
                        flush=True,
                    )
                    logger.info(
                        "initialize_bootstrap_context starting",
                        extra=log_record(
                            event="bootstrap-context-start",
                            health_check=args.health_check,
                        ),
                    )
                    artifacts = _load_bootstrap_artifacts()
                    completed_steps = set(artifacts.get("completed_steps", []) or [])
                    if artifacts.get("status") == "complete":
                        logger.info(
                            "bootstrap artifacts detected; enabling warm-start hints",
                            extra=log_record(
                                event="bootstrap-artifacts-loaded",
                                artifact_path=str(BOOTSTRAP_ARTIFACT_PATH),
                                context_hint=artifacts.get("context_hint"),
                                completed_steps=list(completed_steps),
                            ),
                        )

                    sentinel_wait = _check_bootstrap_sentinel()
                    if sentinel_wait > 0:
                        logger.warning(
                            "bootstrap sentinel requires backoff before restart",
                            extra=log_record(
                                event="bootstrap-sentinel-wait",
                                wait_time=round(sentinel_wait, 2),
                            ),
                        )
                        time.sleep(sentinel_wait)

                    stagger_delay = _determine_bootstrap_stagger()
                    if stagger_delay > 0:
                        logger.info(
                            "applying bootstrap stagger to reduce host contention",
                            extra=log_record(
                                event="bootstrap-stagger",
                                delay=round(stagger_delay, 2),
                                guidance="close broad filesystem watchers or scope them to the active repo",
                            ),
                        )
                        time.sleep(stagger_delay)

                    try:
                        bootstrap_lock, bootstrap_lock_guard = _acquire_bootstrap_lock(logger)
                    except Timeout:
                        ready_to_launch = False
                        failure_reasons.append("bootstrap lock acquisition timed out")
                        logger.error(
                            "bootstrap lock acquisition timed out",
                            extra=log_record(
                                event="bootstrap-lock-timeout",
                                lock_path=str(BOOTSTRAP_LOCK_PATH),
                                timeout=BOOTSTRAP_LOCK_TIMEOUT,
                            ),
                        )
                        sys.exit(1)
                    except Exception:
                        ready_to_launch = False
                        failure_reasons.append("bootstrap lock acquisition failed")
                        logger.exception(
                            "failed to acquire bootstrap lock",
                            extra=log_record(
                                event="bootstrap-lock-error",
                                lock_path=str(BOOTSTRAP_LOCK_PATH),
                                timeout=BOOTSTRAP_LOCK_TIMEOUT,
                            ),
                        )
                        sys.exit(1)

                    max_attempts = max(1, BOOTSTRAP_MAX_RETRIES + 1)
                    bootstrap_context: dict[str, Any] | None = None
                    bootstrap_error: BaseException | None = None
                    bootstrap_seen_steps: set[str] = set()
                    lagging_optional_stages: set[str] = set()

                    for attempt in range(1, max_attempts + 1):
                        last_pre_meta_trace_step = "initialize_bootstrap_context invocation"
                        bootstrap_context_result: dict[str, Any] | None = None
                        bootstrap_error = None
                        bootstrap_start = time.monotonic()
                        deadline_timeout = effective_bootstrap_timeout
                        bootstrap_deadline = (
                            None
                            if deadline_timeout is None
                            else bootstrap_start + deadline_timeout
                        )
                        bootstrap_stop_event = threading.Event()

                        logger.info(
                            "initialize_bootstrap_context deadline configured",
                                extra=log_record(
                                    event="bootstrap-deadline-configured",
                                    deadline=bootstrap_deadline,
                                    deadline_timeout=deadline_timeout,
                                    baseline_timeout=calibrated_bootstrap_timeout,
                                    policy=bootstrap_deadline_policy,
                                ),
                            )

                        def _run_bootstrap() -> None:
                            nonlocal bootstrap_context_result, bootstrap_error
                            try:
                                bootstrap_context_result = initialize_bootstrap_context(
                                    stop_event=bootstrap_stop_event,
                                    bootstrap_deadline=bootstrap_deadline,
                                    heavy_bootstrap=heavy_bootstrap_detected,
                                    stage_deadlines=stage_deadline_policy,
                                    progress_signal=progress_signal,
                                )
                            except BaseException as exc:  # pragma: no cover - propagate errors
                                bootstrap_error = exc

                        bootstrap_thread = threading.Thread(
                            target=_run_bootstrap,
                            name=f"bootstrap-context-{attempt}",
                            daemon=True,
                        )
                        bootstrap_thread.start()
                        print(
                            "[BOOTSTRAP-TRACE] bootstrap thread started "
                            f"(attempt={attempt} elapsed={time.monotonic() - bootstrap_start:.3f}s, "
                            f"last_step={BOOTSTRAP_PROGRESS.get('last_step', 'unknown')})",
                            flush=True,
                        )

                        (
                            timed_out,
                            last_bootstrap_step,
                            elapsed,
                            budget_used,
                            observed_steps,
                            lagging_optional,
                            step_durations,
                            stage_timeout_context,
                            ) = _monitor_bootstrap_thread(
                                bootstrap_thread=bootstrap_thread,
                                bootstrap_stop_event=bootstrap_stop_event,
                                bootstrap_start=bootstrap_start,
                                step_budgets=calibrated_step_budgets,
                                adaptive_grace=ADAPTIVE_LONG_STAGE_GRACE,
                                completed_steps=completed_steps,
                                vector_heavy_steps=VECTOR_HEAVY_STEPS,
                                stage_policy=stage_deadline_policy,
                                stage_signal=progress_signal,
                            )
                        lagging_optional_stages.update(lagging_optional)
                        bootstrap_seen_steps.update(observed_steps)
                        _persist_step_durations(step_durations, logger)

                        if timed_out:
                            bootstrap_stop_event.set()
                            dump_path = Path("maintenance-logs") / (
                                f"bootstrap_timeout_traceback_{int(time.time())}.log"
                            )
                            dump_path.parent.mkdir(exist_ok=True)
                            try:
                                with dump_path.open("w", encoding="utf-8") as dump_file:
                                    dump_file.write(
                                        "Bootstrap timeout thread dump\n"
                                        f"thread={bootstrap_thread.name} "
                                        f"ident={bootstrap_thread.ident}\n"
                                    )
                                    if bootstrap_thread.ident is not None:
                                        frame = sys._current_frames().get(
                                            bootstrap_thread.ident
                                        )
                                        if frame is not None:
                                            dump_file.write(
                                                "\n--- bootstrap thread stack ---\n"
                                            )
                                            traceback.print_stack(
                                                frame, file=dump_file
                                            )
                                        else:
                                            dump_file.write(
                                                "\n[bootstrap thread frame unavailable]\n"
                                            )
                                    dump_file.write(
                                        "\n--- all thread tracebacks ---\n"
                                    )
                                    faulthandler.dump_traceback(
                                        file=dump_file, all_threads=True
                                    )
                                logger.warning(
                                    "captured bootstrap timeout thread dump",
                                    extra=log_record(
                                        event="bootstrap-timeout-thread-dump",
                                        dump_path=str(dump_path),
                                        last_step=last_bootstrap_step,
                                        elapsed=elapsed,
                                    ),
                                )
                            except Exception:
                                logger.exception(
                                    "failed to write bootstrap thread dump after timeout",
                                    extra=log_record(
                                        event="bootstrap-timeout-thread-dump-error",
                                        last_step=last_bootstrap_step,
                                        elapsed=elapsed,
                                    ),
                                )

                            logger.error(
                                "initialize_bootstrap_context exceeded per-step budget",
                                extra=log_record(
                                    event="bootstrap-context-timeout",
                                    elapsed=elapsed,
                                    timeout=deadline_timeout or args.bootstrap_timeout,
                                    last_bootstrap_step=last_bootstrap_step,
                                    budget_used=budget_used,
                                    attempt=attempt,
                                    policy=bootstrap_deadline_policy,
                                    stage_timeout=stage_timeout_context,
                                ),
                            )
                            backoff = _record_bootstrap_timeout(
                                last_step=last_bootstrap_step,
                                elapsed=elapsed,
                                attempt=attempt,
                                step_budget=budget_used,
                                stage_timeout=stage_timeout_context,
                            )
                            try:
                                shutdown_autonomous_sandbox(timeout=5)
                            except Exception:  # pragma: no cover - best effort cleanup
                                logger.exception(
                                    "cleanup after bootstrap timeout failed",
                                    extra=log_record(
                                        event="bootstrap-timeout-cleanup-error"
                                    ),
                                )
                            try:
                                bootstrap_thread.join(2)
                            except Exception:
                                logger.debug(
                                    "bootstrap thread join after timeout raised",
                                    exc_info=True,
                                )
                            if attempt >= max_attempts:
                                bootstrap_manager.mark_ready(
                                    ready=False,
                                    error=(
                                        "initialize_bootstrap_context exceeded timeout; "
                                        f"last_step={last_bootstrap_step} elapsed={elapsed:.1f}s"
                                    ),
                                )
                                raise TimeoutError(
                                    "initialize_bootstrap_context exceeded timeout; "
                                    f"last_step={last_bootstrap_step} elapsed={elapsed:.1f}s"
                                )
                            logger.info(
                                "retrying bootstrap after backoff",  # expose retry policy
                                extra=log_record(
                                    event="bootstrap-retry-backoff",
                                    attempt=attempt,
                                    backoff_seconds=backoff,
                                    last_step=last_bootstrap_step,
                                ),
                            )
                            time.sleep(backoff)
                            continue

                        bootstrap_thread.join(2)
                        if bootstrap_error:
                            if attempt >= max_attempts:
                                bootstrap_manager.mark_ready(
                                    ready=False, error=str(bootstrap_error)
                                )
                                raise bootstrap_error
                            delay = min(
                                BOOTSTRAP_BACKOFF_BASE * attempt, BOOTSTRAP_BACKOFF_MAX
                            )
                            logger.warning(
                                "bootstrap raised; retrying after backoff",
                                extra=log_record(
                                    event="bootstrap-exception-retry",
                                    attempt=attempt,
                                    delay=delay,
                                    error=str(bootstrap_error),
                                ),
                            )
                            time.sleep(delay)
                            continue

                        bootstrap_context = bootstrap_context_result
                        _persist_bootstrap_artifacts(
                            last_step=last_bootstrap_step,
                            bootstrap_context=bootstrap_context or {},
                            completed_steps=bootstrap_seen_steps or observed_steps,
                        )
                        _reset_bootstrap_sentinel()
                        bootstrap_manager.mark_ready()
                        break

                    if bootstrap_context is None and bootstrap_error:
                        bootstrap_manager.mark_ready(
                            ready=False, error=str(bootstrap_error)
                        )
                        raise bootstrap_error

                    bootstrap_elapsed = time.monotonic() - bootstrap_start
                    if args.bootstrap_timeout and bootstrap_elapsed > args.bootstrap_timeout:
                        logger.warning(
                            "bootstrap exceeded original budget but allowed by relaxed deadline",
                            extra=log_record(
                                event="bootstrap-deadline-stretched",
                                elapsed=bootstrap_elapsed,
                                original_budget=args.bootstrap_timeout,
                                adjusted_timeout=deadline_timeout,
                                heavy_bootstrap=heavy_bootstrap_detected,
                                policy=bootstrap_deadline_policy,
                                hints=deadline_hints,
                            ),
                        )

                    online_state_snapshot = dict(BOOTSTRAP_ONLINE_STATE)
                    pending_optional = lagging_optional_components(online_state_snapshot)
                    pending_optional.update(lagging_optional_stages)
                    if pending_optional:
                        logger.warning(
                            "optional bootstrap stages still warming",
                            extra=log_record(
                                event="bootstrap-optional-pending",
                                pending=list(sorted(pending_optional)),
                                online_state=online_state_snapshot,
                            ),
                        )
                        print(
                            "[BOOTSTRAP] optional stages pending: %s"
                            % ", ".join(sorted(pending_optional)),
                            flush=True,
                        )

                    print(
                        "[DEBUG] initialize_bootstrap_context completed successfully",
                        flush=True,
                    )
                    logger.info(
                        "initialize_bootstrap_context completed",
                        extra=log_record(event="bootstrap-context-complete"),
                    )
                    os.environ.setdefault("META_PLANNING_LOOP", "1")
                    os.environ.setdefault("META_PLANNING_INTERVAL", "10")
                    os.environ.setdefault("META_IMPROVEMENT_THRESHOLD", "0.01")
                    _emit_meta_trace(
                        logger,
                        "preparing meta planning environment",
                        loop=os.environ.get("META_PLANNING_LOOP"),
                        interval=os.environ.get("META_PLANNING_INTERVAL"),
                        improvement_threshold=os.environ.get("META_IMPROVEMENT_THRESHOLD"),
                    )
                    last_pre_meta_trace_step = "importing self_improvement.meta_planning"
                    from self_improvement import meta_planning
                    print(
                        "[META-TRACE] meta_planning module import completed; capturing module attributes",
                        flush=True,
                    )
                    logger.info(
                        "meta_planning module import completed; enumerating attributes",
                        extra=log_record(
                            event="meta-planning-import-finished",
                            attr_count=len(dir(meta_planning)),
                            attrs_preview=list(sorted(dir(meta_planning)))[:25],
                        ),
                    )
                    logger.info(
                        "meta_planning module imported for autonomous sandbox",
                        extra=log_record(
                            event="meta-planning-import",
                            module=str(meta_planning),
                            module_dir=list(sorted(dir(meta_planning))),
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "meta planning module imported",
                        module=str(meta_planning),
                        meta_planning_interval=os.environ.get("META_PLANNING_INTERVAL"),
                    )
                    last_pre_meta_trace_step = "importing self_improvement_cycle"
                    from self_improvement.meta_planning import (  # noqa: F401
                        self_improvement_cycle,
                    )
                    print(
                        "[META-TRACE] self_improvement_cycle imported; meta planner wiring begins",
                        flush=True,
                    )
                    logger.info(
                        "self_improvement_cycle imported; preparing to reload settings",
                        extra=log_record(
                            event="meta-planning-cycle-imported",
                            module_has_cycle=hasattr(meta_planning, "self_improvement_cycle"),
                        ),
                    )
    
                    last_pre_meta_trace_step = "reloading meta_planning settings"
                    meta_planning.reload_settings(settings)
                    print(
                        "[META-TRACE] meta_planning.reload_settings invoked; settings synchronized",
                        flush=True,
                    )
                    logger.info(
                        "meta planning settings synchronized",
                        extra=log_record(
                            event="meta-planning-settings-reloaded",
                            include_orphans=settings.include_orphans,
                            recursive_orphans=settings.recursive_orphan_scan,
                            sandbox_log_level=settings.sandbox_log_level,
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "meta planning settings reloaded",
                        include_orphans=settings.include_orphans,
                        recursive_orphans=settings.recursive_orphan_scan,
                        log_level=settings.sandbox_log_level,
                    )
                    workflow_evolver = WorkflowEvolutionManager()
                    print(
                        "[META-TRACE] WorkflowEvolutionManager instantiated; preparing planner resolution",
                        flush=True,
                    )
                    _emit_meta_trace(
                        logger,
                        "workflow evolver instantiated for meta planning",
                        evolver_class=WorkflowEvolutionManager.__name__,
                    )
                    print(
                        "[META-TRACE] workflow evolver ready; resolving planner with force reload",
                        flush=True,
                    )
                    planner_cls = meta_planning.resolve_meta_workflow_planner(
                        force_reload=True
                    )
                    logger.info(
                        "meta workflow planner resolved",  # dense log for planner resolution
                        extra=log_record(
                            event="meta-planning-planner-resolved",
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                            force_reload=True,
                        ),
                    )
                    print(
                        "[META-TRACE] meta workflow planner resolution finished; class=%s"
                        % getattr(planner_cls, "__name__", str(planner_cls)),
                        flush=True,
                    )
                    logger.info(
                        "meta workflow planner resolution detailed trace",
                        extra=log_record(
                            event="meta-planning-planner-resolution-detail",
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                            planner_module=getattr(planner_cls, "__module__", None),
                            planner_dict=sorted(list(getattr(planner_cls, "__dict__", {}).keys())),
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "meta workflow planner resolution attempted",
                        planner_resolved=planner_cls is not None,
                        planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                    )
                    if planner_cls is None:
                        logger.error(
                            "MetaWorkflowPlanner not found; aborting sandbox launch",
                            extra=log_record(event="meta-planning-missing"),
                        )
                        print(
                            "[META-TRACE] planner resolution failed; aborting launch pipeline",
                            flush=True,
                        )
                        sys.exit(1)
    
                    interval = float(
                        os.getenv(
                            "META_PLANNING_INTERVAL",
                            getattr(settings, "meta_planning_interval", 10),
                        )
                    )
                    logger.info(
                        "meta planning cadence calculated",
                        extra=log_record(
                            event="meta-planning-interval",
                            interval=interval,
                            source_env=os.getenv("META_PLANNING_INTERVAL"),
                            settings_interval=getattr(settings, "meta_planning_interval", None),
                            settings_namespace=vars(settings),
                        ),
                    )
                    print(
                        "[META-TRACE] meta planning interval established at %.2fs" % interval,
                        flush=True,
                    )
                    logger.info(
                        "meta planning cadence fully resolved with environment and settings context",
                        extra=log_record(
                            event="meta-planning-interval-detail",
                            interval=interval,
                            env_interval=os.getenv("META_PLANNING_INTERVAL"),
                            env_loop=os.getenv("META_PLANNING_LOOP"),
                            improvement_threshold=os.getenv("META_IMPROVEMENT_THRESHOLD"),
                        ),
                    )
                    discovered_specs = []
                    try:
                        discovered_specs = discover_workflow_specs(logger=logger)
                        logger.info(
                            "workflow discovery completed",
                            extra=log_record(
                                event="workflow-discovery-complete",
                                discovered_count=len(discovered_specs),
                            ),
                        )
                        _emit_meta_trace(
                            logger,
                            "workflow discovery completed",
                            discovered_count=len(discovered_specs),
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                        )
                        print(
                            "[META-TRACE] workflow discovery finished; discovered=%d"
                            % len(discovered_specs),
                            flush=True,
                        )
                    except Exception:
                        logger.exception(
                            "failed to auto-discover workflow specs",
                            extra=log_record(event="workflow-discovery-error"),
                        )
                    print(
                        "[META-TRACE] workflow discovery post-processing; specs=%s"
                        % [spec.get("workflow_id") for spec in discovered_specs],
                        flush=True,
                    )
                    logger.info(
                        "workflow discovery snapshot",
                        extra=log_record(
                            event="workflow-discovery-snapshot",
                            discovered_ids=[spec.get("workflow_id") for spec in discovered_specs],
                            discovered_preview=discovered_specs[:3],
                        ),
                    )
                    _emoji_step(
                        logger,
                        "🧭",
                        "Discovered workflow specifications",
                        event="step-discover-workflows",
                        discovered_count=len(discovered_specs),
                    )
    
                    orphan_specs: list[Mapping[str, Any]] = []
                    include_orphans = bool(
                        getattr(settings, "include_orphans", False)
                        and not getattr(settings, "disable_orphans", False)
                    )
                    recursive_orphans = bool(
                        getattr(settings, "recursive_orphan_scan", False)
                    )
                    logger.info(
                        "orphan inclusion parameters evaluated",
                        extra=log_record(
                            event="orphan-parameters",
                            include_orphans=include_orphans,
                            recursive_orphans=recursive_orphans,
                        ),
                    )
                    print(
                        "[META-TRACE] orphan settings finalized; include=%s recursive=%s"
                        % (include_orphans, recursive_orphans),
                        flush=True,
                    )
                    logger.info(
                        "orphan settings finalized for meta planning",
                        extra=log_record(
                            event="orphan-settings-finalized",
                            include_orphans=include_orphans,
                            recursive_orphans=recursive_orphans,
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                        ),
                    )
                    if include_orphans:
                        try:
                            orphan_modules = integrate_orphans_sync(
                                recursive=recursive_orphans
                            )
                            orphan_specs.extend(
                                {
                                    "workflow": [module],
                                    "workflow_id": module,
                                    "task_sequence": [module],
                                    "source": "orphan_discovery",
                                }
                                for module in orphan_modules
                                if isinstance(module, str)
                            )
                            logger.info(
                                "orphan integration completed",
                                extra=log_record(
                                    event="orphan-integration-complete",
                                    orphan_modules=orphan_modules,
                                    orphan_spec_count=len(orphan_specs),
                                    recursive=recursive_orphans,
                                ),
                            )
                            print(
                                "[META-TRACE] orphan integration complete; modules=%s specs=%d"
                                % (orphan_modules, len(orphan_specs)),
                                flush=True,
                            )
                            _emoji_step(
                                logger,
                                "🧩",
                                "Included orphan modules in workflow set",
                                event="step-include-orphans",
                                recursive=recursive_orphans,
                                orphan_count=len(orphan_specs),
                            )
                        except Exception:
                            logger.exception(
                                "startup orphan integration failed",
                                extra=log_record(event="startup-orphan-discovery"),
                            )
    
                    if include_orphans:
                        print(
                            "[META-TRACE] orphan discovery sequence completed; total specs now=%d" % len(orphan_specs),
                            flush=True,
                        )
                        if recursive_orphans:
                            try:
                                result = post_round_orphan_scan(recursive=True)
                                integrated = (
                                    result.get("integrated")
                                    if isinstance(result, Mapping)
                                    else None
                                )
                                if integrated:
                                    orphan_specs.extend(
                                        {
                                            "workflow": [module],
                                            "workflow_id": module,
                                            "task_sequence": [module],
                                            "source": "recursive_orphan_discovery",
                                        }
                                        for module in integrated
                                        if isinstance(module, str)
                                    )
                                    logger.info(
                                        "recursive orphan scan integrated modules",
                                        extra=log_record(
                                            event="recursive-orphan-scan",
                                            integrated=integrated,
                                            orphan_spec_count=len(orphan_specs),
                                        ),
                                    )
                                    print(
                                        "[META-TRACE] recursive orphan scan added modules=%s"
                                        % integrated,
                                        flush=True,
                                    )
                            except Exception:
                                logger.exception(
                                    "startup recursive orphan scan failed",
                                    extra=log_record(event="startup-orphan-recursive"),
                                )
    
                    _emit_meta_trace(
                        logger,
                        "orphan integration complete",
                        include_orphans=include_orphans,
                        recursive_orphans=recursive_orphans,
                        orphan_specs=len(orphan_specs),
                    )
                    print(
                        "[META-TRACE] orphan integration trace emitted; combined specs=%d"
                        % (len(discovered_specs) + len(orphan_specs)),
                        flush=True,
                    )
    
                    workflows, workflow_graph_obj = _build_self_improvement_workflows(
                        bootstrap_context,
                        settings,
                        workflow_evolver,
                        logger=logger,
                        discovered_specs=[*discovered_specs, *orphan_specs],
                    )
                    print(
                        "[META-TRACE] self-improvement workflows constructed; workflow_count=%d graph_nodes=%d"
                        % (
                            len(workflows),
                            len(getattr(workflow_graph_obj, "graph", {}) or {}),
                        ),
                        flush=True,
                    )
                    logger.info(
                        "self-improvement workflows constructed for meta planner",
                        extra=log_record(
                            event="workflows-constructed",
                            workflow_ids=list(workflows.keys()),
                            graph_summary=getattr(workflow_graph_obj, "graph", {}),
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "workflows built for meta planning",
                        workflow_count=len(workflows),
                        graph_nodes=len(getattr(workflow_graph_obj, "graph", {})),
                        planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                    )
                    print(
                        "[META-TRACE] workflow build meta trace emitted; planner=%s"
                        % getattr(planner_cls, "__name__", str(planner_cls)),
                        flush=True,
                    )
                    logger.info(
                        "workflow registration result",
                        extra=log_record(
                            event="workflow-registration",
                            workflow_count=len(workflows),
                            planner_available=planner_cls is not None,
                        ),
                    )
                    logger.info(
                        "workflow registration snapshot",
                        extra=log_record(
                            event="workflow-registration-detail",
                            workflow_keys=list(workflows.keys()),
                            workflow_graph_nodes=len(
                                getattr(workflow_graph_obj, "graph", {}) or {}
                            ),
                        ),
                    )
                    if not workflows:
                        logger.error(
                            "no workflows discovered; startup halted before launching sandbox",
                            extra=log_record(
                                event="startup-no-workflows",
                                planner_available=planner_cls is not None,
                            ),
                        )
                        sys.exit(1)
                    if planner_cls is None:
                        logger.error(
                            "planner resolution failed; cannot coordinate ROI for launch",
                            extra=log_record(event="startup-no-planner", workflow_count=len(workflows)),
                        )
                        sys.exit(1)
                    bootstrap_mode = not _roi_baseline_available()
                    logger.info(
                        "evaluating sandbox startup readiness",
                        extra=log_record(
                            event="startup-readiness",
                            workflow_count=len(workflows),
                            planner_available=planner_cls is not None,
                            bootstrap_mode=bootstrap_mode,
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "startup readiness evaluation beginning",
                        workflow_count=len(workflows),
                        planner_available=planner_cls is not None,
                        bootstrap_mode=bootstrap_mode,
                    )
                    print(
                        "[META-TRACE] startup readiness evaluation initiated; workflows=%d planner=%s bootstrap=%s"
                        % (
                            len(workflows),
                            getattr(planner_cls, "__name__", str(planner_cls)),
                            bootstrap_mode,
                        ),
                        flush=True,
                    )
                    readiness_error: str | None = None
                    logger.info(
                        "🧭 meta-planning gate: beginning last-mile checks before launch",
                        extra=log_record(
                            event="meta-planning-gate-begin",
                            workflow_ids=list(workflows.keys()),
                            planner_resolved=planner_cls is not None,
                            bootstrap_mode=bootstrap_mode,
                        ),
                    )
                    try:
                        ready_to_launch, roi_backoff_triggered = _run_prelaunch_improvement_cycles(
                            workflows,
                            planner_cls=planner_cls,
                            settings=settings,
                            logger=logger,
                            bootstrap_mode=bootstrap_mode,
                        )
                        print(
                            "[META-TRACE] prelaunch ROI cycles completed; ready=%s backoff=%s"
                            % (ready_to_launch, roi_backoff_triggered),
                            flush=True,
                        )
                        logger.info(
                            "✅ prelaunch ROI cycles finished without raising",  # emoji for quick scanning
                            extra=log_record(
                                event="startup-prelaunch-success",
                                ready_to_launch=ready_to_launch,
                                roi_backoff=roi_backoff_triggered,
                                workflow_count=len(workflows),
                                planner_available=planner_cls is not None,
                            ),
                        )
                    except RuntimeError as exc:
                        readiness_error = str(exc)
                        ready_to_launch = False
                        roi_backoff_triggered = False
                        logger.error(
                            "❌ runtime error during prelaunch ROI cycles",  # emoji for quick scanning
                            extra=log_record(
                                event="startup-prelaunch-runtime-error",
                                readiness_error=readiness_error,
                                workflow_count=len(workflows),
                                planner_available=planner_cls is not None,
                            ),
                        )
                    except Exception as exc:
                        readiness_error = f"unexpected prelaunch failure: {exc}"
                        ready_to_launch = False
                        roi_backoff_triggered = False
                        logger.exception(
                            "❌ unexpected exception during prelaunch ROI cycles",  # emoji for quick scanning
                            extra=log_record(
                                event="startup-prelaunch-unexpected-error",
                                readiness_error=readiness_error,
                                workflow_count=len(workflows),
                                planner_available=planner_cls is not None,
                            ),
                        )
                    finally:
                        logger.info(
                            "ℹ️ prelaunch ROI cycle invocation finished",
                            extra=log_record(
                                event="startup-prelaunch-finished",
                                ready_to_launch=ready_to_launch,
                                roi_backoff=roi_backoff_triggered,
                                readiness_error=readiness_error,
                            ),
                        )
                        print(
                            "[META-TRACE] prelaunch ROI cycle finished; ready=%s backoff=%s error=%s"
                            % (ready_to_launch, roi_backoff_triggered, readiness_error),
                            flush=True,
                        )
                        _emit_meta_trace(
                            logger,
                            "prelaunch ROI cycle invocation finished",
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                        )
    
                    if (
                        not ready_to_launch
                        and bootstrap_mode
                        and not roi_backoff_triggered
                        and planner_cls is not None
                        and workflows
                    ):
                        logger.info(
                            "✅ bootstrap mode overriding diminishing returns gate; ROI baseline unavailable",
                            extra=log_record(
                                event="startup-bootstrap-diminishing-bypass",
                                workflow_count=len(workflows),
                            ),
                        )
                        ready_to_launch = True
    
                    if not ready_to_launch:
                        failure_reasons: list[str] = []
                        if not workflows:
                            failure_reasons.append("no workflows discovered")
                        if planner_cls is None:
                            failure_reasons.append("MetaWorkflowPlanner unavailable")
                        if readiness_error:
                            failure_reasons.append(readiness_error)
                        elif roi_backoff_triggered:
                            failure_reasons.append("ROI backoff triggered before launch")
                        else:
                            failure_reasons.append("ROI gate not satisfied")
    
                        logger.error(
                            "❌ sandbox readiness failed; aborting launch: %s",
                            "; ".join(failure_reasons),
                            extra=log_record(
                                event="startup-readiness-failed",
                                failure_reasons=failure_reasons,
                                planner_available=planner_cls is not None,
                                workflow_count=len(workflows),
                                roi_backoff=roi_backoff_triggered,
                            ),
                        )
                        sys.exit(1)
    
                    meta_planning.reload_settings(settings)
                    logger.info(
                        "startup readiness evaluation complete",
                        extra=log_record(
                            event="startup-readiness-result",
                            workflow_count=len(workflows),
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            planner_available=planner_cls is not None,
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "startup readiness evaluation complete",
                        workflow_count=len(workflows),
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        planner_available=planner_cls is not None,
                    )
                    logger.info(
                        "🔬 meta-planning readiness diagnostics collected",
                        extra=log_record(
                            event="meta-planning-readiness-diagnostics",
                            planner_status="✅ available" if planner_cls else "❌ missing",
                            workflow_status="✅ present" if workflows else "❌ none discovered",
                            roi_gate="✅ clear" if ready_to_launch else "❌ blocked",
                            roi_backoff="✅ none" if not roi_backoff_triggered else "❌ backoff",
                            readiness_error=readiness_error,
                            workflow_ids=list(workflows.keys()),
                        ),
                    )
                    logger.info(
                        "🧭 meta-planning gate: evaluating final decision criteria",
                        extra=log_record(
                            event="meta-planning-gate-eval",
                            has_workflows=bool(workflows),
                            planner_resolved=planner_cls is not None,
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            bootstrap_mode=bootstrap_mode,
                        ),
                    )
                    logger.info(
                        "✅ checkpoint: workflows present" if workflows else "❌ checkpoint failed: no workflows present",
                        extra=log_record(
                            event="meta-planning-gate-workflows",
                            condition_passed=bool(workflows),
                            workflow_count=len(workflows),
                        ),
                    )
                    logger.info(
                        "✅ checkpoint: planner resolved" if planner_cls is not None else "❌ checkpoint failed: planner missing",
                        extra=log_record(
                            event="meta-planning-gate-planner",
                            condition_passed=planner_cls is not None,
                            planner_cls=str(planner_cls),
                        ),
                    )
                    logger.info(
                        "✅ checkpoint: prelaunch ROI gate cleared" if ready_to_launch else "❌ checkpoint failed: ROI gate blocked",
                        extra=log_record(
                            event="meta-planning-gate-roi-ready",
                            condition_passed=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                        ),
                    )
                    logger.info(
                        "✅ checkpoint: no ROI backoff detected" if not roi_backoff_triggered else "❌ checkpoint failed: ROI backoff active",
                        extra=log_record(
                            event="meta-planning-gate-roi-backoff",
                            condition_passed=not roi_backoff_triggered,
                            roi_backoff=roi_backoff_triggered,
                        ),
                    )
                    logger.info(
                        "🔎 meta-planning launch decision inputs gathered",
                        extra=log_record(
                            event="meta-planning-gate-inputs",
                            ready_to_launch=ready_to_launch,
                            planner_resolved=planner_cls is not None,
                            workflow_count=len(workflows),
                            bootstrap_mode=bootstrap_mode,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                        ),
                    )
                    logger.info(
                        "🔍 evaluating meta planning launch gate",
                        extra=log_record(
                            event="meta-planning-launch-eval",
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            planner_resolved=planner_cls is not None,
                            workflow_count=len(workflows),
                            workflow_ids=list(workflows.keys()),
                            bootstrap_mode=bootstrap_mode,
                            readiness_error=readiness_error,
                        ),
                    )
                    logger.info(
                        "🔁 meta-planning gate status report (workflows=%s, planner=%s, backoff=%s, ready=%s)",
                        len(workflows),
                        bool(planner_cls),
                        roi_backoff_triggered,
                        ready_to_launch,
                        extra=log_record(
                            event="meta-planning-gate-status",
                            workflow_count=len(workflows),
                            planner_resolved=planner_cls is not None,
                            roi_backoff=roi_backoff_triggered,
                            ready_to_launch=ready_to_launch,
                            readiness_error=readiness_error,
                        ),
                    )
                    logger.info(
                        "🔎 meta planning gate checkpoints: workflows=%d, planner=%s, ready=%s, backoff=%s",
                        len(workflows),
                        bool(planner_cls),
                        ready_to_launch,
                        roi_backoff_triggered,
                        extra=log_record(
                            event="meta-planning-gate-checkpoints",
                            workflow_count=len(workflows),
                            planner_resolved=planner_cls is not None,
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                        ),
                    )
                    gating_checklist = {
                        "workflows_present": bool(workflows),
                        "planner_resolved": planner_cls is not None,
                        "roi_gate_clear": ready_to_launch,
                        "roi_backoff_clear": not roi_backoff_triggered,
                    }
                    for check, passed in gating_checklist.items():
                        logger.info(
                            "✅ gating checkpoint passed: %s" % check
                            if passed
                            else "❌ gating checkpoint failed: %s" % check,
                            extra=log_record(
                                event="meta-planning-gate-check", check=check, passed=passed
                            ),
                        )
                    logger.info(
                        "🔦 meta-planning gate checklist compiled",
                        extra=log_record(event="meta-planning-gate-checklist", **gating_checklist),
                    )
                    logger.info(
                        "✅ meta planning gate decision computed; entering final launch guard",
                        extra=log_record(
                            event="meta-planning-gate-decision",
                            workflows_present=bool(workflows),
                            planner_available=planner_cls is not None,
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                            bootstrap_mode=bootstrap_mode,
                        ),
                    )
                    if not workflows:
                        logger.error(
                            "❌ gating halted: no workflows discovered for meta planning",
                            extra=log_record(event="meta-planning-gate-no-workflows"),
                        )
                    if planner_cls is None:
                        logger.error(
                            "❌ gating halted: MetaWorkflowPlanner unresolved",
                            extra=log_record(event="meta-planning-gate-no-planner"),
                        )
                    if roi_backoff_triggered:
                        logger.error(
                            "❌ gating halted: ROI backoff triggered before launch",
                            extra=log_record(event="meta-planning-gate-backoff"),
                        )
                    if readiness_error:
                        logger.error(
                            "❌ gating halted: readiness error encountered",
                            extra=log_record(
                                event="meta-planning-gate-readiness-error",
                                readiness_error=readiness_error,
                            ),
                        )
                    failure_reasons = []
                    if not workflows:
                        failure_reasons.append("no workflows discovered for meta planning")
                    if planner_cls is None:
                        failure_reasons.append("MetaWorkflowPlanner unresolved")
                    if roi_backoff_triggered:
                        failure_reasons.append("ROI backoff triggered before launch")
                    if readiness_error:
                        failure_reasons.append(readiness_error)
                    if not ready_to_launch and not readiness_error:
                        failure_reasons.append(
                            "workflows did not meet diminishing returns threshold"
                        )
                    logger.info(
                        "🔦 meta-planning gate failure reasons compiled",
                        extra=log_record(
                            event="meta-planning-gate-failure-reasons",
                            failure_reasons=failure_reasons,
                            checklist=gating_checklist,
                        ),
                    )
                    logger.info(
                        "🧭 meta-planning gate summary computed; preparing branch selection",
                        extra=log_record(
                            event="meta-planning-gate-branch-summary",
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            planner_available=planner_cls is not None,
                            workflows_present=bool(workflows),
                            failure_reasons=failure_reasons,
                        ),
                    )
                    logger.info(
                        "🔎 launch condition breakdown: workflows=%s, planner=%s, roi_backoff=%s, readiness_error=%s",
                        bool(workflows),
                        bool(planner_cls),
                        roi_backoff_triggered,
                        readiness_error,
                        extra=log_record(
                            event="meta-planning-launch-breakdown",
                            has_workflows=bool(workflows),
                            planner_available=planner_cls is not None,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                        ),
                    )
                    if ready_to_launch:
                        logger.info(
                            "🚦 meta planning launch block reached; beginning verbose instrumentation",
                            extra=log_record(
                                event="meta-planning-launch-block-entry",
                                workflow_count=len(workflows),
                                planner_cls=str(planner_cls),
                                roi_backoff=roi_backoff_triggered,
                                readiness_error=readiness_error,
                                bootstrap_mode=bootstrap_mode,
                                correlation_id=cid,
                            ),
                        )
                        logger.info(
                            "meta planning loop prerequisites verified",
                            extra=log_record(
                                event="meta-planning-loop-prereq",
                                workflow_count=len(workflows),
                                planner_available=planner_cls is not None,
                            ),
                        )
                        if not workflows or planner_cls is None:
                            logger.error(
                                "❌ meta planning loop prerequisites missing; aborting start",
                                extra=log_record(
                                    event="meta-planning-loop-prereq-missing",
                                    workflow_count=len(workflows),
                                    planner_available=planner_cls is not None,
                                ),
                            )
                            sys.exit(1)
                        logger.info(
                            "✅ gating green: all launch conditions satisfied; proceeding to thread bootstrap",
                            extra=log_record(
                                event="meta-planning-gate-green",
                                workflow_count=len(workflows),
                                planner_resolved=planner_cls is not None,
                                roi_backoff=roi_backoff_triggered,
                            ),
                        )
                        logger.info(
                            "🧭 meta planning start: entering bootstrap+thread block (expect subsequent checkpoints)",
                            extra=log_record(
                                event="meta-planning-start-block-enter",
                                workflow_ids=list(workflows.keys()),
                                planner_resolved=planner_cls is not None,
                                interval_seconds=interval,
                                bootstrap_mode=bootstrap_mode,
                                roi_backoff=roi_backoff_triggered,
                                ready_to_launch=ready_to_launch,
                            ),
                        )
                        logger.info(
                            "✅ prelaunch checks passed; proceeding with meta-planning start sequence",
                            extra=log_record(
                                event="meta-planning-launch-sequence-begin",
                                workflow_ids=list(workflows.keys()),
                                planner_class=str(planner_cls),
                                roi_backoff=roi_backoff_triggered,
                                bootstrap_mode=bootstrap_mode,
                            ),
                        )
                        logger.info(
                            "✅ launch gate green: ROI stagnation satisfied and planner resolved",
                            extra=log_record(
                                event="meta-planning-launch-green",
                                workflow_ids=list(workflows.keys()),
                                planner_cls=str(planner_cls),
                                roi_backoff=roi_backoff_triggered,
                            ),
                        )
                        logger.info(
                            "✅ readiness gate cleared; preparing to start meta planning loop",
                            extra=log_record(
                                event="meta-planning-ready",
                                roi_backoff=roi_backoff_triggered,
                                workflow_count=len(workflows),
                                planner_resolved=planner_cls is not None,
                                bootstrap_mode=bootstrap_mode,
                                prelaunch_ready=ready_to_launch,
                            ),
                        )
                        logger.info(
                            "✅ meta-planning launch prerequisites satisfied",
                            extra=log_record(
                                event="meta-planning-prereqs",
                                planner_cls=str(planner_cls),
                                interval_seconds=interval,
                                workflow_ids=list(workflows.keys()),
                                workflow_graph_built=workflow_graph_obj is not None,
                            ),
                        )
                        logger.info(
                            "✅ gating checklist satisfied; proceeding to meta planning bootstrap",
                            extra=log_record(
                                event="meta-planning-gate-green-checklist",
                                checklist=gating_checklist,
                                workflow_ids=list(workflows.keys()),
                            ),
                        )
                        logger.info(
                            "✅ meta planning gate satisfied; initializing launch choreography",
                            extra=log_record(
                                event="meta-planning-gate-satisfied",
                                workflow_ids=list(workflows.keys()),
                                planner_resolved=planner_cls is not None,
                                bootstrap_mode=bootstrap_mode,
                            ),
                        )
                        logger.info(
                            "🔧 configuring meta planning loop thread creation",
                            extra=log_record(
                                event="meta-planning-thread-config",
                                interval_seconds=interval,
                                workflow_graph_built=workflow_graph_obj is not None,
                                workflow_count=len(workflows),
                            ),
                        )
                        logger.info(
                            "🧠 preparing to invoke start_self_improvement_cycle() with event bus and workflow graph",
                            extra=log_record(
                                event="meta-planning-pre-bootstrap-call",
                                workflow_ids=list(workflows.keys()),
                                planner_cls=str(planner_cls),
                                interval_seconds=interval,
                                workflow_graph_present=workflow_graph_obj is not None,
                                event_bus_available=shared_event_bus is not None,
                                workflow_graph_nodes=
                                    list(workflow_graph_obj.keys())
                                    if isinstance(workflow_graph_obj, Mapping)
                                    else None,
                                workflow_graph_type=type(workflow_graph_obj).__name__,
                                workflow_graph_is_graph=(
                                    getattr(workflow_graph_obj, "graph", None) is not None
                                ),
                                event_bus_type=type(shared_event_bus).__name__
                                if shared_event_bus is not None
                                else None,
                                event_bus_handlers=getattr(
                                    shared_event_bus, "listeners", None
                                ),
                            ),
                        )
                        logger.info(
                            "🛰️ verifying meta_planning module attributes prior to start_self_improvement_cycle()",
                            extra=log_record(
                                event="meta-planning-module-precheck",
                                module_dir=list(sorted(dir(meta_planning))),
                                has_reload_settings=hasattr(meta_planning, "reload_settings"),
                                has_self_improvement_cycle=hasattr(
                                    meta_planning, "start_self_improvement_cycle"
                                ),
                                callable_self_improvement_cycle=callable(
                                    getattr(meta_planning, "start_self_improvement_cycle", None)
                                ),
                            ),
                        )
                        try:
                            logger.info(
                                "🔧 invoking meta planning loop bootstrap",
                                extra=log_record(
                                    event="meta-planning-bootstrap-call",
                                    workflow_count=len(workflows),
                                    planner_cls=str(planner_cls),
                                    interval_seconds=interval,
                                    workflow_graph_present=workflow_graph_obj is not None,
                                    workflow_graph_len=len(workflow_graph_obj or {}),
                                    event_bus_connected=shared_event_bus is not None,
                                ),
                            )
                            logger.info(
                                "🛰️ deep-dive meta planning bootstrap parameter snapshot",
                                extra=log_record(
                                    event="meta-planning-bootstrap-param-snapshot",
                                    workflow_keys=list(workflows.keys()),
                                    workflow_len=len(workflows),
                                    planner_cls=str(planner_cls),
                                    interval_seconds=interval,
                                    workflow_graph_type=type(workflow_graph_obj).__name__,
                                    workflow_graph_keys=list((workflow_graph_obj or {}).keys())
                                    if isinstance(workflow_graph_obj, Mapping)
                                    else None,
                                    event_bus_type=type(shared_event_bus).__name__
                                    if shared_event_bus is not None
                                    else None,
                                    event_bus_has_listeners=bool(
                                        getattr(shared_event_bus, "listeners", None)
                                    ),
                                ),
                            )
                            logger.info(
                                "🛰️ recording meta planning invocation parameters for traceability",
                                extra=log_record(
                                    event="meta-planning-bootstrap-args",
                                    workflow_ids=list(workflows.keys()),
                                    workflow_count=len(workflows),
                                    interval_seconds=interval,
                                    planner_cls=str(planner_cls),
                                    workflow_graph_repr=repr(workflow_graph_obj),
                                    event_bus_repr=repr(shared_event_bus),
                                ),
                            )
                            logger.info(
                                "🛰️ meta planning bootstrap call about to execute start_self_improvement_cycle()",
                                extra=log_record(
                                    event="meta-planning-bootstrap-about-to-call",
                                    workflow_count=len(workflows),
                                    planner_cls=str(planner_cls),
                                    interval_seconds=interval,
                                    workflow_graph_keys=list((workflow_graph_obj or {}).keys())
                                    if isinstance(workflow_graph_obj, Mapping)
                                    else None,
                                    workflow_graph_type=type(workflow_graph_obj).__name__,
                                    event_bus_type=type(shared_event_bus).__name__
                                    if shared_event_bus is not None
                                    else None,
                                ),
                            )
                            logger.info(
                                "🛰️ verifying start_self_improvement_cycle callable availability before invoke",
                                extra=log_record(
                                    event="meta-planning-bootstrap-pre-call-verify",
                                    callable_present=hasattr(meta_planning, "start_self_improvement_cycle"),
                                    planner_cls=str(planner_cls),
                                    workflow_count=len(workflows),
                                    callable_object=getattr(
                                        meta_planning, "start_self_improvement_cycle", None
                                    ),
                                    callable_is_function=callable(
                                        getattr(
                                            meta_planning, "start_self_improvement_cycle", None
                                        )
                                    ),
                                ),
                            )
                            if not hasattr(meta_planning, "start_self_improvement_cycle"):
                                logger.error(
                                    "❌ start_self_improvement_cycle missing on meta_planning module",
                                    extra=log_record(
                                        event="meta-planning-missing-entrypoint",
                                        module_dir=list(dir(meta_planning)),
                                        planner_cls=str(planner_cls),
                                    ),
                                )
                                raise RuntimeError(
                                    "start_self_improvement_cycle missing on meta_planning"
                                )
                            if not callable(
                                getattr(meta_planning, "start_self_improvement_cycle", None)
                            ):
                                logger.error(
                                    "❌ start_self_improvement_cycle present but not callable",
                                    extra=log_record(
                                        event="meta-planning-entrypoint-not-callable",
                                        type_info=type(
                                            getattr(
                                                meta_planning, "start_self_improvement_cycle", None
                                            )
                                        ).__name__,
                                        planner_cls=str(planner_cls),
                                    ),
                                )
                                raise RuntimeError(
                                    "start_self_improvement_cycle is not callable"
                                )
                            logger.info(
                                "🛰️ start_self_improvement_cycle() callable confirmed; executing with detailed context",
                                extra=log_record(
                                    event="meta-planning-callable-confirmed",
                                    workflow_count=len(workflows),
                                    planner_cls=str(planner_cls),
                                    interval_seconds=interval,
                                    workflow_graph_snapshot=repr(workflow_graph_obj),
                                    event_bus_snapshot=repr(shared_event_bus),
                                    caller_module=__name__,
                                ),
                            )
                            thread = meta_planning.start_self_improvement_cycle(
                                workflows,
                                event_bus=shared_event_bus,
                                interval=interval,
                                workflow_graph=workflow_graph_obj,
                            )
                            logger.info(
                                "🛰️ start_self_improvement_cycle() invocation completed; capturing return object",
                                extra=log_record(
                                    event="meta-planning-post-invoke",
                                    returned_type=type(thread).__name__ if thread is not None else None,
                                    returned_is_thread=isinstance(thread, threading.Thread),
                                    returned_repr=repr(thread),
                                ),
                            )
                            logger.info(
                                "🛰️ meta planning bootstrap call returned from start_self_improvement_cycle()",
                                extra=log_record(
                                    event="meta-planning-bootstrap-returned",
                                    thread_is_none=thread is None,
                                    thread_type=type(thread).__name__ if thread is not None else None,
                                    thread_dir=list(sorted(set(dir(thread)) if thread is not None else [])),
                                    workflow_count=len(workflows),
                                    interval_seconds=interval,
                                    thread_target=getattr(thread, "_target", None),
                                    thread_args=getattr(thread, "_args", None),
                                    thread_kwargs=getattr(thread, "_kwargs", None),
                                ),
                            )
                            if thread is None:
                                logger.error(
                                    "❌ meta planning bootstrap returned None thread",
                                    extra=log_record(
                                        event="meta-planning-thread-none",
                                        planner_cls=str(planner_cls),
                                        workflow_ids=list(workflows.keys()),
                                    ),
                                )
                                raise RuntimeError("meta planning bootstrap returned None")
                            logger.info(
                                "🛰️ meta planning bootstrap returned valid thread object; proceeding to post-call checks",
                                extra=log_record(
                                    event="meta-planning-bootstrap-post-call",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    alive=getattr(thread, "is_alive", lambda: False)(),
                                    planner_cls=str(planner_cls),
                                    workflow_ids=list(workflows.keys()),
                                    event_bus_type=type(shared_event_bus).__name__
                                    if shared_event_bus is not None
                                    else None,
                                ),
                            )
                            logger.info(
                                "✅ meta planning bootstrap call returned",
                                extra=log_record(
                                    event="meta-planning-bootstrap-return",
                                    thread_repr=repr(thread),
                                    thread_name=getattr(thread, "name", "unknown"),
                                    thread_ident=getattr(thread, "ident", None),
                                ),
                            )
                            logger.info(
                                "✅ meta planning thread object created",
                                extra=log_record(
                                    event="meta-planning-thread-created",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    planner_cls=str(planner_cls),
                                    workflow_count=len(workflows),
                                    target=getattr(thread, "_target", None),
                                    native_id=getattr(thread, "native_id", None),
                                    ident=getattr(thread, "ident", None),
                                ),
                            )
                            logger.info(
                                "✅ meta planning thread attributes captured",
                                extra=log_record(
                                    event="meta-planning-thread-attrs",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    alive=getattr(thread, "is_alive", lambda: False)(),
                                    thread_ident=getattr(thread, "ident", None),
                                    native_id=getattr(thread, "native_id", None),
                                ),
                            )
                            logger.info(
                                "✅ meta planning bootstrap pipeline completed; preparing start() call",
                                extra=log_record(
                                    event="meta-planning-bootstrap-finished",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    planner_cls=str(planner_cls),
                                    alive_pre_start=getattr(thread, "is_alive", lambda: False)(),
                                ),
                            )
                        except Exception as exc:
                            logger.exception(
                                "❌ meta planning loop bootstrap failed; thread object missing",
                                extra=log_record(
                                    event="meta-loop-error",
                                    workflow_count=len(workflows),
                                    planner_cls=str(planner_cls),
                                    interval_seconds=interval,
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                ),
                            )
                            logger.exception(
                                "❌ meta planning bootstrap returned invalid thread",
                                extra=log_record(
                                    event="meta-planning-thread-invalid",
                                    planner_cls=str(planner_cls),
                                    workflow_count=len(workflows),
                                    error_type=type(exc).__name__,
                                ),
                            )
                            logger.exception(
                                "failed to initialize meta planning loop; sandbox launch halted",
                                extra=log_record(event="meta-loop-error"),
                            )
                            sys.exit(1)
    
                        try:
                            logger.info(
                                "🧭 entering meta planning thread.start() block",  # explicit boundary marker
                                extra=log_record(
                                    event="meta-planning-thread-start-boundary",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    alive_pre=getattr(thread, "is_alive", lambda: False)(),
                                    thread_ident=getattr(thread, "ident", None),
                                    thread_native_id=getattr(thread, "native_id", None),
                                    thread_target=getattr(thread, "_target", None),
                                ),
                            )
                            logger.info(
                                "🔧 attempting to start meta planning loop thread",
                                extra=log_record(
                                    event="meta-planning-start-attempt",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    planner_cls=str(planner_cls),
                                    workflow_count=len(workflows),
                                ),
                            )
                            thread.start()
                            logger.info(
                                "✅ thread.start() invoked successfully for meta planning loop",
                                extra=log_record(
                                    event="meta-planning-thread-start-invoked",
                                    thread_name=getattr(thread, "name", "unknown"),
                                ),
                            )
                            logger.info(
                                "✅ meta planning thread start invoked",
                                extra=log_record(
                                    event="meta-planning-start-invoke",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    thread_ident=getattr(thread, "ident", None),
                                    native_id=getattr(thread, "native_id", None),
                                ),
                            )
                            logger.info(
                                "✅ meta planning loop thread started successfully",
                                extra=log_record(
                                    event="meta-planning-start",
                            thread_name=getattr(thread, "name", "unknown"),
                            is_alive=getattr(thread, "is_alive", lambda: False)(),
                            planner_cls=str(planner_cls),
                            workflow_count=len(workflows),
                            workflow_graph_present=workflow_graph_obj is not None,
                                    native_id=getattr(thread, "native_id", None),
                                ),
                            )
                            logger.info(
                                "🛰️ meta planning loop thread start diagnostics captured",
                                extra=log_record(
                                    event="meta-planning-thread-start-diagnostics",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    thread_ident=getattr(thread, "ident", None),
                                    native_id=getattr(thread, "native_id", None),
                                    daemon=getattr(thread, "daemon", None),
                                    alive=getattr(thread, "is_alive", lambda: False)(),
                                    target=getattr(thread, "_target", None),
                                    args=getattr(thread, "_args", None),
                                    kwargs=getattr(thread, "_kwargs", None),
                                ),
                            )
                            logger.info(
                                "✅ meta planning thread alive status confirmed",
                                extra=log_record(
                                    event="meta-planning-thread-alive",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    is_alive=getattr(thread, "is_alive", lambda: False)(),
                                ),
                            )
                            alive_state = getattr(thread, "is_alive", lambda: False)()
                            if not alive_state:
                                logger.error(
                                    "❌ meta planning loop thread reported not alive after start",
                                    extra=log_record(
                                        event="meta-planning-thread-not-alive",
                                        thread_name=getattr(thread, "name", "unknown"),
                                        planner_cls=str(planner_cls),
                                        workflow_count=len(workflows),
                                    ),
                                )
                                raise RuntimeError("meta planning loop thread failed to stay alive")
                            logger.info(
                                "🟢 meta planning loop thread running",
                                extra=log_record(
                                    event="meta-planning-thread-running",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    native_id=getattr(thread, "native_id", None),
                                    ident=getattr(thread, "ident", None),
                                ),
                            )
                            logger.info(
                                "✅ meta planning loop thread alive verification passed",
                                extra=log_record(
                                    event="meta-planning-thread-alive-verified",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    planner_cls=str(planner_cls),
                                    workflow_count=len(workflows),
                                ),
                            )
                            logger.info(
                                "✅ meta planning loop start confirmed; orchestrator warm-up next",
                                extra=log_record(
                                    event="meta-planning-start-confirm",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    daemon=getattr(thread, "daemon", None),
                                    is_alive=getattr(thread, "is_alive", lambda: False)(),
                                ),
                            )
                            logger.info(
                                "🎯 meta planning start block completed without exceptions; handing off to orchestrator",
                                extra=log_record(
                                    event="meta-planning-start-block-complete",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    alive=getattr(thread, "is_alive", lambda: False)(),
                                    planner_cls=str(planner_cls),
                                    workflow_ids=list(workflows.keys()),
                                ),
                            )
                        except Exception:
                            logger.exception(
                                "❌ meta planning loop thread failed to start",  # emoji for quick scanning
                                extra=log_record(event="meta-planning-start-error"),
                            )
                            sys.exit(1)
                    else:
                        logger.error(
                            "❌ gating red: prelaunch ROI or planner checks failed; aborting meta-planning start",
                            extra=log_record(
                                event="meta-planning-gate-red-summary",
                                ready_to_launch=ready_to_launch,
                                roi_backoff=roi_backoff_triggered,
                                planner_available=planner_cls is not None,
                                workflows_present=bool(workflows),
                                readiness_error=readiness_error,
                            ),
                        )
                        logger.error(
                            "❌ meta planning loop launch gate failed",  # emoji for quick scanning
                            extra=log_record(
                                event="meta-planning-gate-failed",
                                roi_backoff=roi_backoff_triggered,
                                ready_to_launch=ready_to_launch,
                                planner_available=planner_cls is not None,
                                workflow_count=len(workflows),
                                readiness_error=readiness_error,
                            ),
                        )
                        logger.error(
                            "❌ readiness gate failed; meta planning launch conditions not met",
                            extra=log_record(
                                event="meta-planning-readiness-failure",
                                roi_backoff=roi_backoff_triggered,
                                ready_to_launch=ready_to_launch,
                                planner_available=planner_cls is not None,
                                workflow_count=len(workflows),
                                workflow_ids=list(workflows.keys()),
                            ),
                        )
                        failure_reason = readiness_error or (
                            "workflows did not reach ROI stagnation; sandbox launch aborted"
                        )
                        logger.error(
                            "❌ launch gate red; blocking meta planning loop",  # emoji for quick scanning
                            extra=log_record(
                                event="meta-planning-gate-red",
                                planner_available=planner_cls is not None,
                                workflow_count=len(workflows),
                                roi_backoff=roi_backoff_triggered,
                                ready_to_launch=ready_to_launch,
                                readiness_error=readiness_error,
                            ),
                        )
                        logger.error(
                            "❌ meta planning launch vetoed after readiness evaluation",
                            extra=log_record(
                                event="meta-planning-veto",
                                reason=failure_reason,
                                planner_available=planner_cls is not None,
                                workflow_count=len(workflows),
                                roi_backoff=roi_backoff_triggered,
                            ),
                        )
                        logger.error(
                            "❌ readiness gate blocked meta planning loop",
                            extra=log_record(
                                event="meta-planning-ready-false",
                                reason=failure_reason,
                                roi_backoff=roi_backoff_triggered,
                                ready_to_launch=ready_to_launch,
                            ),
                        )
                        logger.error(
                            "meta planning loop not started: %s",
                            failure_reason,
                            extra=log_record(
                                event="meta-planning-skipped",
                                reason=failure_reason,
                                workflow_count=len(workflows),
                                planner_available=planner_cls is not None,
                            ),
                        )
                        sys.exit(1)
                    logger.info(
                        "preseeded bootstrap context in use; pipeline and manager are cached",
                        extra=log_record(event="bootstrap-preseed"),
                    )
    
                    if ready_to_launch:
                        try:
                            orchestrator = SandboxOrchestrator(
                                workflows,
                                logger=logger,
                                loop_interval=float(os.getenv("GLOBAL_ORCHESTRATOR_INTERVAL", "30")),
                                diminishing_threshold=float(
                                    os.getenv("GLOBAL_ROI_DIMINISHING_THRESHOLD", "0.01")
                                ),
                                patience=int(os.getenv("GLOBAL_ROI_PATIENCE", "3")),
                            )
                            logger.info(
                                "✅ sandbox orchestrator object created",  # emoji for quick scanning
                                extra=log_record(
                                    event="orchestrator-created",
                                    workflow_count=len(workflows),
                                    loop_interval=os.getenv("GLOBAL_ORCHESTRATOR_INTERVAL", "30"),
                                    diminishing_threshold=os.getenv("GLOBAL_ROI_DIMINISHING_THRESHOLD", "0.01"),
                                    patience=os.getenv("GLOBAL_ROI_PATIENCE", "3"),
                                ),
                            )
                        except Exception:
                            logger.exception(
                                "❌ failed to build sandbox orchestrator",  # emoji for quick scanning
                                extra=log_record(event="orchestrator-build-error"),
                            )
                            sys.exit(1)
    
                        try:
                            orchestrator_thread = threading.Thread(
                                target=orchestrator.run,
                                name="sandbox-orchestrator",
                                daemon=True,
                            )
                            orchestrator_thread.start()
                            logger.info(
                                "✅ sandbox orchestrator started",  # emoji for quick scanning
                                extra=log_record(
                                    event="orchestrator-start",
                                    thread_name="sandbox-orchestrator",
                                    is_alive=orchestrator_thread.is_alive(),
                                    workflow_count=len(workflows),
                                ),
                            )
                        except Exception:
                            logger.exception(
                                "❌ failed to start sandbox orchestrator thread",  # emoji for quick scanning
                                extra=log_record(event="orchestrator-start-error"),
                            )
                            sys.exit(1)
                except Exception:  # pragma: no cover - defensive bootstrap hint
                    logger.exception(
                        "failed to preseed bootstrap context before bot loading",
                        extra=log_record(event="bootstrap-preseed-error"),
                    )

            except Exception as exc:
                logger.exception(
                    "Early startup failure before meta-trace instrumentation",
                    extra=log_record(
                        event="startup-pre-meta-trace-failure",
                        last_step=last_pre_meta_trace_step,
                        error_type=type(exc).__name__,
                    ),
                )
                print(
                    "[DEBUG] startup failed before META-TRACE logging; last_step=%s; error=%s"
                    % (last_pre_meta_trace_step, exc),
                    flush=True,
                )
                raise

        if args.health_check:
            bootstrap_environment(
                initialize=False,
                enforce_dependencies=False,
            )
            try:
                health_snapshot = sandbox_health()
            except Exception as exc:  # pragma: no cover - defensive fallback
                sandbox_crashes_total.inc()
                sandbox_last_failure_ts.set(time.time())
                logger.exception(
                    "Sandbox health probe failed", extra=log_record(event="health-error")
                )
                failures = [f"health probe failed: {exc}"]
                _emit_health_report(
                    {"error": str(exc)},
                    healthy=False,
                    failures=failures,
                )
                shutdown_autonomous_sandbox()
                logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
                sys.exit(2)

            logger.info(
                "Sandbox health", extra=log_record(health=health_snapshot)
            )
            healthy, failures = _evaluate_health(
                health_snapshot,
                dependency_mode=_resolve_dependency_mode(settings),
            )
            _emit_health_report(
                health_snapshot,
                healthy=healthy,
                failures=failures,
            )
            shutdown_autonomous_sandbox()
            logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
            if not healthy:
                logger.error(
                    "Sandbox health check failed: %s",
                    "; ".join(failures) if failures else "unknown reason",
                )
                sys.exit(2)
            return
        if roi_backoff_triggered or not ready_to_launch:
            if roi_backoff_triggered and "ROI backoff triggered before launch" not in failure_reasons:
                failure_reasons.append("ROI backoff triggered before launch")
            summary = "; ".join(failure_reasons) if failure_reasons else "launch conditions not met"
            logger.error(
                "sandbox launch blocked: %s",
                summary,
                extra=log_record(
                    event="sandbox-launch-blocked",
                    failure_reasons=failure_reasons,
                    roi_backoff=roi_backoff_triggered,
                    ready_to_launch=ready_to_launch,
                    correlation_id=cid,
                ),
            )
            sys.stderr.write(f"sandbox launch blocked: {summary}\n")
            sys.stderr.flush()
            sys.exit(3)
        _emoji_step(
            logger,
            "🧠",
            "Starting sandbox orchestrator and launch thread",
            event="step-start-sandbox",
            roi_backoff=roi_backoff_triggered,
            ready=ready_to_launch,
        )
        print("[start_autonomous_sandbox] launching sandbox", flush=True)
        launch_sandbox()
        print("[start_autonomous_sandbox] sandbox exited", flush=True)
        logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
    except Exception:  # pragma: no cover - defensive catch
        sandbox_crashes_total.inc()
        sandbox_last_failure_ts.set(time.time())
        logger.exception("Failed to launch sandbox", extra=log_record(event="failure"))
        sys.exit(1)
    finally:
        try:
            broadcast_timeout_floors(
                source="start_autonomous_sandbox",
                timeout_floors=load_escalated_timeout_floors(),
                component_floors=load_component_timeout_floors(),
                guard_context=get_bootstrap_guard_context(),
            )
        except Exception:
            logger.debug(
                "failed to broadcast timeout floors", exc_info=True
            )
        if bootstrap_lock_guard is not None:
            try:
                bootstrap_lock_guard.__exit__(None, None, None)
            except Exception:
                logger.exception(
                    "failed to release bootstrap lock",
                    extra=log_record(
                        event="bootstrap-lock-release-error",
                        lock_path=str(BOOTSTRAP_LOCK_PATH),
                    ),
                )
        elif bootstrap_lock is not None and getattr(bootstrap_lock, "is_locked", False):
            try:
                bootstrap_lock.release()
            except Exception:
                logger.exception(
                    "failed to release bootstrap lock via fallback",
                    extra=log_record(
                        event="bootstrap-lock-release-fallback",
                        lock_path=str(BOOTSTRAP_LOCK_PATH),
                    ),
                )
        try:
            shutdown_autonomous_sandbox()
        except Exception:
            logger.exception(
                "sandbox shutdown failed", extra=log_record(event="shutdown-error")
            )
        set_correlation_id(None)


if __name__ == "__main__":
    main()
