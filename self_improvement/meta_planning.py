"""Meta-planning routines for the self-improvement package.

The :func:`self_improvement_cycle` coroutine drives background workflow
optimization using the optional :class:`MetaWorkflowPlanner` component.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Sequence

from importlib import import_module

from statistics import fmean
import asyncio
import json
import logging
import os
import sys
import traceback
import time
import threading
import queue
from pathlib import Path
from contextlib import contextmanager, nullcontext

if __package__ in {None, ""}:  # pragma: no cover - script execution compatibility
    _here = Path(__file__).resolve()
    for _candidate in (_here.parent, *_here.parents):
        marker = _candidate / "import_compat.py"
        if marker.exists() or (_candidate / "pyproject.toml").exists():
            candidate_str = str(_candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

try:  # pragma: no cover - optional dependency location
    from menace_sandbox import dynamic_path_router as _dynamic_path_router
except Exception:  # pragma: no cover
    import dynamic_path_router as _dynamic_path_router  # type: ignore
resolve_path = _dynamic_path_router.resolve_path  # type: ignore
sys.modules.setdefault("menace.dynamic_path_router", _dynamic_path_router)

if "menace.logging_utils" in sys.modules and not hasattr(
    sys.modules["menace.logging_utils"], "setup_logging"
):  # pragma: no cover - test stub fallback
    sys.modules["menace.logging_utils"].setup_logging = lambda *a, **k: None

try:  # pragma: no cover - allow flat package layout
    from menace_sandbox.logging_utils import get_logger, log_record
except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
    from logging_utils import get_logger, log_record  # type: ignore

try:  # pragma: no cover - allow flat package layout
    from menace_sandbox.sandbox_settings import SandboxSettings, DEFAULT_SEVERITY_SCORE_MAP
except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
    from sandbox_settings import (  # type: ignore
        SandboxSettings,
        DEFAULT_SEVERITY_SCORE_MAP,
    )

from . import init as _init

try:  # pragma: no cover - allow flat package layout
    from menace_sandbox.workflow_stability_db import WorkflowStabilityDB
except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
    from workflow_stability_db import WorkflowStabilityDB  # type: ignore

try:  # pragma: no cover - allow flat package layout
    from menace_sandbox.roi_results_db import ROIResultsDB
except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
    from roi_results_db import ROIResultsDB  # type: ignore

try:  # pragma: no cover - allow flat package layout
    from menace_sandbox.lock_utils import SandboxLock, Timeout, LOCK_TIMEOUT
except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
    from lock_utils import SandboxLock, Timeout, LOCK_TIMEOUT  # type: ignore

try:  # pragma: no cover - allow flat package layout
    from menace_sandbox.workflow_evolution_manager import WorkflowCycleController
except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
    try:
        from workflow_evolution_manager import WorkflowCycleController  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        WorkflowCycleController = None  # type: ignore

from .baseline_tracker import BaselineTracker, TRACKER as BASELINE_TRACKER
from .orphan_handling import (
    integrate_orphans,
    integrate_orphans_sync,
    post_round_orphan_scan,
    post_round_orphan_scan_sync,
)

if TYPE_CHECKING:  # pragma: no cover - used for static analysis only
    try:
        from menace_sandbox.error_logger import TelemetryEvent  # type: ignore
    except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
        from error_logger import TelemetryEvent  # type: ignore
else:
    TelemetryEvent = Any  # type: ignore[assignment]

from .sandbox_score import get_latest_sandbox_score
from context_builder_util import create_context_builder


_TelemetryEventCls: Any | None = None


def get_telemetry_event():
    """Return the :class:`TelemetryEvent` class via a lazy import."""

    global _TelemetryEventCls

    if _TelemetryEventCls is None:
        try:  # pragma: no cover - allow flat package layout
            from menace_sandbox.error_logger import TelemetryEvent as _ResolvedTelemetryEvent
        except (ImportError, ValueError):  # pragma: no cover - fallback for manual_bootstrap environments
            from error_logger import TelemetryEvent as _ResolvedTelemetryEvent  # type: ignore

        _TelemetryEventCls = _ResolvedTelemetryEvent

    return _TelemetryEventCls


_cycle_thread: Any | None = None
_stop_event: threading.Event | None = None
_cycle_watchdog_stop: threading.Event | None = None


class _CycleTickState:
    """Thread-safe tracking of cycle ticks for watchdog monitoring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.last_tick: float | None = None
        self.tick_count = 0
        self.last_tick_thread_name: str | None = None
        self.last_tick_thread_ident: int | None = None

    def tick(self) -> dict[str, Any]:
        now = time.time()
        current_thread = threading.current_thread()
        with self._lock:
            self.last_tick = now
            self.tick_count += 1
            self.last_tick_thread_name = current_thread.name
            self.last_tick_thread_ident = current_thread.ident
            return {
                "last_tick": self.last_tick,
                "tick_count": self.tick_count,
                "last_tick_thread_name": self.last_tick_thread_name,
                "last_tick_thread_ident": self.last_tick_thread_ident,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "last_tick": self.last_tick,
                "tick_count": self.tick_count,
                "last_tick_thread_name": self.last_tick_thread_name,
                "last_tick_thread_ident": self.last_tick_thread_ident,
            }

    def reset(self) -> None:
        with self._lock:
            self.started_at = time.time()
            self.last_tick = None
            self.tick_count = 0
            self.last_tick_thread_name = None
            self.last_tick_thread_ident = None


class _LoopHeartbeatState:
    """Thread-safe tracking for event loop heartbeats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.last_heartbeat: float | None = None
        self.heartbeat_count = 0
        self.last_heartbeat_thread_name: str | None = None
        self.last_heartbeat_thread_ident: int | None = None

    def beat(self) -> dict[str, Any]:
        now = time.time()
        current_thread = threading.current_thread()
        with self._lock:
            self.last_heartbeat = now
            self.heartbeat_count += 1
            self.last_heartbeat_thread_name = current_thread.name
            self.last_heartbeat_thread_ident = current_thread.ident
            return {
                "last_heartbeat": self.last_heartbeat,
                "heartbeat_count": self.heartbeat_count,
                "last_heartbeat_thread_name": self.last_heartbeat_thread_name,
                "last_heartbeat_thread_ident": self.last_heartbeat_thread_ident,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "last_heartbeat": self.last_heartbeat,
                "heartbeat_count": self.heartbeat_count,
                "last_heartbeat_thread_name": self.last_heartbeat_thread_name,
                "last_heartbeat_thread_ident": self.last_heartbeat_thread_ident,
            }

    def reset(self) -> None:
        with self._lock:
            self.started_at = time.time()
            self.last_heartbeat = None
            self.heartbeat_count = 0
            self.last_heartbeat_thread_name = None
            self.last_heartbeat_thread_ident = None


class _LoopPingState:
    """Thread-safe tracking for event loop ping health."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.last_ping: float | None = None
        self.ping_count = 0
        self.last_ping_thread_name: str | None = None
        self.last_ping_thread_ident: int | None = None

    def ping(self) -> dict[str, Any]:
        now = time.time()
        current_thread = threading.current_thread()
        with self._lock:
            self.last_ping = now
            self.ping_count += 1
            self.last_ping_thread_name = current_thread.name
            self.last_ping_thread_ident = current_thread.ident
            return {
                "last_ping": self.last_ping,
                "ping_count": self.ping_count,
                "last_ping_thread_name": self.last_ping_thread_name,
                "last_ping_thread_ident": self.last_ping_thread_ident,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "last_ping": self.last_ping,
                "ping_count": self.ping_count,
                "last_ping_thread_name": self.last_ping_thread_name,
                "last_ping_thread_ident": self.last_ping_thread_ident,
            }

    def reset(self) -> None:
        with self._lock:
            self.started_at = time.time()
            self.last_ping = None
            self.ping_count = 0
            self.last_ping_thread_name = None
            self.last_ping_thread_ident = None


class _WorkflowROICycleController:
    """Track ROI deltas for a workflow across repeated iterations.

    Controllers keep running until the observed ROI delta drops below the
    configured ``threshold`` after at least ``patience`` iterations.  This
    mirrors the ROI backoff configuration so the controller behaviour aligns
    with the rest of the meta-planning pipeline.
    """

    __slots__ = (
        "workflow_id",
        "threshold",
        "patience",
        "iteration",
        "last_roi",
        "last_delta",
        "status",
        "halted_at",
        "updated_at",
    )

    def __init__(self, workflow_id: str, threshold: float, patience: int) -> None:
        self.workflow_id = workflow_id
        self.threshold = float(threshold)
        self.patience = max(1, int(patience))
        self.iteration = 0
        self.last_roi = 0.0
        self.last_delta = 0.0
        self.status = "pending"
        self.halted_at: float | None = None
        self.updated_at: float | None = None

    # ------------------------------------------------------------------
    def matches(self, threshold: float, patience: int) -> bool:
        """Return ``True`` when controller configuration matches inputs."""

        return self.threshold == float(threshold) and self.patience == max(1, int(patience))

    # ------------------------------------------------------------------
    def observe(self, roi_gain: float, roi_delta: float | None) -> bool:
        """Record an iteration and return ``True`` if improvements continue."""

        self.iteration += 1
        self.last_roi = float(roi_gain)
        self.last_delta = float(roi_delta if roi_delta is not None else roi_gain)
        self.updated_at = time.time()

        if self.threshold <= 0:
            self.status = "running"
            return True

        if self.iteration < self.patience:
            self.status = "running"
            return True

        if self.last_delta > self.threshold:
            self.status = "running"
            return True

        self.status = "halted"
        if self.halted_at is None:
            self.halted_at = self.updated_at
        return False

    # ------------------------------------------------------------------
    def snapshot(self) -> Mapping[str, Any]:
        """Return a serialisable snapshot of the controller state."""

        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "iterations": self.iteration,
            "last_roi": self.last_roi,
            "last_delta": self.last_delta,
            "threshold": self.threshold,
            "patience": self.patience,
            "halted_at": self.halted_at,
            "updated_at": self.updated_at,
        }


_WORKFLOW_CONTROLLERS: dict[str, _WorkflowROICycleController] = {}
_WORKFLOW_CONTROLLER_LOCK = threading.Lock()


def _controller_key(workflow_id: str | int) -> str:
    """Normalise *workflow_id* to a string key."""

    return str(workflow_id)


def _get_or_create_controller(
    workflow_id: str, *, threshold: float, patience: int
) -> _WorkflowROICycleController:
    """Return a workflow controller configured for *workflow_id*."""

    controller = _WORKFLOW_CONTROLLERS.get(workflow_id)
    if controller is None or not controller.matches(threshold, patience):
        controller = _WorkflowROICycleController(workflow_id, threshold, patience)
        _WORKFLOW_CONTROLLERS[workflow_id] = controller
    return controller


def record_workflow_iteration(
    workflow_id: str | int,
    *,
    roi_gain: float,
    roi_delta: float | None,
    threshold: float | None = None,
    patience: int | None = None,
) -> Mapping[str, Any]:
    """Record an iteration for *workflow_id* and return controller status."""

    effective_threshold = float(
        ROI_BACKOFF_THRESHOLD if threshold is None else threshold
    )
    effective_patience = max(1, int(patience if patience is not None else ROI_BACKOFF_CONSECUTIVE))
    key = _controller_key(workflow_id)
    with _WORKFLOW_CONTROLLER_LOCK:
        controller = _get_or_create_controller(
            key, threshold=effective_threshold, patience=effective_patience
        )
        controller.observe(float(roi_gain), roi_delta)
        return dict(controller.snapshot())


def workflow_controller_status() -> Mapping[str, Mapping[str, Any]]:
    """Return a snapshot of all workflow controller states."""

    with _WORKFLOW_CONTROLLER_LOCK:
        return {key: dict(ctrl.snapshot()) for key, ctrl in _WORKFLOW_CONTROLLERS.items()}


if __package__:  # pragma: no cover - support package imports
    try:  # pragma: no cover - optional dependency
        from menace_sandbox.unified_event_bus import UnifiedEventBus
    except (ImportError, ValueError):  # pragma: no cover - fallback when event bus missing
        try:
            from unified_event_bus import UnifiedEventBus  # type: ignore
        except Exception as exc2:  # pragma: no cover - gracefully degrade when unavailable
            get_logger(__name__).warning(
                "unified event bus unavailable",  # noqa: TRY300
                extra=log_record(component=__name__, dependency="unified_event_bus"),
                exc_info=exc2,
            )
            UnifiedEventBus = None  # type: ignore
else:  # pragma: no cover - script execution fallback
    try:
        from unified_event_bus import UnifiedEventBus  # type: ignore
    except Exception as exc:  # pragma: no cover - gracefully degrade when unavailable
        get_logger(__name__).warning(
            "unified event bus unavailable",  # noqa: TRY300
            extra=log_record(component=__name__, dependency="unified_event_bus"),
            exc_info=exc,
        )
        UnifiedEventBus = None  # type: ignore


def _load_meta_workflow_planner() -> Any | None:
    """Import :class:`MetaWorkflowPlanner` handling flat/packaged layouts."""

    logger = get_logger(__name__)

    logger.info(
        "meta workflow planner load requested", extra=log_record(component=__name__)
    )
    print("[META-TRACE] initiating meta_workflow_planner import resolution", flush=True)

    candidate_modules: list[str] = []

    if __package__:
        package_parts = __package__.split(".")
        for idx in range(len(package_parts), -1, -1):
            parent = ".".join(package_parts[:idx])
            candidate_modules.append(
                f"{parent + '.' if parent else ''}meta_workflow_planner"
            )
    else:
        candidate_modules.append("meta_workflow_planner")

    for module_name in candidate_modules:
        logger.info(
            "attempting meta_workflow_planner import",
            extra=log_record(
                component=__name__, dependency="meta_workflow_planner", module=module_name
            ),
        )
        print(
            f"[META-TRACE] trying meta_workflow_planner import from {module_name}",
            flush=True,
        )
        try:  # pragma: no cover - optional dependency
            module = import_module(module_name)
            logger.info(
                "meta_workflow_planner import succeeded",
                extra=log_record(
                    component=__name__, dependency="meta_workflow_planner", module=module_name
                ),
            )
            print(
                f"[META-TRACE] meta_workflow_planner loaded from {module_name}",
                flush=True,
            )
            return getattr(module, "MetaWorkflowPlanner")
        except Exception as exc:  # pragma: no cover - gracefully degrade
            logger.warning(
                "meta_workflow_planner import failed",  # noqa: TRY300
                extra=log_record(
                    component=__name__, dependency="meta_workflow_planner", module=module_name
                ),
                exc_info=exc,
            )
            print(
                f"[META-TRACE] import failed for {module_name}: {exc}", flush=True
            )

    return None


MetaWorkflowPlanner: Any | None = None
_META_PLANNER_RESOLVED = False


def resolve_meta_workflow_planner(force_reload: bool = False) -> Any | None:
    """Return the optional :class:`MetaWorkflowPlanner` implementation.

    The resolver defers importing :mod:`meta_workflow_planner` until the
    planner is explicitly required.  When ``force_reload`` is ``True`` the
    cached result is discarded and the import is attempted again.
    """

    global MetaWorkflowPlanner, _META_PLANNER_RESOLVED

    logger = get_logger(__name__)
    logger.info(
        "meta workflow planner resolution invoked",
        extra=log_record(component=__name__, force_reload=force_reload),
    )
    print(
        f"[META-TRACE] resolve_meta_workflow_planner called (force_reload={force_reload})",
        flush=True,
    )

    if force_reload:
        MetaWorkflowPlanner = None
        _META_PLANNER_RESOLVED = False
        logger.info(
            "meta workflow planner cache cleared",
            extra=log_record(component=__name__, action="force_reload"),
        )
        print("[META-TRACE] cleared cached meta planner", flush=True)

    if _META_PLANNER_RESOLVED:
        logger.info(
            "meta workflow planner already resolved",
            extra=log_record(
                component=__name__,
                planner_cached=MetaWorkflowPlanner is not None,
                planner_class=getattr(MetaWorkflowPlanner, "__name__", str(MetaWorkflowPlanner)),
            ),
        )
        print(
            f"[META-TRACE] returning cached meta planner {MetaWorkflowPlanner}",
            flush=True,
        )
        return MetaWorkflowPlanner

    MetaWorkflowPlanner = _load_meta_workflow_planner()
    _META_PLANNER_RESOLVED = True
    logger.info(
        "meta workflow planner resolution complete",
        extra=log_record(
            component=__name__,
            planner_found=MetaWorkflowPlanner is not None,
            planner_class=getattr(MetaWorkflowPlanner, "__name__", str(MetaWorkflowPlanner)),
        ),
    )
    print(
        f"[META-TRACE] meta planner resolved to {MetaWorkflowPlanner}", flush=True
    )
    return MetaWorkflowPlanner


class _FallbackPlanner:
    """Lightweight planner with basic mutation and persistence.

    Concurrency:
        Access to the persistent state file is guarded by a
        :class:`SandboxLock`, ensuring that multiple planner instances do not
        corrupt state.  Writes are performed atomically by first writing to a
        temporary file and then replacing the target.
    """

    def __init__(self) -> None:
        cfg = _init.settings
        roi_window = int(getattr(cfg, "roi_window", getattr(cfg, "roi_cycles", 5) or 5))
        try:
            self.roi_db: ROIResultsDB | None = ROIResultsDB(window=roi_window)
        except (OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            get_logger(__name__).warning(
                "ROIResultsDB unavailable",
                extra=log_record(component=__name__),
                exc_info=exc,
            )
            self.roi_db = None
        try:
            self.stability_db: WorkflowStabilityDB | None = WorkflowStabilityDB()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            get_logger(__name__).warning(
                "WorkflowStabilityDB unavailable",
                extra=log_record(component=__name__),
                exc_info=exc,
            )
            self.stability_db = None

        self.logger = get_logger("FallbackPlanner")
        data_dir = Path(resolve_path(SandboxSettings().sandbox_data_dir))
        self.state_path = Path(resolve_path(data_dir / "fallback_planner.json"))
        self.state_lock = SandboxLock(
            str(self.state_path.with_suffix(self.state_path.suffix + ".lock"))
        )
        self.cluster_map: dict[tuple[str, ...], dict[str, Any]] = {}
        self.state_capacity = getattr(cfg, "meta_state_capacity", 1000)
        self.mutation_rate = getattr(cfg, "mutation_rate", getattr(cfg, "meta_mutation_rate", 1.0))
        self.roi_weight = getattr(cfg, "roi_weight", getattr(cfg, "meta_roi_weight", 1.0))
        self.domain_transition_penalty = getattr(
            cfg, "domain_transition_penalty", getattr(cfg, "meta_domain_penalty", 1.0)
        )
        self.entropy_weight = getattr(cfg, "meta_entropy_weight", 0.0)
        self.stability_weight = getattr(
            cfg, "stability_weight", getattr(cfg, "meta_stability_weight", 1.0)
        )
        self.entropy_threshold = 0.0
        self.state_prune_strategy = getattr(
            cfg, "state_prune_strategy", getattr(cfg, "meta_state_prune_strategy", "recent")
        )
        self.roi_window = roi_window
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self, *, lock: bool = True) -> None:
        """Load planner state from disk with optional locking."""

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            ctx = (
                self.state_lock.acquire(timeout=LOCK_TIMEOUT)
                if lock
                else nullcontext()
            )
            with ctx:
                if self.state_path.exists():
                    data = json.loads(self.state_path.read_text())
                else:
                    data = {}
            clusters = data.get("clusters", data)
            self.cluster_map = {tuple(k.split("|")): v for k, v in clusters.items()}
            baseline_state = data.get("baseline", {})
            if isinstance(baseline_state, Mapping):
                BASELINE_TRACKER.load_state(baseline_state)
        except Timeout as exc:  # pragma: no cover - lock contention
            self.logger.warning(
                "state lock acquisition timed out",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
            self.cluster_map = {}
        except (OSError, json.JSONDecodeError) as exc:
            get_logger(__name__).debug(
                "failed to load fallback planner state",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
            self.cluster_map = {}

    def _save_state(self, *, lock: bool = True) -> None:
        """Persist planner state to disk atomically.

        A file lock guards against concurrent writers.  Data is first written to
        a temporary file and then moved into place to avoid partial saves.
        """

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "clusters": {"|".join(k): v for k, v in self.cluster_map.items()},
                "baseline": BASELINE_TRACKER.to_state(),
            }
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            ctx = (
                self.state_lock.acquire(timeout=LOCK_TIMEOUT)
                if lock
                else nullcontext()
            )
            with ctx:
                tmp_path.write_text(json.dumps(data, indent=2))
                os.replace(tmp_path, self.state_path)
        except Timeout as exc:  # pragma: no cover - lock contention
            self.logger.warning(
                "state lock acquisition timed out",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
        except OSError as exc:  # pragma: no cover - best effort
            self.logger.debug(
                "failed to persist fallback planner state",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )

    # ------------------------------------------------------------------
    @contextmanager
    def _state_update(self) -> Any:
        """Context manager providing atomic load/modify/save of state."""

        try:
            with self.state_lock.acquire(timeout=LOCK_TIMEOUT):
                self._load_state(lock=False)
                yield
                self._save_state(lock=False)
        except Timeout as exc:  # pragma: no cover - lock contention
            self.logger.warning(
                "state lock acquisition timed out",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
            self.cluster_map = {}
            yield

    # ------------------------------------------------------------------
    def _prune_state(self) -> None:
        """Trim ``cluster_map`` according to pruning strategy."""

        if self.state_capacity <= 0:
            self.logger.debug(
                "state capacity disabled; skipping prune",
                extra=log_record(component=__name__),
            )
            return
        if len(self.cluster_map) <= self.state_capacity:
            self.logger.debug(
                "state within capacity; skipping prune",
                extra=log_record(component=__name__),
            )
            return
        if self.state_prune_strategy == "score":
            items = sorted(
                self.cluster_map.items(),
                key=lambda kv: float(kv[1].get("score", 0.0)),
                reverse=True,
            )
        else:  # default "recent"
            items = sorted(
                self.cluster_map.items(),
                key=lambda kv: float(kv[1].get("ts", 0.0)),
                reverse=True,
            )
        self.cluster_map = dict(items[: self.state_capacity])

    # ------------------------------------------------------------------
    def begin_run(self, workflow_id: str, run_id: str) -> None:
        """Record the start of a workflow run.

        The fallback planner lacks advanced tracking, but we still persist a
        minimal record so ROI and stability metrics capture the run context.
        """

        self.logger.info(
            "begin run %s/%s",
            workflow_id,
            run_id,
            extra=log_record(workflow_id=workflow_id, run_id=run_id),
        )
        if self.roi_db is not None:
            try:
                self.roi_db.log_result(
                    workflow_id=workflow_id,
                    run_id=run_id,
                    runtime=0.0,
                    success_rate=0.0,
                    roi_gain=0.0,
                    workflow_synergy_score=0.0,
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    module_deltas={},
                )
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "ROI logging failed",
                    extra=log_record(workflow_id=workflow_id, run_id=run_id),
                    exc_info=exc,
                )
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(workflow_id, 0.0, 0.0, 0.0, roi_delta=0.0)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "stability logging failed",
                    extra=log_record(workflow_id=workflow_id, run_id=run_id),
                    exc_info=exc,
                )

    # ------------------------------------------------------------------
    def _domain(self, wid: str) -> str:
        return wid.split(".", 1)[0]

    def _score(
        self, chain: Sequence[str], roi_delta: float, entropy_delta: float, failures: int
    ) -> float:
        transitions = sum(
            1
            for i in range(1, len(chain))
            if self._domain(chain[i]) != self._domain(chain[i - 1])
        )
        return (
            self.roi_weight * roi_delta
            - self.domain_transition_penalty * transitions
            - self.entropy_weight * abs(entropy_delta)
            - self.stability_weight * failures
        )

    def discover_and_persist(
        self, workflows: Mapping[str, Callable[[], Any]]
    ) -> list[Mapping[str, Any]]:
        """Explore pipelines via heuristic mutations and persist state."""

        results: list[Mapping[str, Any]] = []
        with self._state_update():
            evaluated: set[tuple[str, ...]] = set()

            existing = [list(k) for k in self.cluster_map.keys()]

            for chain in existing:
                rec = self._evaluate_chain(chain, allow_existing=True)
                if rec:
                    key = tuple(rec["chain"])
                    evaluated.add(key)
                    results.append(rec)

            for wid in workflows:
                key = (wid,)
                if key in evaluated:
                    continue
                rec = self._evaluate_chain([wid])
                if rec:
                    evaluated.add(key)
                    results.append(rec)
                    existing.append([wid])

            for chain in existing:
                for rec in self.mutate_pipeline(chain, workflows):
                    key = tuple(rec["chain"])
                    if key not in evaluated:
                        evaluated.add(key)
                        results.append(rec)

                for rec in self.split_pipeline(chain, workflows):
                    key = tuple(rec["chain"])
                    if key not in evaluated:
                        evaluated.add(key)
                        results.append(rec)

            if len(existing) >= 2:
                for rec in self.remerge_pipelines(existing, workflows):
                    key = tuple(rec["chain"])
                    if key not in evaluated:
                        evaluated.add(key)
                        results.append(rec)

            self._prune_state()

        return sorted(results, key=lambda r: r["score"], reverse=True)

    # ------------------------------------------------------------------
    def _evaluate_chain(
        self, chain: Sequence[str], *, allow_existing: bool = False
    ) -> dict[str, Any] | None:
        if len(chain) != len(set(chain)):
            self.logger.debug(
                "rejecting chain %s due to cycle", "->".join(chain),
                extra=log_record(workflow_id="->".join(chain)),
            )
            return None
        if not allow_existing and tuple(chain) in self.cluster_map:
            self.logger.debug(
                "rejecting existing chain %s",
                "->".join(chain),
                extra=log_record(workflow_id="->".join(chain)),
            )
            return None

        roi_values: list[float] = []
        delta_rois: list[float] = []
        entropies: list[float] = []
        failures = 0

        for wid in chain:
            current_roi = 0.0
            moving_avg = 0.0
            delta_roi = 0.0
            if self.roi_db is not None:
                try:
                    stats = self.roi_db.fetch_chain_stats(wid)
                    current_roi = float(stats.get("last_roi", 0.0))
                    moving_avg = float(stats.get("moving_avg_roi", 0.0))
                    delta_roi = current_roi - moving_avg
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.warning(
                        "roi fetch failed", extra=log_record(workflow_id=wid), exc_info=exc
                    )
                    current_roi = moving_avg = delta_roi = 0.0

            stable = True
            entropy = 0.0
            if self.stability_db is not None:
                try:
                    entry = self.stability_db.data.get(wid, {})
                    failures += int(entry.get("failures", 0))
                    entropy = float(entry.get("entropy", 0.0))
                    stable = self.stability_db.is_stable(
                        wid, current_roi=current_roi, threshold=moving_avg
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.warning(
                        "stability check failed", extra=log_record(workflow_id=wid), exc_info=exc
                    )
                    stable = True

            roi_base = BASELINE_TRACKER.get("roi_delta")
            roi_tol = getattr(getattr(_init.settings, "roi", None), "deviation_tolerance", 0.0)
            if delta_roi < roi_base - roi_tol or not stable:
                self.logger.debug(
                    "rejecting chain %s", "->".join(chain), extra=log_record(workflow_id=wid)
                )
                return None

            roi_values.append(current_roi)
            delta_rois.append(delta_roi)
            entropies.append(entropy)

        if not roi_values:
            self.logger.debug(
                "no ROI values for chain %s", "->".join(chain),
                extra=log_record(workflow_id="->".join(chain)),
            )
            return None

        chain_roi = fmean(roi_values)
        chain_roi_delta = fmean(delta_rois) if delta_rois else chain_roi
        chain_entropy = fmean(entropies) if entropies else 0.0
        entropy_delta = chain_entropy - BASELINE_TRACKER.get("entropy")
        score = self._score(chain, chain_roi_delta, entropy_delta, failures)
        record = {
            "chain": list(chain),
            "roi_gain": chain_roi,
            "roi_delta": chain_roi_delta,
            "failures": failures,
            "entropy": chain_entropy,
            "entropy_delta": entropy_delta,
            "score": score,
        }
        self.cluster_map[tuple(chain)] = {
            "last_roi": chain_roi,
            "last_entropy": chain_entropy,
            "score": score,
            "failures": failures,
            "ts": time.time(),
        }
        self.logger.debug(
            "evaluated chain %s score %.3f",
            "->".join(chain),
            score,
            extra=log_record(workflow_id="->".join(chain)),
        )

        if self.roi_db is not None:
            try:
                self.roi_db.log_result(
                    workflow_id="->".join(chain),
                    run_id="evaluation",
                    runtime=0.0,
                    success_rate=1.0,
                    roi_gain=chain_roi,
                    workflow_synergy_score=max(0.0, 1.0 - chain_entropy),
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    code_entropy=chain_entropy,
                    entropy_delta=entropy_delta,
                    module_deltas={},
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                self.logger.exception(
                    "ROI logging failed",
                    extra=log_record(workflow_id="->".join(chain)),
                    exc_info=exc,
                )
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(
                    "->".join(chain), chain_roi, failures, chain_entropy, roi_delta=chain_roi
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                self.logger.exception(
                    "stability logging failed",
                    extra=log_record(workflow_id="->".join(chain)),
                    exc_info=exc,
                )

        return record

    # ------------------------------------------------------------------
    def _generate_mutations(
        self, chain: list[str], pool: list[str], depth: int
    ) -> list[list[str]]:
        if depth == 0:
            return []
        mutations: list[list[str]] = []
        for wid in pool:
            if wid in chain:
                continue
            mutated = chain + [wid]
            mutations.append(mutated)
            next_pool = [w for w in pool if w not in mutated]
            mutations.extend(self._generate_mutations(mutated, next_pool, depth - 1))
        return mutations

    def mutate_pipeline(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:
        """Create mutations by appending up to ``mutation_rate`` steps."""

        depth = max(1, int(self.mutation_rate))
        pool = [wid for wid in workflows if wid not in chain]
        candidate_chains = self._generate_mutations(list(chain), pool, depth)
        results: list[Mapping[str, Any]] = []
        for cand in candidate_chains:
            rec = self._evaluate_chain(cand)
            if rec:
                results.append(rec)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    def split_pipeline(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:
        """Split ``chain`` into two halves and score each half."""

        if len(chain) <= 1:
            return []

        mid = len(chain) // 2
        segments = [chain[:mid], chain[mid:]]
        results: list[Mapping[str, Any]] = []
        for seg in segments:
            rec = self._evaluate_chain(seg)
            if rec:
                results.append(rec)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    def remerge_pipelines(
        self,
        pipelines: Sequence[Sequence[str]],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:
        """Combine pipelines pairwise and score merged candidates."""

        results: list[Mapping[str, Any]] = []
        for i in range(len(pipelines)):
            for j in range(i + 1, len(pipelines)):
                merged = list(pipelines[i]) + [
                    w for w in pipelines[j] if w not in pipelines[i]
                ]
                rec = self._evaluate_chain(merged)
                if rec:
                    results.append(rec)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results


settings = _init.settings
DISCOVER_ORPHANS = False
RECURSIVE_ORPHANS = False
ROI_BACKOFF_THRESHOLD = 0.0
ROI_BACKOFF_CONSECUTIVE = 1
_roi_backoff_count = 0


def reload_settings(cfg: SandboxSettings) -> None:
    """Update module-level settings and derived constants."""
    global settings, PLANNER_INTERVAL, MUTATION_RATE, ROI_WEIGHT, DOMAIN_PENALTY, ENTROPY_THRESHOLD
    global SEARCH_DEPTH, BEAM_WIDTH, ENTROPY_WEIGHT
    global DISCOVER_ORPHANS, RECURSIVE_ORPHANS, ROI_BACKOFF_THRESHOLD, ROI_BACKOFF_CONSECUTIVE
    settings = cfg
    _validate_config(settings)
    PLANNER_INTERVAL = getattr(settings, "meta_planning_interval", 0)
    MUTATION_RATE = settings.meta_mutation_rate
    ROI_WEIGHT = settings.meta_roi_weight
    DOMAIN_PENALTY = settings.meta_domain_penalty
    ENTROPY_WEIGHT = settings.meta_entropy_weight
    SEARCH_DEPTH = settings.meta_search_depth
    BEAM_WIDTH = settings.meta_beam_width
    DISCOVER_ORPHANS = bool(
        getattr(settings, "include_orphans", True) and not getattr(settings, "disable_orphans", False)
    )
    RECURSIVE_ORPHANS = bool(getattr(settings, "recursive_orphan_scan", False))
    roi_settings = getattr(settings, "roi", None)
    ROI_BACKOFF_THRESHOLD = float(
        getattr(roi_settings, "stagnation_threshold", 0.0)
        if roi_settings is not None
        else 0.0
    )
    ROI_BACKOFF_CONSECUTIVE = int(
        getattr(roi_settings, "stagnation_cycles", 1)
        if roi_settings is not None
        else 1
    )
    if settings.meta_entropy_threshold is not None:
        ENTROPY_THRESHOLD = float(settings.meta_entropy_threshold)
    else:
        dev = getattr(settings, "entropy_deviation", 1.0)
        base = BASELINE_TRACKER.get("entropy")
        std = BASELINE_TRACKER.std("entropy")
        ENTROPY_THRESHOLD = base + dev * std


def _validate_config(cfg: SandboxSettings) -> None:
    """Validate meta planning configuration on import."""
    if cfg.meta_entropy_threshold is not None and not 0 <= cfg.meta_entropy_threshold <= 1:
        raise ValueError("meta_entropy_threshold must be between 0 and 1")
    for attr in (
        "meta_mutation_rate",
        "meta_roi_weight",
        "meta_domain_penalty",
        "meta_entropy_weight",
    ):
        if getattr(cfg, attr) < 0:
            raise ValueError(f"{attr} must be non-negative")
    for attr in ("meta_search_depth", "meta_beam_width"):
        if getattr(cfg, attr) <= 0:
            raise ValueError(f"{attr} must be positive")


reload_settings(settings)
_stable_workflows: WorkflowStabilityDB | None = None


def get_stable_workflows() -> WorkflowStabilityDB:
    """Return the cached :class:`WorkflowStabilityDB` instance.

    The database connection is created lazily on first access and wrapped in a
    ``try``/``except`` block so callers receive a descriptive ``RuntimeError``
    if initialisation fails.
    """

    global _stable_workflows
    if _stable_workflows is None:
        try:
            _stable_workflows = WorkflowStabilityDB()
        except Exception as exc:  # pragma: no cover - best effort
            raise RuntimeError("WorkflowStabilityDB initialisation failed") from exc
    return _stable_workflows


def _get_entropy_threshold(cfg: SandboxSettings, tracker: BaselineTracker) -> float:
    """Determine entropy threshold from baseline statistics."""
    threshold = cfg.meta_entropy_threshold
    if threshold is not None:
        return float(threshold)

    base = tracker.get("entropy")
    std = tracker.std("entropy")
    dev = getattr(cfg, "entropy_deviation", 1.0)
    return base + dev * std


def _percentile(data: Sequence[float], pct: float) -> float:
    """Return the *pct* percentile for *data* where *pct* is 0-1."""
    if not data:
        return 0.0
    if not 0 <= pct <= 1:
        raise ValueError("percentile must be between 0 and 1")
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def _get_overfit_thresholds(
    cfg: SandboxSettings, tracker: BaselineTracker
) -> tuple[int, float]:
    """Return dynamic error and entropy thresholds.

    ``max_allowed_errors`` and ``entropy_overfit_threshold`` may be configured
    explicitly on *cfg*.  When they are ``None`` the thresholds are derived from
    tracker histories using a percentile-based heuristic (default 95th
    percentile).  This keeps the fallback logic responsive to recent behaviour
    without requiring static configuration.
    """

    max_errors = getattr(cfg, "max_allowed_errors", None)
    if max_errors is None:
        pct = getattr(cfg, "error_overfit_percentile", 0.95)
        errors = tracker.to_dict().get("error_count", [])
        max_errors = _percentile(errors, pct)

    entropy_thresh = getattr(cfg, "entropy_overfit_threshold", None)
    if entropy_thresh is None:
        pct = getattr(cfg, "entropy_overfit_percentile", 0.95)
        deltas = [abs(x) for x in tracker.to_dict().get("entropy_delta", [])]
        entropy_thresh = _percentile(deltas, pct)

    return int(max_errors), float(entropy_thresh)


def _should_encode(
    record: Mapping[str, Any],
    tracker: BaselineTracker,
    *,
    entropy_threshold: float,
) -> tuple[bool, str]:
    """Determine if the latest cycle warrants encoding.

    The tracker is expected to be updated with the metrics contained in
    ``record`` *before* this function is invoked.  Deltas for all tracked
    metrics are therefore derived from :class:`BaselineTracker` rather than
    recomputed from raw ``record`` values.  A tuple ``(should_encode, reason)``
    is returned where ``reason`` provides additional context when encoding is
    skipped.
    """

    # Momentum-weighted ROI delta must be positive.
    momentum = getattr(tracker, "momentum", 1.0) or 1.0
    roi_delta = tracker.delta("roi") * momentum
    if roi_delta <= 0:
        return False, "no_delta"

    # All other tracked metrics (excluding momentum and entropy) require
    # positive deltas.
    for metric in tracker._history:
        if metric.endswith("_delta") or metric in {
            "roi",
            "momentum",
            "entropy",
        }:
            continue
        if tracker.delta(metric) < 0:
            return False, "no_delta"

    # Treat entropy spikes as potential overfitting even when ROI is high.
    history = tracker._history.get("entropy", [])
    prev_entropy = history[-2] if len(history) > 1 else (history[-1] if history else 0.0)
    current_entropy = float(record.get("entropy", prev_entropy))
    if abs(current_entropy - prev_entropy) >= float(entropy_threshold):
        return False, "entropy_spike"

    # Any failures or critical errors invalidate the improvement signal.
    if int(record.get("failures", 0)) > 0:
        return False, "errors_present"
    for err in record.get("errors", []) or []:
        sev = getattr(getattr(err, "error_type", None), "severity", None)
        if sev == "critical":
            return False, "errors_present"

    return True, "improved"


def _recent_error_entropy(
    error_log: Any | None,
    tracker: BaselineTracker,
    limit: int | None = None,
) -> tuple[Sequence[Any], float, int, float, float]:
    """Return recent error traces and entropy statistics.

    Parameters
    ----------
    error_log:
        Error log or telemetry object exposing ``recent_errors`` or
        ``recent_events``.  ``None`` is treated as no errors.
    tracker:
        Baseline tracker providing access to entropy history.

    Returns
    -------
    tuple[Sequence[Any], float, int, float, float]
        ``(events, entropy_delta, error_count, delta_mean, delta_std)`` where
        ``entropy_delta`` is the change in entropy since the previous update
        while ``delta_mean`` and ``delta_std`` are the rolling average and
        standard deviation of recent entropy deltas as tracked by
        :class:`BaselineTracker`.
    """

    events: Sequence[Any] | None = None
    # Allow callers or configuration to control the error history window.
    window = int(limit or getattr(_init.settings, "error_window", 5))
    try:
        if error_log is None:
            events = []
        elif isinstance(error_log, Sequence):
            events = error_log
        elif hasattr(error_log, "recent_errors"):
            events = error_log.recent_errors(limit=window)
        elif hasattr(error_log, "recent_events"):
            events = error_log.recent_events(limit=window)
        elif hasattr(getattr(error_log, "db", None), "recent_errors"):
            events = error_log.db.recent_errors(limit=window)  # type: ignore[attr-defined]
        else:
            events = []
    except Exception:
        events = []

    entropy_delta = getattr(tracker, "entropy_delta", 0.0)
    delta_mean = getattr(tracker, "get", lambda _m: 0.0)("entropy_delta")
    delta_std = getattr(tracker, "std", lambda _m: 0.0)("entropy_delta")
    error_count = len(events or [])
    return (
        list(events or []),
        float(entropy_delta),
        error_count,
        float(delta_mean),
        float(delta_std),
    )


# Canonical metrics that must be tracked for cycle evaluation.
REQUIRED_METRICS: tuple[str, ...] = ("roi", "pass_rate", "entropy")


def _evaluate_cycle(
    tracker: BaselineTracker, error_state: Any
) -> tuple[str, Mapping[str, Any]]:
    """Evaluate whether the improvement cycle should run.

    Parameters
    ----------
    tracker:
        Baseline metric tracker providing recent deltas.
    error_state:
        Error log or sequence of :class:`TelemetryEvent` instances used to
        detect critical failures.

    Returns
    -------
    tuple[str, Mapping[str, Any]]
        A decision of ``"run"`` or ``"skip"`` and a mapping containing the
        quantitative metric deltas along with a ``reason`` string.
    """

    metrics: dict[str, float] = {}
    history = getattr(tracker, "_history", {})
    for name in history.keys():
        if name.endswith("_delta"):
            continue
        metrics[name] = float(tracker.delta(name))

    missing = [m for m in REQUIRED_METRICS if m not in metrics]

    # Examine recent errors for critical severity
    critical = False
    events: Sequence[TelemetryEvent] | Sequence[Any] | None = None
    try:
        if isinstance(error_state, Sequence):
            events = error_state
        elif hasattr(error_state, "recent_errors"):
            events = error_state.recent_errors(limit=5)
        elif hasattr(error_state, "recent_events"):
            events = error_state.recent_events(limit=5)
        elif hasattr(getattr(error_state, "db", None), "recent_errors"):
            events = error_state.db.recent_errors(limit=5)  # type: ignore[attr-defined]
    except Exception:
        events = None

    threshold = getattr(_init.settings, "critical_severity_threshold", 75.0)
    severity_map = getattr(
        _init.settings, "severity_score_map", DEFAULT_SEVERITY_SCORE_MAP
    )

    def _severity_to_score(sev: Any) -> float | None:
        if isinstance(sev, str):
            s = sev.lower()
            if s in severity_map:
                return severity_map[s]
            try:
                sev = float(sev)
            except ValueError:
                return None
        if isinstance(sev, (int, float)):
            val = float(sev)
            if 0 <= val <= 1:
                return val * 100
            if 0 <= val <= 5:
                return val * 20
            if 0 <= val <= 10:
                return val * 10
            return val
        return None

    for ev in events or []:
        sev = getattr(getattr(ev, "error_type", None), "severity", None)
        score = _severity_to_score(sev)
        if score is not None and score >= threshold:
            critical = True
            break
    logger = get_logger(__name__)

    global _roi_backoff_count
    roi_delta = float(metrics.get("roi", 0.0))
    if roi_delta <= ROI_BACKOFF_THRESHOLD:
        _roi_backoff_count += 1
    else:
        _roi_backoff_count = 0

    if (
        ROI_BACKOFF_THRESHOLD > 0
        and _roi_backoff_count >= max(1, ROI_BACKOFF_CONSECUTIVE)
    ):
        logger.debug(
            "cycle evaluation paused; roi stagnation detected",
            extra=log_record(
                reason="roi_backoff",
                roi_delta=roi_delta,
                roi_backoff_threshold=ROI_BACKOFF_THRESHOLD,
                roi_backoff_count=_roi_backoff_count,
            ),
        )
        return "skip", {
            "reason": "roi_backoff",
            "metrics": metrics,
            "backoff_count": _roi_backoff_count,
        }

    if not missing and not critical and metrics and all(v > 0 for v in metrics.values()):
        logger.debug(
            "cycle evaluation skipped; all deltas positive",
            extra=log_record(reason="all_deltas_positive", metrics=metrics),
        )
        return "skip", {"reason": "all_deltas_positive", "metrics": metrics}
    if missing:
        logger.debug(
            "cycle evaluation running; missing metrics",
            extra=log_record(reason="missing_metrics", metrics=metrics, missing=missing),
        )
        return "run", {"reason": "missing_metrics", "metrics": metrics, "missing": missing}
    if critical:
        logger.debug(
            "cycle evaluation running; critical error detected",
            extra=log_record(reason="critical_error", metrics=metrics),
        )
        reason = "critical_error"
    else:
        reason = "non_positive_delta"
        logger.debug(
            "cycle evaluation running; non-positive delta",
            extra=log_record(reason=reason, metrics=metrics),
        )
    return "run", {"reason": reason, "metrics": metrics}


async def self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    interval: float = PLANNER_INTERVAL,
    event_bus: UnifiedEventBus | None = None,
    error_log: Any | None = None,
    stop_event: threading.Event | None = None,
    tick_state: _CycleTickState | None = None,
    should_encode: Callable[[Mapping[str, Any], "BaselineTracker", float], tuple[bool, str]]
    | None = None,
    evaluate_cycle: Callable[["BaselineTracker", Any | None], tuple[str, Mapping[str, Any]]]
    = _evaluate_cycle,
) -> None:
    """Background loop evolving ``workflows`` using the meta planner."""
    logger = get_logger("SelfImprovementCycle")
    cfg = _init.settings
    planner_cls = resolve_meta_workflow_planner()
    if planner_cls is None:
        if getattr(cfg, "enable_meta_planner", False):
            raise RuntimeError("MetaWorkflowPlanner required but not installed")
        logger.warning("MetaWorkflowPlanner unavailable; using fallback planner")
        planner = _FallbackPlanner()
    else:
        planner = planner_cls(context_builder=create_context_builder())

    mutation_rate = cfg.meta_mutation_rate
    roi_weight = cfg.meta_roi_weight
    domain_penalty = cfg.meta_domain_penalty
    stagnated_workflows: dict[str, dict[str, float]] = {}
    threshold = ROI_BACKOFF_THRESHOLD if ROI_BACKOFF_THRESHOLD != 0 else 0.0
    streak_required = max(1, ROI_BACKOFF_CONSECUTIVE)
    controller: WorkflowCycleController | None = None

    def _controller_status(
        workflow_id: str, status: str, stats: Mapping[str, Any]
    ) -> None:
        payload = {
            "workflow_id": workflow_id,
            "status": status,
            "roi_delta": float(stats.get("delta_roi", 0.0)),
            "non_positive_streak": int(stats.get("non_positive_streak", 0)),
            "threshold": float(stats.get("threshold", threshold)),
            "streak_required": int(stats.get("streak_required", streak_required)),
            "iterations": int(stats.get("iterations", 0)),
        }
        logger.info(
            "workflow %s controller status %s (roi delta %.4f)",
            workflow_id,
            status,
            payload["roi_delta"],
            extra=log_record(
                workflow_id=workflow_id,
                roi_delta=payload["roi_delta"],
                non_positive_streak=payload["non_positive_streak"],
                streak_required=payload["streak_required"],
                controller_status=status,
                iterations=payload["iterations"],
            ),
        )
        if event_bus is not None:
            try:
                event_bus.publish("meta-planning.workflow-status", payload)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(
                    "failed to publish workflow controller status",
                    extra=log_record(workflow_id=workflow_id),
                    exc_info=exc,
                )

    if WorkflowCycleController is not None:
        try:
            controller = WorkflowCycleController(
                roi_db=getattr(planner, "roi_db", None),
                threshold=threshold,
                streak_required=streak_required,
                status_callback=_controller_status,
                logger=logger,
            )
            try:
                setattr(planner, "workflow_controller", controller)
            except Exception:
                logger.debug(
                    "planner does not expose workflow_controller hook",
                    extra=log_record(component=type(planner).__name__),
                )
        except Exception:
            logger.exception("failed to initialize workflow controller")
            controller = None

    for name, value in {
        "mutation_rate": mutation_rate,
        "roi_weight": roi_weight,
        "domain_transition_penalty": domain_penalty,
    }.items():
        if hasattr(planner, name):
            setattr(planner, name, value)

    stability_db = get_stable_workflows()
    setattr(planner, "entropy_threshold", _get_entropy_threshold(cfg, BASELINE_TRACKER))
    repo_root_fn = getattr(_init, "_repo_path", lambda: Path("."))
    repo_path = Path(resolve_path(repo_root_fn()))
    try:  # pragma: no cover - optional snapshot dependencies
        from .snapshot_tracker import SnapshotTracker  # local import to avoid heavy deps
        snapshot_tracker = SnapshotTracker()
    except Exception:  # pragma: no cover - best effort fallback
        snapshot_tracker = None
    stage_timeout = float(getattr(cfg, "meta_planner_stage_timeout", 300.0))
    cycle_workflow_id = "meta_planning_cycle"

    class _StageTimeoutError(RuntimeError):
        """Signals a stage timeout in the meta planning cycle."""

        def __init__(self, stage: str) -> None:
            super().__init__(f"meta planning stage timed out: {stage}")
            self.stage = stage

    async def _run_stage(stage: str, action: Awaitable[Any], *, workflow_id: str) -> Any:
        start = time.perf_counter()
        logger.info(
            "meta planning stage start",
            extra=log_record(
                stage=stage,
                workflow_id=workflow_id,
                elapsed_s=0.0,
                timeout_s=stage_timeout,
            ),
        )
        try:
            result = await asyncio.wait_for(action, timeout=stage_timeout)
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            logger.warning(
                "meta planning stage timeout",
                extra=log_record(
                    stage=stage,
                    workflow_id=workflow_id,
                    elapsed_s=elapsed,
                    timeout_s=stage_timeout,
                ),
            )
            raise _StageTimeoutError(stage) from None
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.exception(
                "meta planning stage failed",
                extra=log_record(stage=stage, workflow_id=workflow_id, elapsed_s=elapsed),
                exc_info=exc,
            )
            raise
        elapsed = time.perf_counter() - start
        logger.info(
            "meta planning stage completed",
            extra=log_record(stage=stage, workflow_id=workflow_id, elapsed_s=elapsed),
        )
        return result

    async def _log(record: Mapping[str, Any]) -> None:
        chain = record.get("chain", [])
        cid = "->".join(chain)
        roi = float(record.get("roi_gain", 0.0))
        failures = int(record.get("failures", 0))
        entropy = float(record.get("entropy", 0.0))
        entropy_delta = float(
            record.get("entropy_delta", entropy - BASELINE_TRACKER.get("entropy"))
        )
        try:
            from .metrics import record_entropy as _record_entropy
            _record_entropy(
                float(record.get("code_diversity", entropy)),
                float(record.get("token_complexity", entropy)),
                roi=roi,
            )
        except Exception as exc:
            logger.warning(
                "entropy metric recording failed",
                extra=log_record(workflow_id=cid),
                exc_info=exc,
            )
        if planner.roi_db is not None:
            try:
                planner.roi_db.log_result(
                    workflow_id=cid,
                    run_id="bg",
                    runtime=0.0,
                    success_rate=1.0,
                    roi_gain=roi,
                    workflow_synergy_score=max(0.0, 1.0 - entropy),
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    code_entropy=entropy,
                    entropy_delta=entropy_delta,
                    module_deltas={},
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                logger.exception(
                    "ROI logging failed", extra=log_record(workflow_id=cid), exc_info=exc
                )
        chain_stats: dict[str, float] = {}
        try:
            if planner.roi_db is not None:
                chain_stats = planner.roi_db.fetch_chain_stats(cid)
        except Exception as exc:  # pragma: no cover - logging best effort
            logger.debug(
                "roi chain stats fetch failed",
                extra=log_record(workflow_id=cid),
                exc_info=exc,
            )
        if planner.stability_db is not None:
            try:
                planner.stability_db.record_metrics(
                    cid, roi, failures, entropy, roi_delta=roi
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                logger.exception(
                    "stability logging failed", extra=log_record(workflow_id=cid), exc_info=exc
                )
        if event_bus is not None:
            try:
                event_bus.publish(
                    "metrics:new",
                    {
                        "bot": cid,
                        "errors": failures,
                        "entropy": entropy,
                        "expense": 1.0,
                        "revenue": 1.0 + roi,
                        "roi_delta": chain_stats.get("delta_roi", roi),
                    },
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.exception(
                    "failed to publish metrics",
                    extra=log_record(workflow_id=cid),
                    exc_info=exc,
                )

    def _debug_cycle(
        outcome: str,
        *,
        reason: str | None = None,
        **extra: Any,
    ) -> None:
        tracker = BASELINE_TRACKER
        metrics: dict[str, float] = {}
        for name in getattr(tracker, "_history", {}):
            if name.endswith("_delta"):
                continue
            if name == "entropy":
                metrics["entropy_delta"] = float(
                    getattr(tracker, "entropy_delta", tracker.delta("entropy"))
                )
            else:
                metrics[f"{name}_delta"] = float(tracker.delta(name))
        extra_fields: dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            else:
                extra_fields[key] = value
        logger.debug(
            "cycle",
            extra=log_record(
                outcome=outcome,
                reason=reason,
                **metrics,
                **extra_fields,
            ),
        )

    def _is_stagnant(chain_id: str, roi_delta: float | None = None) -> tuple[bool, dict[str, float]]:
        threshold = ROI_BACKOFF_THRESHOLD if ROI_BACKOFF_THRESHOLD != 0 else 0.0
        streak_required = max(1, ROI_BACKOFF_CONSECUTIVE)
        stats: dict[str, float] = {}
        try:
            if planner.roi_db is not None:
                stats = planner.roi_db.fetch_chain_stats(chain_id)
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(
                "stagnation stats lookup failed",
                extra=log_record(workflow_id=chain_id),
                exc_info=exc,
            )
            return False, stats
        delta = float(stats.get("delta_roi", roi_delta or 0.0))
        streak = int(stats.get("non_positive_streak", 0))
        stagnant = streak >= streak_required and delta <= threshold
        if stagnant:
            stagnated_workflows[chain_id] = stats
        else:
            stagnated_workflows.pop(chain_id, None)
        return stagnant, stats

    if evaluate_cycle is None:
        raise ValueError("evaluate_cycle callable required")

    while True:
        if stop_event is not None and stop_event.is_set():
            _debug_cycle("skipped", reason="stop_event")
            break
        if tick_state is not None:
            tick_snapshot = tick_state.tick()
            logger.info(
                "cycle tick",
                extra=log_record(
                    tick_timestamp=tick_snapshot["last_tick"],
                    tick_count=tick_snapshot["tick_count"],
                    tick_thread_name=tick_snapshot["last_tick_thread_name"],
                    tick_thread_ident=tick_snapshot["last_tick_thread_ident"],
                ),
            )
        cycle_ok = False
        try:
            decision, info = evaluate_cycle(BASELINE_TRACKER, error_log)
            if info.get("reason") == "missing_metrics":
                _debug_cycle(
                    "run",
                    reason="missing_metrics",
                    missing_metrics=",".join(info.get("missing", [])),
                )
            if decision == "skip":
                traces, ent_delta, err_count, delta_mean, delta_std = _recent_error_entropy(
                    error_log,
                    BASELINE_TRACKER,
                    getattr(cfg, "error_window", 5),
                )
                max_errors, z_threshold = _get_overfit_thresholds(cfg, BASELINE_TRACKER)
                z_score = (
                    abs(ent_delta - delta_mean) / delta_std if delta_std > 0 else 0.0
                )
                if err_count > max_errors or z_score > z_threshold:
                    logger.debug(
                        "fallback_overfitting",
                        extra=log_record(
                            decision="run",
                            reason="fallback_overfitting",
                            entropy_delta=ent_delta,
                            entropy_z=z_score,
                            errors=err_count,
                            max_allowed_errors=max_errors,
                            entropy_overfit_threshold=z_threshold,
                        ),
                    )
                    decision = "run"
                    _debug_cycle(
                        "fallback",
                        reason="overfitting",
                        errors=err_count,
                        entropy_delta=ent_delta,
                        entropy_z=z_score,
                        entropy_overfit_threshold=z_threshold,
                        error_traces=traces,
                    )
                else:
                    _debug_cycle("skipped", reason=info.get("reason"))
                    continue

            if snapshot_tracker is not None:
                await _run_stage(
                    "snapshot_before",
                    asyncio.to_thread(
                        snapshot_tracker.capture,
                        "before",
                        {
                            "files": list(repo_path.rglob("*.py")),
                            "roi": BASELINE_TRACKER.current("roi"),
                            "sandbox_score": get_latest_sandbox_score(
                                SandboxSettings().sandbox_score_db
                            ),
                        },
                        repo_path=repo_path,
                    ),
                    workflow_id=cycle_workflow_id,
                )
            records = await _run_stage(
                "discover_and_persist",
                asyncio.to_thread(planner.discover_and_persist, workflows),
                workflow_id=cycle_workflow_id,
            )
            active: list[list[str]] = []
            orphan_workflows: list[str] = []
            if DISCOVER_ORPHANS:
                try:
                    integrate_result = await _run_stage(
                        "integrate_orphans",
                        integrate_orphans(recursive=RECURSIVE_ORPHANS),
                        workflow_id=cycle_workflow_id,
                    )
                    if integrate_result:
                        orphan_workflows.extend(
                            w for w in integrate_result if isinstance(w, str)
                        )
                except _StageTimeoutError:
                    raise
                except Exception:
                    logger.exception("orphan integration failed")

                if RECURSIVE_ORPHANS:
                    try:
                        scan_result = await _run_stage(
                            "post_round_orphan_scan",
                            post_round_orphan_scan(recursive=True),
                            workflow_id=cycle_workflow_id,
                        )
                        integrated = (
                            scan_result.get("integrated")
                            if isinstance(scan_result, dict)
                            else None
                        )
                        if integrated:
                            orphan_workflows.extend(
                                w for w in integrated if isinstance(w, str)
                            )
                    except _StageTimeoutError:
                        raise
                    except Exception:
                        logger.exception("recursive orphan discovery failed")

            if orphan_workflows:
                unique_orphans: list[str] = []
                seen_orphans: set[str] = set()
                for wf in orphan_workflows:
                    if wf not in seen_orphans:
                        seen_orphans.add(wf)
                        unique_orphans.append(wf)
                for wid in unique_orphans:
                    chain = [wid]
                    if tuple(chain) not in planner.cluster_map:
                        planner.cluster_map[tuple(chain)] = {
                            "last_roi": 0.0,
                            "last_entropy": 0.0,
                            "score": 0.0,
                            "failures": 0,
                            "ts": time.time(),
                        }
                        try:
                            planner._save_state()
                        except Exception:
                            logger.exception("failed to persist orphan workflows")
                    active.append(chain)
            outcome_logged = False
            for rec in records:
                await _log(rec)
                roi = float(rec.get("roi_gain", 0.0))
                failures = int(rec.get("failures", 0))
                entropy = float(rec.get("entropy", 0.0))
                pass_rate = float(rec.get("pass_rate", 1.0 if failures == 0 else 0.0))
                repo = _init._repo_path()
                try:
                    from .metrics import compute_call_graph_complexity
                    call_complexity = compute_call_graph_complexity(repo)
                except Exception:
                    call_complexity = 0.0
                try:
                    from .metrics import compute_entropy_metrics
                    _, _, token_div = compute_entropy_metrics(
                        list(repo.rglob("*.py")), settings=cfg
                    )
                except Exception:
                    token_div = 0.0
                BASELINE_TRACKER.update(
                    roi=roi,
                    pass_rate=pass_rate,
                    entropy=entropy,
                    call_graph_complexity=call_complexity,
                    token_diversity=token_div,
                )
                tracker = BASELINE_TRACKER
                encode_fn = should_encode
                chain_id = "->".join(rec.get("chain", [])) if rec.get("chain") else None
                stagnant_stats: dict[str, float] = {}
                controller_state: Mapping[str, Any] | None = None
                if chain_id:
                    stagnant, stagnant_stats = _is_stagnant(chain_id, roi_delta=roi)
                    controller_state = record_workflow_iteration(
                        chain_id,
                        roi_gain=roi,
                        roi_delta=float(stagnant_stats.get("delta_roi", roi)),
                        threshold=ROI_BACKOFF_THRESHOLD,
                        patience=max(1, ROI_BACKOFF_CONSECUTIVE),
                    )
                    if controller_state.get("status") == "halted":
                        _debug_cycle(
                            "skipped",
                            reason="controller_threshold",
                            roi_delta=controller_state.get("last_delta", 0.0),
                            workflow_id=chain_id,
                            controller_threshold=controller_state.get("threshold", 0.0),
                        )
                        outcome_logged = True
                        logger.info(
                            "workflow %s halted by ROI controller",
                            chain_id,
                            extra=log_record(
                                workflow_id=chain_id,
                                roi_delta=controller_state.get("last_delta", 0.0),
                                non_positive_streak=stagnant_stats.get(
                                    "non_positive_streak", 0
                                ),
                                decision="controller_halt",
                                controller_threshold=controller_state.get(
                                    "threshold", 0.0
                                ),
                            ),
                        )
                        continue
                    if stagnant:
                        _debug_cycle(
                            "skipped",
                            reason="roi_stagnation",
                            roi_delta=stagnant_stats.get("delta_roi", 0.0),
                            stagnation_streak=stagnant_stats.get("non_positive_streak", 0),
                            workflow_id=chain_id,
                        )
                        outcome_logged = True
                        logger.info(
                            "workflow %s paused due to ROI stagnation",
                            chain_id,
                            extra=log_record(
                                workflow_id=chain_id,
                                roi_delta=stagnant_stats.get("delta_roi", 0.0),
                                non_positive_streak=stagnant_stats.get(
                                    "non_positive_streak", 0
                                ),
                                decision="pause",
                            ),
                        )
                        continue
                    if controller is not None:
                        active_chain, stats = controller.record(chain_id, roi_delta=roi)
                        stagnant_stats = {k: float(v) for k, v in stats.items()}
                        if not active_chain:
                            stagnated_workflows[chain_id] = stagnant_stats
                            _debug_cycle(
                                "skipped",
                                reason="roi_stagnation",
                                roi_delta=stagnant_stats.get("delta_roi", 0.0),
                                stagnation_streak=stagnant_stats.get(
                                    "non_positive_streak", 0
                                ),
                                workflow_id=chain_id,
                            )
                            outcome_logged = True
                            logger.info(
                                "workflow %s paused due to ROI stagnation",
                                chain_id,
                                extra=log_record(
                                    workflow_id=chain_id,
                                    roi_delta=stagnant_stats.get("delta_roi", 0.0),
                                    non_positive_streak=stagnant_stats.get(
                                        "non_positive_streak", 0
                                    ),
                                    decision="pause",
                                ),
                            )
                            continue
                        else:
                            stagnated_workflows.pop(chain_id, None)
                    else:
                        stagnant, stagnant_stats = _is_stagnant(chain_id, roi_delta=roi)
                        if stagnant:
                            _debug_cycle(
                                "skipped",
                                reason="roi_stagnation",
                                roi_delta=stagnant_stats.get("delta_roi", 0.0),
                                stagnation_streak=stagnant_stats.get("non_positive_streak", 0),
                                workflow_id=chain_id,
                            )
                            outcome_logged = True
                            logger.info(
                                "workflow %s paused due to ROI stagnation",
                                chain_id,
                                extra=log_record(
                                    workflow_id=chain_id,
                                    roi_delta=stagnant_stats.get("delta_roi", 0.0),
                                    non_positive_streak=stagnant_stats.get(
                                        "non_positive_streak", 0
                                    ),
                                    decision="pause",
                                ),
                            )
                            continue
                if encode_fn is None:
                    momentum = getattr(tracker, "momentum", 1.0) or 1.0
                    roi_delta = tracker.delta("roi") * momentum
                    if roi_delta <= 0:
                        should_encode, reason = False, "no_delta"
                    else:
                        skip_reason = None
                        for metric in tracker._history:
                            if metric.endswith("_delta") or metric in {
                                "roi",
                                "momentum",
                                "entropy",
                            }:
                                continue
                            if tracker.delta(metric) <= 0:
                                skip_reason = "no_delta"
                                break
                        if skip_reason is not None:
                            should_encode, reason = False, skip_reason
                        elif abs(tracker.delta("entropy")) > float(
                            cfg.overfitting_entropy_threshold
                        ):
                            should_encode, reason = False, "entropy_spike"
                        elif failures > 0:
                            should_encode, reason = False, "errors_present"
                        else:
                            should_encode, reason = True, "improved"
                else:
                    should_encode, reason = encode_fn(
                        rec,
                        tracker,
                        cfg.overfitting_entropy_threshold,
                    )
                if not should_encode:
                    outcome = (
                        "fallback" if reason in {"entropy_spike", "errors_present"} else "skipped"
                    )
                    _debug_cycle(outcome, reason=reason)
                    outcome_logged = True
                    continue
                else:
                    _debug_cycle("improved")
                    outcome_logged = True
                chain = rec.get("chain", [])
                if chain and roi > 0:
                    active.append(chain)
                    chain_id = "->".join(chain)
                    try:
                        stability_db.record_metrics(
                            chain_id, roi, failures, entropy, roi_delta=roi
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.exception(
                            "global stability logging failed",
                            extra=log_record(workflow_id=chain_id),
                            exc_info=exc,
                        )
            if not outcome_logged:
                _debug_cycle("skipped", reason="no_records")

            for chain in list(active):
                planner.cluster_map.pop(tuple(chain), None)

            if snapshot_tracker is not None:
                await _run_stage(
                    "snapshot_after",
                    asyncio.to_thread(
                        snapshot_tracker.capture,
                        "after",
                        {
                            "files": list(repo_path.rglob("*.py")),
                            "roi": BASELINE_TRACKER.current("roi"),
                            "sandbox_score": get_latest_sandbox_score(
                                SandboxSettings().sandbox_score_db
                            ),
                        },
                        repo_path=repo_path,
                    ),
                    workflow_id=cycle_workflow_id,
                )
                delta = snapshot_tracker.delta()
                _debug_cycle(
                    "snapshot",
                    roi_delta=delta.get("roi", 0.0),
                    entropy_delta=delta.get("entropy", 0.0),
                )
            cycle_ok = True
        except _StageTimeoutError as exc:
            _debug_cycle("timeout", reason=exc.stage)
            logger.warning(
                "meta planning stage timed out; deferring to next tick",
                extra=log_record(stage=exc.stage, workflow_id=cycle_workflow_id),
            )
        except Exception as exc:  # pragma: no cover - planner is best effort
            _debug_cycle("error", reason=str(exc))
            logger.debug("error", extra=log_record(err=str(exc)))
            logger.exception("meta planner execution failed", exc_info=exc)
        finally:
            tick_snapshot = tick_state.snapshot() if tick_state is not None else None
            if cycle_ok:
                logger.info(
                    "cycle tick completed; scheduling next run in %ss",
                    interval,
                    extra=log_record(
                        tick_count=tick_snapshot.get("tick_count")
                        if tick_snapshot
                        else None,
                        last_tick=tick_snapshot.get("last_tick") if tick_snapshot else None,
                        sleep_interval=interval,
                    ),
                )
            else:
                logger.info(
                    "cycle tick completed after error; scheduling next run in %ss",
                    interval,
                    extra=log_record(
                        tick_count=tick_snapshot.get("tick_count")
                        if tick_snapshot
                        else None,
                        last_tick=tick_snapshot.get("last_tick") if tick_snapshot else None,
                        sleep_interval=interval,
                    ),
                )
            await asyncio.sleep(interval)


def start_self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    event_bus: UnifiedEventBus | None = None,
    interval: float = PLANNER_INTERVAL,
    error_log: Any | None = None,
    workflow_graph: Any | None = None,
    should_encode: Callable[
        [Mapping[str, Any], "BaselineTracker", float], tuple[bool, str]
    ]
    | None = None,
    evaluate_cycle: Callable[["BaselineTracker", Any | None], tuple[str, Mapping[str, Any]]]
    = _evaluate_cycle,
):
    """Prepare a background thread running :func:`self_improvement_cycle`.

    The returned object exposes ``start()``, ``join()`` and ``stop()`` methods.
    It captures exceptions raised inside the background task and re-raises them
    when ``join()`` is invoked.  ``stop()`` gracefully cancels the running
    coroutine.
    """

    if evaluate_cycle is None:
        raise ValueError("evaluate_cycle callable required")

    def _env_flag(name: str) -> bool | None:
        val = os.getenv(name)
        if val is None:
            return None
        return val.lower() in {"1", "true", "yes", "on"}

    try:
        settings = SandboxSettings()
    except Exception:
        settings = None

    include_env = _env_flag("SANDBOX_INCLUDE_ORPHANS")
    recursive_env = _env_flag("SANDBOX_RECURSIVE_ORPHANS")

    include_cfg = getattr(settings, "include_orphans", None) if settings else None
    disable_cfg = getattr(settings, "disable_orphans", False) if settings else False
    recursive_cfg = getattr(settings, "recursive_orphan_scan", False) if settings else False

    include_orphans = (
        include_env
        if include_env is not None
        else (bool(include_cfg) if include_cfg is not None else False)
    )
    discover_orphans = bool(include_orphans and not disable_cfg)

    recursive_orphans = (
        recursive_env if recursive_env is not None else bool(recursive_cfg)
    )

    workflow_plan: dict[str, Callable[[], Any]] = dict(workflows)

    if discover_orphans:
        workflow_plan.setdefault(
            "integrate_orphans",
            lambda recursive=recursive_orphans: integrate_orphans(
                recursive=recursive
            ),
        )
        if recursive_orphans:
            workflow_plan.setdefault(
                "recursive_orphan_scan",
                lambda: post_round_orphan_scan(recursive=True),
            )

    if discover_orphans:
        try:
            integrate_orphans_sync(recursive=recursive_orphans)
            if recursive_orphans:
                post_round_orphan_scan_sync(recursive=True)
        except Exception:
            get_logger(__name__).exception(
                "startup orphan discovery failed",
                extra=log_record(event="pre-cycle-orphans"),
            )

    try:
        _init.workflow_graph = workflow_graph
    except Exception:
        get_logger(__name__).debug(
            "workflow graph injection failed", extra=log_record(component=__name__)
        )

    print("💡 SI-7: preparing self-improvement cycle thread scaffold")
    tick_state = _CycleTickState()
    loop_heartbeat_state = _LoopHeartbeatState()
    loop_ping_state = _LoopPingState()

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    class _CycleThread:
        def __init__(self, stop_event: threading.Event) -> None:
            self._loop: asyncio.AbstractEventLoop | None = None
            self._task: asyncio.Task[None] | None = None
            self._heartbeat_task: asyncio.Task[None] | None = None
            self._stop_task: asyncio.Task[None] | None = None
            self._exc: queue.Queue[BaseException] = queue.Queue()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._stop_event = stop_event
            self._error_log = error_log
            self._should_encode = should_encode
            self._evaluate_cycle = evaluate_cycle
            self._heartbeat_interval = max(1.0, min(5.0, interval / 2.0))
            self._cancel_requested = threading.Event()

        # --------------------------------------------------
        def _run(self) -> None:
            from inspect import signature

            print("💡 SI-8: starting self-improvement event loop")
            monitor_state["loop_heartbeat_started"] = time.monotonic()

            kwargs: dict[str, Any] = {
                "interval": interval,
                "event_bus": event_bus,
            }
            if "stop_event" in signature(self_improvement_cycle).parameters:
                kwargs["stop_event"] = self._stop_event
            if "error_log" in signature(self_improvement_cycle).parameters:
                kwargs["error_log"] = self._error_log
            if "tick_state" in signature(self_improvement_cycle).parameters:
                kwargs["tick_state"] = tick_state
            if "should_encode" in signature(self_improvement_cycle).parameters:
                kwargs["should_encode"] = self._should_encode
            if "evaluate_cycle" in signature(self_improvement_cycle).parameters:
                kwargs["evaluate_cycle"] = self._evaluate_cycle
            print("💡 SI-9: scheduling self-improvement cycle coroutine")

            async def _loop_heartbeat() -> None:
                while True:
                    if self._stop_event.is_set() or self._cancel_requested.is_set():
                        break
                    monitor_state["last_loop_heartbeat"] = time.monotonic()
                    loop_heartbeat_state.beat()
                    await asyncio.sleep(self._heartbeat_interval)

            async def _run_cycle() -> None:
                await self_improvement_cycle(workflow_plan, **kwargs)

            async def _await_stop() -> None:
                await asyncio.to_thread(self._stop_event.wait)

            async def _main() -> None:
                self._task = asyncio.create_task(_run_cycle())
                self._heartbeat_task = asyncio.create_task(_loop_heartbeat())
                self._stop_task = asyncio.create_task(_await_stop())
                done, pending = await asyncio.wait(
                    {self._task, self._stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._stop_task in done and self._task not in done:
                    self._task.cancel()
                for task in pending:
                    task.cancel()
                task_list = [self._task, self._heartbeat_task, self._stop_task]
                results = await asyncio.gather(
                    *task_list, return_exceptions=True  # type: ignore[arg-type]
                )
                for task, result in zip(task_list, results):
                    if not isinstance(result, BaseException):
                        continue
                    if (
                        task is self._task
                        and isinstance(result, asyncio.CancelledError)
                    ):
                        msg = getattr(self._task, "_cancel_message", None)
                        reason = str(msg) if msg else "cancelled"
                        logger = get_logger(__name__)
                        logger.info(
                            "self improvement cycle cancelled",
                            extra=log_record(reason=reason),
                        )
                        try:  # pragma: no cover - metrics are best effort
                            from menace_sandbox.metrics_exporter import (
                                self_improvement_failure_total,
                            )

                            self_improvement_failure_total.labels(reason=reason).inc()
                        except Exception as exc:
                            logger.debug(
                                "cancellation metric update failed",
                                extra=log_record(reason=reason),
                                exc_info=exc,
                            )
                    elif task is self._task:
                        self._exc.put(result)

            try:
                with asyncio.Runner() as runner:
                    self._loop = runner.get_loop()
                    runner.run(_main())
            except BaseException as exc:  # pragma: no cover - best effort
                self._exc.put(exc)

        # --------------------------------------------------
        def start(self) -> None:
            print("💡 SI-10: launching self-improvement thread")
            self._thread.start()

        def join(self, timeout: float | None = None) -> None:
            print("💡 SI-11: joining self-improvement thread")
            effective_timeout = join_timeout if timeout is None else timeout
            self._thread.join(effective_timeout)
            if self._thread.is_alive():
                logger = get_logger(__name__)
                logger.warning(
                    "self improvement cycle zombie loop detected; continuing without waiting",
                    extra=log_record(
                        thread_ident=self._thread.ident,
                        join_timeout_seconds=effective_timeout,
                    ),
                )
                return
            if not self._exc.empty():
                raise self._exc.get()

        def stop(self, timeout: float | None = 5.0) -> None:
            print("💡 SI-12: stopping self-improvement thread")
            effective_timeout = stop_timeout if timeout is None else timeout
            self._stop_event.set()
            self._cancel_requested.set()
            loop = self._loop
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(
                        lambda: self._task.cancel() if self._task is not None else None
                    )
                except RuntimeError:
                    pass
            self._thread.join(effective_timeout)
            if self._thread.is_alive():
                heartbeat_snapshot = loop_heartbeat_state.snapshot()
                last_heartbeat = heartbeat_snapshot.get("last_heartbeat")
                now = time.time()
                last_heartbeat_age = (
                    (now - last_heartbeat) if last_heartbeat is not None else None
                )
                try:
                    loop_running = self._loop.is_running() if self._loop else False
                except Exception:
                    loop_running = False
                stack_dump = _get_thread_stack(self._thread.ident)
                monitor_state["stuck_thread"] = {
                    "thread": self._thread,
                    "ident": self._thread.ident,
                    "detected_at": now,
                    "stack": stack_dump,
                }
                logger = get_logger(__name__)
                message = "self improvement loop unresponsive during stop; continuing without waiting"
                if stack_dump:
                    message = f"{message}\n{stack_dump}"
                logger.critical(
                    message,
                    extra=log_record(
                        thread_ident=self._thread.ident,
                        loop_running=loop_running,
                        last_heartbeat_timestamp=last_heartbeat,
                        last_heartbeat_age_seconds=last_heartbeat_age,
                        stop_timeout_seconds=effective_timeout,
                        stuck_thread_stack=stack_dump,
                    ),
                )
                return
            if not self._exc.empty():
                raise self._exc.get()

        def request_stop(self) -> None:
            self._stop_event.set()
            self._cancel_requested.set()
            loop = self._loop
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(
                        lambda: self._task.cancel() if self._task is not None else None
                    )
                except RuntimeError:
                    pass

        def state(self) -> dict[str, Any]:
            task = self._task
            try:
                loop_running = self._loop.is_running() if self._loop else False
            except Exception:
                loop_running = False
            return {
                "thread_alive": self._thread.is_alive(),
                "thread_name": self._thread.name,
                "thread_ident": self._thread.ident,
                "loop_running": loop_running,
                "task_done": task.done() if task else None,
                "task_cancelled": task.cancelled() if task else None,
            }

        def schedule_loop_ping(self) -> bool:
            def _record_ping() -> None:
                loop_ping_state.ping()

            loop = self._loop
            if loop is None:
                return False
            try:
                loop.call_soon_threadsafe(_record_ping)
                return True
            except RuntimeError:
                return False

    watchdog_threshold = _env_float(
        "SELF_IMPROVEMENT_CYCLE_WATCHDOG_SECONDS", max(120.0, interval * 3)
    )
    watchdog_restart = _env_bool("SELF_IMPROVEMENT_CYCLE_WATCHDOG_RESTART", True)
    allow_zombie_restart = _env_bool("SELF_IMPROVEMENT_ALLOW_ZOMBIE_RESTART", False)
    watchdog_interval = max(5.0, min(30.0, watchdog_threshold / 2.0))
    stop_timeout = _env_float("SELF_IMPROVEMENT_CYCLE_STOP_TIMEOUT_SECONDS", 5.0)
    join_timeout = _env_float(
        "SELF_IMPROVEMENT_CYCLE_JOIN_TIMEOUT_SECONDS", max(1.0, stop_timeout)
    )

    monitor_state: dict[str, Any] = {
        "thread": None,
        "stop_event": None,
        "last_loop_heartbeat": None,
        "loop_heartbeat_started": None,
        "stuck_thread": None,
    }

    def _get_thread_stack(thread_ident: int | None) -> str | None:
        if thread_ident is None:
            return None
        frame_map = sys._current_frames()
        frame = frame_map.get(thread_ident)
        if frame is None:
            return None
        return "".join(traceback.format_stack(frame))

    def _create_cycle_thread() -> tuple[_CycleThread, threading.Event]:
        local_stop_event = threading.Event()
        thread = _CycleThread(local_stop_event)
        return thread, local_stop_event

    def _restart_cycle_thread(
        logger: logging.Logger, reason: str, *, wait_on_stop: bool = True
    ) -> None:
        stuck_thread = monitor_state.get("stuck_thread")
        if stuck_thread is not None:
            stuck_worker = stuck_thread.get("thread")
            if stuck_worker is not None and stuck_worker.is_alive():
                if not allow_zombie_restart:
                    logger.critical(
                        "refusing to restart self improvement cycle while stuck thread is alive",
                        extra=log_record(
                            reason=reason,
                            thread_ident=stuck_thread.get("ident"),
                            stuck_detected_at=stuck_thread.get("detected_at"),
                            stuck_thread_stack=stuck_thread.get("stack"),
                            override_env="SELF_IMPROVEMENT_ALLOW_ZOMBIE_RESTART",
                        ),
                    )
                    return
                logger.warning(
                    "overriding stuck thread guard; restarting self improvement cycle while stuck thread is alive",
                    extra=log_record(
                        reason=reason,
                        thread_ident=stuck_thread.get("ident"),
                        stuck_detected_at=stuck_thread.get("detected_at"),
                        stuck_thread_stack=stuck_thread.get("stack"),
                        override_env="SELF_IMPROVEMENT_ALLOW_ZOMBIE_RESTART",
                    ),
                )
        current_thread = monitor_state.get("thread")
        if current_thread is not None:
            try:
                effective_timeout = stop_timeout if wait_on_stop else 0.0
                current_thread.stop(timeout=effective_timeout)
            except Exception:
                logger.exception(
                    "cycle stop failed during restart",
                    extra=log_record(reason=reason),
                )
            thread_state = current_thread.state()
            if thread_state.get("thread_alive"):
                stack_dump = _get_thread_stack(thread_state.get("thread_ident"))
                monitor_state["stuck_thread"] = {
                    "thread": current_thread._thread,
                    "ident": thread_state.get("thread_ident"),
                    "detected_at": time.time(),
                    "stack": stack_dump,
                }
                stuck_message = "self improvement cycle thread stuck during restart"
                if stack_dump:
                    stuck_message = f"{stuck_message}\n{stack_dump}"
                logger.warning(
                    stuck_message,
                    extra=log_record(
                        reason=reason,
                        stop_timeout_seconds=stop_timeout,
                        thread_state=thread_state,
                        stuck_thread_stack=stack_dump,
                    ),
                )
                if not allow_zombie_restart:
                    logger.critical(
                        "refusing to start replacement thread while stuck thread is alive",
                        extra=log_record(
                            reason=reason,
                            thread_ident=thread_state.get("thread_ident"),
                            stop_timeout_seconds=stop_timeout,
                            override_env="SELF_IMPROVEMENT_ALLOW_ZOMBIE_RESTART",
                            stuck_thread_stack=stack_dump,
                        ),
                    )
                    return
        tick_state.reset()
        loop_heartbeat_state.reset()
        loop_ping_state.reset()
        monitor_state["last_loop_heartbeat"] = None
        monitor_state["loop_heartbeat_started"] = None
        monitor_state["stuck_thread"] = None
        new_thread, new_stop_event = _create_cycle_thread()
        monitor_state["thread"] = new_thread
        monitor_state["stop_event"] = new_stop_event
        global _cycle_thread, _stop_event
        _cycle_thread = new_thread
        _stop_event = new_stop_event
        new_thread.start()

    _ = _init.settings
    logger_fn = globals().get("get_logger")
    log_record_fn = globals().get("log_record")
    logger = logger_fn(__name__) if logger_fn else None
    print("💡 SI-13: initialising ROI results database")
    try:
        ROIResultsDB()
    except (OSError, RuntimeError) as exc:
        if logger is not None:
            logger.error(
                "ROIResultsDB initialisation failed",
                extra=(log_record_fn(module=__name__) if log_record_fn else None),
                exc_info=exc,
            )
        raise RuntimeError("ROIResultsDB initialisation failed") from exc
    print("💡 SI-14: initialising workflow stability database")
    try:
        WorkflowStabilityDB()
    except (OSError, RuntimeError) as exc:
        if logger is not None:
            logger.error(
                "WorkflowStabilityDB initialisation failed",
                extra=(log_record_fn(module=__name__) if log_record_fn else None),
                exc_info=exc,
            )
        raise RuntimeError("WorkflowStabilityDB initialisation failed") from exc

    thread, stop_event = _create_cycle_thread()
    monitor_state["thread"] = thread
    monitor_state["stop_event"] = stop_event
    global _cycle_thread, _stop_event, _cycle_watchdog_stop
    _cycle_thread = thread
    _stop_event = stop_event
    _cycle_watchdog_stop = threading.Event()

    def _watchdog() -> None:
        watchdog_logger = get_logger("SelfImprovementCycleWatchdog")
        while True:
            if _cycle_watchdog_stop is not None and _cycle_watchdog_stop.is_set():
                break
            current_stop = monitor_state.get("stop_event")
            if current_stop is not None and current_stop.is_set():
                break
            time.sleep(watchdog_interval)
            if _cycle_watchdog_stop is not None and _cycle_watchdog_stop.is_set():
                break
            current_thread = monitor_state.get("thread")
            if current_thread is None:
                continue
            loop_ping_scheduled = current_thread.schedule_loop_ping()
            snapshot = tick_state.snapshot()
            heartbeat_snapshot = loop_heartbeat_state.snapshot()
            ping_snapshot = loop_ping_state.snapshot()
            now = time.time()
            monotonic_now = time.monotonic()
            last_tick = snapshot.get("last_tick")
            started_at = snapshot.get("started_at", now)
            last_tick_age = (now - last_tick) if last_tick is not None else None
            idle_since_start = now - started_at
            last_loop_heartbeat = monitor_state.get("last_loop_heartbeat")
            loop_heartbeat_started = monitor_state.get("loop_heartbeat_started")
            last_loop_heartbeat_age = (
                (monotonic_now - last_loop_heartbeat)
                if last_loop_heartbeat is not None
                else None
            )
            loop_heartbeat_idle_since_start = (
                (monotonic_now - loop_heartbeat_started)
                if loop_heartbeat_started is not None
                else None
            )
            last_heartbeat = heartbeat_snapshot.get("last_heartbeat")
            heartbeat_started_at = heartbeat_snapshot.get("started_at", now)
            last_heartbeat_age = (
                (now - last_heartbeat) if last_heartbeat is not None else None
            )
            heartbeat_idle_since_start = now - heartbeat_started_at
            heartbeat_stalled = (
                last_heartbeat_age is not None
                and last_heartbeat_age > watchdog_threshold
            ) or (
                last_heartbeat is None
                and heartbeat_idle_since_start > watchdog_threshold
            )
            loop_heartbeat_stalled = (
                last_loop_heartbeat_age is not None
                and last_loop_heartbeat_age > watchdog_threshold
            ) or (
                last_loop_heartbeat is None
                and loop_heartbeat_idle_since_start is not None
                and loop_heartbeat_idle_since_start > watchdog_threshold
            )
            last_ping = ping_snapshot.get("last_ping")
            ping_started_at = ping_snapshot.get("started_at", now)
            last_ping_age = (now - last_ping) if last_ping is not None else None
            ping_idle_since_start = now - ping_started_at
            ping_stalled = (
                not loop_ping_scheduled
                or (last_ping_age is not None and last_ping_age > watchdog_threshold)
                or (last_ping is None and ping_idle_since_start > watchdog_threshold)
            )
            stalled = (
                last_tick_age is not None and last_tick_age > watchdog_threshold
            ) or (last_tick is None and idle_since_start > watchdog_threshold)
            if (
                not stalled
                and not heartbeat_stalled
                and not ping_stalled
                and not loop_heartbeat_stalled
            ):
                continue
            thread_state = current_thread.state()
            tick_thread_ident = snapshot.get("last_tick_thread_ident")
            heartbeat_thread_ident = heartbeat_snapshot.get("last_heartbeat_thread_ident")
            ping_thread_ident = ping_snapshot.get("last_ping_thread_ident")
            stall_thread_ident = (
                ping_thread_ident
                or heartbeat_thread_ident
                or tick_thread_ident
                or thread_state.get("thread_ident")
            )
            stack_dump = None
            if stall_thread_ident is not None:
                frame_map = sys._current_frames()
                frame = frame_map.get(stall_thread_ident)
                if frame is not None:
                    stack_dump = "".join(traceback.format_stack(frame))
            stall_message = "self improvement cycle stalled"
            if stack_dump:
                stall_message = f"{stall_message}\n{stack_dump}"
            log_payload = log_record(
                last_tick_timestamp=last_tick,
                last_tick_age_seconds=last_tick_age,
                tick_count=snapshot.get("tick_count"),
                tick_thread_name=snapshot.get("last_tick_thread_name"),
                tick_thread_ident=snapshot.get("last_tick_thread_ident"),
                loop_heartbeat_timestamp=last_heartbeat,
                loop_heartbeat_age_seconds=last_heartbeat_age,
                loop_heartbeat_count=heartbeat_snapshot.get("heartbeat_count"),
                loop_heartbeat_thread_name=heartbeat_snapshot.get(
                    "last_heartbeat_thread_name"
                ),
                loop_heartbeat_thread_ident=heartbeat_thread_ident,
                last_loop_heartbeat=last_loop_heartbeat,
                last_loop_heartbeat_age_seconds=last_loop_heartbeat_age,
                loop_running=thread_state.get("loop_running"),
                loop_ping_timestamp=last_ping,
                loop_ping_age_seconds=last_ping_age,
                loop_ping_count=ping_snapshot.get("ping_count"),
                loop_ping_thread_name=ping_snapshot.get("last_ping_thread_name"),
                loop_ping_thread_ident=ping_thread_ident,
                watchdog_threshold_seconds=watchdog_threshold,
                thread_state=thread_state,
                stalled_due_to_heartbeat=heartbeat_stalled,
                stalled_due_to_loop_heartbeat=loop_heartbeat_stalled,
                stalled_due_to_loop_ping=ping_stalled,
                event_loop_unresponsive_seconds=(
                    last_ping_age if last_ping_age is not None else ping_idle_since_start
                )
                if ping_stalled
                else None,
                stall_stack_trace=stack_dump,
            )
            if ping_stalled:
                watchdog_logger.critical(stall_message, extra=log_payload)
            elif heartbeat_stalled:
                watchdog_logger.critical(stall_message, extra=log_payload)
            else:
                watchdog_logger.warning(stall_message, extra=log_payload)
            if watchdog_restart:
                restart_reason = (
                    "watchdog_restart_loop_ping"
                    if ping_stalled
                    else "watchdog_restart_loop_heartbeat"
                    if loop_heartbeat_stalled
                    else "watchdog_restart_heartbeat"
                    if heartbeat_stalled
                    else "watchdog_restart"
                )
                watchdog_logger.warning(
                    "restarting self improvement cycle after stall; forcing process exit if stop timeout elapses",
                    extra=log_record(
                        reason=restart_reason,
                        last_tick_timestamp=last_tick,
                        last_tick_age_seconds=last_tick_age,
                        thread_state=thread_state,
                        stop_timeout_seconds=stop_timeout,
                        last_resort_recovery="process_exit",
                    ),
                )
                _restart_cycle_thread(
                    watchdog_logger,
                    restart_reason,
                    wait_on_stop=not (heartbeat_stalled or loop_heartbeat_stalled),
                )

    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    watchdog_thread.start()
    return thread


def stop_self_improvement_cycle() -> None:
    """Signal the background self improvement cycle to stop and wait for it."""
    global _cycle_thread, _stop_event, _cycle_watchdog_stop
    if _cycle_watchdog_stop is not None:
        _cycle_watchdog_stop.set()
    if _cycle_thread is None:
        return
    if _stop_event is not None:
        _stop_event.set()
    _cycle_thread.stop()
    _cycle_thread = None
    _stop_event = None


__all__ = [
    "self_improvement_cycle",
    "start_self_improvement_cycle",
    "stop_self_improvement_cycle",
    "reload_settings",
    "resolve_meta_workflow_planner",
    "record_workflow_iteration",
    "workflow_controller_status",
    "PLANNER_INTERVAL",
    "MUTATION_RATE",
    "ROI_WEIGHT",
    "DOMAIN_PENALTY",
    "ENTROPY_THRESHOLD",
    "SEARCH_DEPTH",
    "BEAM_WIDTH",
    "ENTROPY_WEIGHT",
]
