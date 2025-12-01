"""Bootstrap helpers for seeding shared self-coding context.

The utilities here build the same pipeline/manager setup used by the runtime
bootstrapping helpers and then expose that state to lazy modules so they can
skip re-entrant ``prepare_pipeline_for_bootstrap`` calls.

Timeouts are sourced from ``coding_bot_interface._resolve_bootstrap_wait_timeout``
when available so they respect ``MENACE_BOOTSTRAP_WAIT_SECS`` and
``MENACE_BOOTSTRAP_VECTOR_WAIT_SECS`` while retaining a generous 720s/900s
fallback derived from ``_get_bootstrap_wait_timeout`` (or
``BOOTSTRAP_STEP_TIMEOUT``) unless timeouts are explicitly disabled. Operators
who hit the legacy 30s cap on ``prepare_pipeline_for_bootstrap`` should set
``MENACE_BOOTSTRAP_WAIT_SECS`` (standard paths) and
``MENACE_BOOTSTRAP_VECTOR_WAIT_SECS`` (vector-heavy pipelines) to the enforced
720s/900s floors before running bootstrap, especially on slow disks or when
vector DB migrations need extra breathing room.
"""

from __future__ import annotations

import contextlib
import io
import inspect
import logging
import os
import sys
import threading
import time
import traceback
import faulthandler
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Mapping

from menace_sandbox import coding_bot_interface as _coding_bot_interface
from menace_sandbox.coding_bot_interface import (
    _bootstrap_dependency_broker,
    _pop_bootstrap_context,
    _push_bootstrap_context,
    advertise_bootstrap_placeholder,
    fallback_helper_manager,
    prepare_pipeline_for_bootstrap,
)

_BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL = advertise_bootstrap_placeholder(
    dependency_broker=_bootstrap_dependency_broker()
)

from lock_utils import LOCK_TIMEOUT, SandboxLock
from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.code_database import CodeDB
from menace_sandbox.context_builder_util import create_context_builder
from menace_sandbox.db_router import set_audit_bootstrap_safe_default
from menace_sandbox.data_bot import DataBot, persist_sc_thresholds
from menace_sandbox.menace_memory_manager import MenaceMemoryManager
from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
from menace_sandbox.self_coding_engine import SelfCodingEngine
from menace_sandbox.self_coding_manager import SelfCodingManager, internalize_coding_bot
from menace_sandbox.self_coding_thresholds import get_thresholds
from menace_sandbox.threshold_service import ThresholdService
from bootstrap_readiness import (
    _COMPONENT_BASELINES,
    READINESS_STAGES,
    build_stage_deadlines,
    lagging_optional_components,
    minimal_online,
    stage_for_step,
)
from bootstrap_metrics import (
    calibrate_overall_timeout,
    calibrate_step_budgets,
    compute_stats,
    load_duration_store,
)
from safe_repr import summarise_value
from security.secret_redactor import redact_dict
from bootstrap_timeout_policy import (
    SharedTimeoutCoordinator,
    _host_load_average,
    _host_load_scale,
    build_progress_signal_hook,
    collect_timeout_telemetry,
    enforce_bootstrap_timeout_policy,
    get_bootstrap_guard_context,
    load_component_timeout_floors,
    read_bootstrap_heartbeat,
    compute_prepare_pipeline_component_budgets,
    render_prepare_pipeline_timeout_hints,
    record_component_budget_violation,
    _PREPARE_PIPELINE_COMPONENT,
)

LOGGER = logging.getLogger(__name__)

_BOOTSTRAP_CACHE: Dict[str, Dict[str, Any]] = {}
_BOOTSTRAP_CACHE_LOCK = threading.Lock()
BOOTSTRAP_PROGRESS: Dict[str, str] = {"last_step": "not-started"}
BOOTSTRAP_ONLINE_STATE: Dict[str, Any] = {"quorum": False, "components": {}}
BOOTSTRAP_STEP_TIMELINE: list[tuple[str, float]] = []
_BOOTSTRAP_TIMELINE_START: float | None = None
_BOOTSTRAP_TIMELINE_LOCK = threading.Lock()
_STEP_START_OBSERVER: Callable[[str], None] | None = None
_STEP_END_OBSERVER: Callable[[str, float], None] | None = None
_BOOTSTRAP_TIMEOUT_FLOOR = getattr(_coding_bot_interface, "_BOOTSTRAP_TIMEOUT_FLOOR", 720.0)
_PREPARE_STANDARD_TIMEOUT_FLOOR = 720.0
_PREPARE_VECTOR_TIMEOUT_FLOOR = 900.0
_PREPARE_SAFE_TIMEOUT_FLOOR = _PREPARE_STANDARD_TIMEOUT_FLOOR
_VECTOR_ENV_MINIMUM = _PREPARE_VECTOR_TIMEOUT_FLOOR
_BOOTSTRAP_LOCK_PATH_ENV = "MENACE_BOOTSTRAP_LOCK_PATH"


class _BootstrapStepScheduler:
    """Track bootstrap readiness and compute adaptive step budgets."""

    _COMPONENTS = (
        "vector_seeding",
        "retriever_hydration",
        "db_index_load",
        "orchestrator_state",
        "background_loops",
    )

    def __init__(self) -> None:
        self._step_history: dict[str, list[float]] = {}
        self._component_state: dict[str, str] = {
            name: "pending" for name in self._COMPONENTS
        }
        self._latest_deadlines: dict[str, float] = {}
        self._component_budgets: dict[str, float] = {}
        self._component_ready_at: dict[str, float] = {}
        self._default_budget = _BOOTSTRAP_TIMEOUT_FLOOR
        self._default_vector_budget = _PREPARE_VECTOR_TIMEOUT_FLOOR

    def set_component_budgets(self, budgets: Mapping[str, float] | None) -> None:
        mapped: dict[str, float] = {}

        alias_map = {
            "vectorizers": "vector_seeding",
            "retrievers": "retriever_hydration",
            "db_indexes": "db_index_load",
            "orchestrator_state": "orchestrator_state",
            "background_loops": "background_loops",
            "pipeline_config": "pipeline_config",
        }

        for key, value in (budgets or {}).items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            mapped[alias_map.get(key, key)] = max(parsed, 0.0)

        self._component_budgets = mapped
        for component in mapped:
            self._component_state.setdefault(component, "pending")
        self._default_budget = max(
            _BOOTSTRAP_TIMEOUT_FLOOR,
            mapped.get("pipeline_config", _BOOTSTRAP_TIMEOUT_FLOOR),
            max(mapped.values(), default=_BOOTSTRAP_TIMEOUT_FLOOR),
        )
        self._default_vector_budget = max(
            _PREPARE_VECTOR_TIMEOUT_FLOOR,
            mapped.get("vector_seeding", _PREPARE_VECTOR_TIMEOUT_FLOOR),
        )

    def default_budget(self, *, vector_heavy: bool = False) -> float:
        return self._default_vector_budget if vector_heavy else self._default_budget

    def _component_for_step(self, step_name: str, *, vector_heavy: bool) -> str | None:
        normalized = step_name.lower()
        if "vector" in normalized:
            return "vector_seeding"
        if "retriev" in normalized:
            return "retriever_hydration"
        if "index" in normalized or "db" in normalized:
            return "db_index_load"
        if "orchestrator" in normalized:
            return "orchestrator_state"
        if "background" in normalized:
            return "background_loops"
        if "prepare" in normalized:
            return "pipeline_config"
        return None

    def _local_load_scale(self, load_average: float | None = None) -> float:
        if load_average is None:
            try:
                load_average = _host_load_average()
            except Exception:
                load_average = None

        if load_average is None:
            try:
                load1, _, _ = os.getloadavg()
                cpus = os.cpu_count() or 1
                load_average = load1 / max(cpus, 1)
            except OSError:
                load_average = 0.0

        # Keep a modest envelope: heavy hosts get extra slack, idle hosts tighten.
        return min(1.6, max(0.8, 1.0 + (float(load_average) - 1.0) * 0.35))

    def _telemetry_scale(self, *, vector_heavy: bool) -> tuple[float, dict[str, Any]]:
        heartbeat = read_bootstrap_heartbeat()
        telemetry: dict[str, Any] = {"vector_heavy": vector_heavy, "heartbeat": heartbeat}
        host_load = None
        active_bootstraps = 0

        if isinstance(heartbeat, Mapping):
            try:
                host_load = float(heartbeat.get("host_load"))
            except (TypeError, ValueError):
                host_load = None
            try:
                active_bootstraps = max(int(heartbeat.get("active_bootstraps", 0) or 0), 0)
            except (TypeError, ValueError):
                active_bootstraps = 0

        peer_scale = _host_load_scale(host_load)
        if active_bootstraps:
            peer_scale *= 1.0 + min(max(active_bootstraps - 1, 0), 4) * 0.08

        local_scale = self._local_load_scale(host_load)
        telemetry.update(
            {
                "host_load": host_load,
                "active_bootstraps": active_bootstraps,
                "peer_scale": peer_scale,
                "local_scale": local_scale,
            }
        )

        scale = peer_scale
        if local_scale < 1.0:
            scale *= local_scale
        else:
            scale *= max(local_scale, 1.0)

        telemetry["scale"] = scale
        return scale, telemetry

    def _step_baseline(self, step_name: str, vector_heavy: bool, *, telemetry_scale: float = 1.0) -> float:
        component = self._component_for_step(step_name, vector_heavy=vector_heavy)
        if component:
            budget = self._component_budgets.get(component)
            if budget is not None:
                floor = _PREPARE_VECTOR_TIMEOUT_FLOOR if vector_heavy else _PREPARE_STANDARD_TIMEOUT_FLOOR
                return max(budget, floor) * telemetry_scale

        baselines = {
            "prepare_pipeline_for_bootstrap": self.default_budget(vector_heavy=vector_heavy),
            "_seed_research_aggregator_context": self.default_budget(vector_heavy=True),
            "_push_bootstrap_context": self.default_budget(vector_heavy=False),
            "promote_pipeline": self.default_budget(vector_heavy=vector_heavy),
        }
        baseline = baselines.get(step_name, self.default_budget(vector_heavy=vector_heavy))
        return baseline * telemetry_scale

    def _historical_mean(self, step_name: str) -> float | None:
        history = self._step_history.get(step_name)
        if not history:
            return None
        return sum(history) / len(history)

    def record_history(self, step_name: str, duration: float) -> None:
        window = self._step_history.setdefault(step_name, [])
        window.append(duration)
        if len(window) > 25:
            del window[0 : len(window) - 25]

    def allocate_timeout(
        self,
        step_name: str,
        base_timeout: float | None,
        *,
        vector_heavy: bool = False,
        contention_scale: float = 1.0,
    ) -> float | None:
        if base_timeout is None:
            base_timeout = _PREPARE_VECTOR_TIMEOUT_FLOOR if vector_heavy else _PREPARE_STANDARD_TIMEOUT_FLOOR

        historical = self._historical_mean(step_name)
        load_scale, telemetry = self._telemetry_scale(vector_heavy=vector_heavy)
        baseline = max(
            base_timeout * 0.75,
            self._step_baseline(step_name, vector_heavy, telemetry_scale=telemetry.get("local_scale", 1.0)),
        )
        predicted = max(baseline, historical if historical is not None else 0.0)
        candidate = predicted * load_scale + BOOTSTRAP_DEADLINE_BUFFER
        min_floor = max(
            self.default_budget(vector_heavy=vector_heavy),
            _PREPARE_VECTOR_TIMEOUT_FLOOR if vector_heavy else _PREPARE_STANDARD_TIMEOUT_FLOOR,
        )
        upper_bound = base_timeout * (1.6 if vector_heavy else 1.35)

        if candidate < baseline:
            candidate = min(base_timeout, max(baseline, min_floor))
        else:
            candidate = min(max(candidate, min_floor), upper_bound)

        if abs(load_scale - 1.0) >= 0.05:
            LOGGER.info(
                "adaptive bootstrap step budget scaled",
                extra={
                    "step": step_name,
                    "scale": round(load_scale, 3),
                    "vector_heavy": vector_heavy,
                    "telemetry": telemetry,
                    "predicted": predicted,
                    "baseline": baseline,
                    "upper_bound": upper_bound,
                },
            )

        if contention_scale != 1.0:
            candidate *= contention_scale
            upper_bound *= contention_scale
            candidate = min(max(candidate, min_floor), upper_bound)

        self._latest_deadlines[step_name] = candidate
        return candidate

    def mark_partial(self, component: str, *, reason: str | None = None) -> None:
        if component not in self._component_state:
            return
        self._component_state[component] = "partial"
        if reason:
            LOGGER.debug("component marked partial", extra={"component": component, "reason": reason})

    def mark_ready(self, component: str, *, reason: str | None = None) -> None:
        if component not in self._component_state:
            return
        self._component_state[component] = "ready"
        self._component_ready_at[component] = time.time()
        if reason:
            LOGGER.debug("component marked ready", extra={"component": component, "reason": reason})

    def quorum_met(self) -> bool:
        tracked_components = [name for name in self._component_state if name != "pipeline_config"]
        ready_count = sum(
            1 for name in tracked_components if self._component_state.get(name) == "ready"
        )
        partial_count = sum(
            1 for name in tracked_components if self._component_state.get(name) == "partial"
        )
        quorum = (len(tracked_components) + 1) // 2 + 1
        return ready_count >= quorum or (
            ready_count + partial_count >= quorum and ready_count >= max(quorum - 1, 1)
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "components": dict(self._component_state),
            "quorum": self.quorum_met(),
            "deadlines": dict(self._latest_deadlines),
            "component_budgets": dict(self._component_budgets),
            "component_ready_at": dict(self._component_ready_at),
            "online": self.quorum_met(),
        }


_BOOTSTRAP_SCHEDULER = _BootstrapStepScheduler()


class _StagedBootstrapController:
    """Coordinate readiness gates across bootstrap stages."""

    def __init__(
        self,
        *,
        stage_policy: Mapping[str, Any] | None,
        coordinator: SharedTimeoutCoordinator | None,
        signal_hook: Callable[[Mapping[str, object]], None] | None,
    ) -> None:
        self._remaining_steps: dict[str, set[str]] = {
            stage.name: set(stage.steps) for stage in READINESS_STAGES
        }
        self._optional: dict[str, bool] = {
            stage.name: stage.optional for stage in READINESS_STAGES
        }
        self._stage_policy = stage_policy or {}
        self._coordinator = coordinator
        self._signal_hook = signal_hook
        self._stage_windows: Mapping[str, Mapping[str, float | None]] = {}
        self._start_component_windows()

    def _start_component_windows(self) -> None:
        if not self._coordinator or not self._stage_policy:
            return
        budgets: dict[str, float] = {}
        for stage, entry in self._stage_policy.items():
            budget = None
            try:
                budget = entry.get("deadline") if isinstance(entry, Mapping) else None
                if budget is None and isinstance(entry, Mapping):
                    budget = entry.get("scaled_budget") or entry.get("budget")
                if budget is not None:
                    budgets[stage] = float(budget)
            except (TypeError, ValueError):
                continue
        if budgets:
            self._stage_windows = self._coordinator.start_component_timers(
                budgets, minimum=0.0
            )
            if self._signal_hook:
                try:
                    self._signal_hook(
                        {
                            "event": "bootstrap-stage-windows",
                            "windows": self._stage_windows,
                        }
                    )
                except Exception:  # pragma: no cover - advisory only
                    LOGGER.debug("stage window signal failed", exc_info=True)

    def _emit_stage_signal(self, stage: str, step: str, state: str) -> None:
        if not self._signal_hook:
            return
        payload = {
            "event": "bootstrap-stage-progress",
            "stage": stage,
            "step": step,
            "state": state,
            "optional": self._optional.get(stage, False),
            "remaining_steps": sorted(self._remaining_steps.get(stage, set())),
        }
        try:
            self._signal_hook(payload)
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("bootstrap stage signal failed", exc_info=True)

    def start_step(self, step_name: str) -> None:
        stage = stage_for_step(step_name)
        if not stage:
            return
        _BOOTSTRAP_SCHEDULER.mark_partial(stage, reason=f"{step_name}_start")
        _set_component_state(stage, "warming")
        if self._coordinator:
            self._coordinator.mark_component_state(stage, "running")
        self._emit_stage_signal(stage, step_name, "running")

    def complete_step(self, step_name: str, elapsed: float) -> None:
        stage = stage_for_step(step_name)
        if not stage:
            return
        remaining = self._remaining_steps.get(stage)
        if remaining is not None:
            remaining.discard(step_name)
        ready = remaining is not None and len(remaining) == 0
        if ready:
            _BOOTSTRAP_SCHEDULER.mark_ready(stage, reason=f"{step_name}_complete")
            _set_component_state(stage, "ready")
        else:
            _BOOTSTRAP_SCHEDULER.mark_partial(stage, reason=f"{step_name}_complete")
            _set_component_state(stage, "warming")

        if self._coordinator:
            self._coordinator.mark_component_state(stage, "ready" if ready else "running")

        self._emit_stage_signal(stage, step_name, "ready" if ready else "running")



def _default_step_floor(*, vector_heavy: bool = False) -> float:
    return _BOOTSTRAP_SCHEDULER.default_budget(vector_heavy=vector_heavy)


def _set_component_state(component: str, state: str) -> None:
    BOOTSTRAP_ONLINE_STATE.setdefault("components", {})[component] = state
    BOOTSTRAP_ONLINE_STATE["warming"] = [
        name for name, status in BOOTSTRAP_ONLINE_STATE["components"].items() if status != "ready"
    ]
    _publish_online_state()


@dataclass
class _BootstrapTask:
    """Represents a bootstrap action within a dependency DAG."""

    name: str
    fn: Callable[[dict[str, Any]], Any]
    requires: tuple[str, ...] = ()
    critical: bool = True
    background: bool = False
    component: str | None = None
    description: str | None = None


class _BootstrapDagRunner:
    """Simple DAG scheduler that allows background warming while gating critical steps."""

    def __init__(self, *, logger: logging.Logger) -> None:
        self.logger = logger
        self._tasks: dict[str, _BootstrapTask] = {}
        self._results: dict[str, Any] = {}
        self._threads: list[threading.Thread] = []
        self._component_state: dict[str, str] = {}

    def add(self, task: _BootstrapTask) -> None:
        if task.name in self._tasks:
            raise ValueError(f"duplicate bootstrap task {task.name}")
        self._tasks[task.name] = task
        if task.component:
            self._component_state.setdefault(task.component, "pending")

    def _mark_component(self, component: str, state: str) -> None:
        previous = self._component_state.get(component)
        self._component_state[component] = state
        _set_component_state(component, state)
        if previous != state:
            self.logger.info("component state changed", extra={"component": component, "state": state})
        _BOOTSTRAP_SCHEDULER.mark_ready(component) if state == "ready" else _BOOTSTRAP_SCHEDULER.mark_partial(component)

    def _execute_task(self, task: _BootstrapTask) -> None:
        try:
            if task.component:
                self._mark_component(task.component, "warming")
            self._results[task.name] = task.fn(self._results)
            if task.component:
                self._mark_component(task.component, "ready")
        except Exception:
            if task.component:
                self._mark_component(task.component, "partial")
            self.logger.exception("bootstrap task failed", extra={"task": task.name})
            if task.critical:
                raise

    def _start_background(self, task: _BootstrapTask) -> None:
        thread = threading.Thread(target=self._execute_task, args=(task,), daemon=True)
        thread.start()
        self._threads.append(thread)

    def run(self, *, critical_gate: Iterable[str]) -> dict[str, Any]:
        pending = dict(self._tasks)
        completed: set[str] = set()
        critical_remaining = set(critical_gate)

        while pending:
            progress_made = False
            for name, task in list(pending.items()):
                if not set(task.requires).issubset(completed):
                    continue
                progress_made = True
                pending.pop(name)
                if task.background and task.name not in critical_remaining:
                    self.logger.info("launching background bootstrap task", extra={"task": name})
                    self._start_background(task)
                    completed.add(name)
                    continue
                self._execute_task(task)
                completed.add(name)
                critical_remaining.discard(name)
            if not progress_made:
                raise RuntimeError(f"unable to resolve bootstrap DAG; remaining tasks: {sorted(pending)}")
            if not critical_remaining:
                break

        _publish_online_state()
        return self._results


class _BootstrapContentionCoordinator:
    """Coordinate bootstrap phases when another host bootstrap is in-flight."""

    def __init__(self, lock_path: str) -> None:
        self._lock_path = lock_path

    def _lock_occupied(self) -> bool:
        if not self._lock_path or not os.path.exists(self._lock_path):
            return False
        try:
            lock = SandboxLock(self._lock_path)
            return not lock.is_lock_stale(timeout=LOCK_TIMEOUT)
        except Exception:  # pragma: no cover - contention hints are best effort
            LOGGER.debug("unable to inspect bootstrap lock for contention", exc_info=True)
            return True

    def negotiate_step(self, step_name: str, *, vector_heavy: bool = False, heavy: bool = False) -> dict[str, Any]:
        contention_active = self._lock_occupied()
        gate: dict[str, Any] = {
            "contention": contention_active,
            "delay": 0.0,
            "timeout_scale": 1.0,
            "parallelism_scale": 1.0,
            "step": step_name,
            "vector_heavy": vector_heavy,
            "heavy": heavy,
        }

        if not contention_active:
            return gate

        delay = 3.0 if (vector_heavy or heavy) else 1.5
        timeout_scale = 1.2 if (vector_heavy or heavy) else 1.1
        parallelism_scale = 0.6 if (vector_heavy or heavy) else 0.75
        gate.update(
            {
                "delay": delay,
                "timeout_scale": timeout_scale,
                "parallelism_scale": parallelism_scale,
            }
        )
        LOGGER.info(
            "contention detected; staggering bootstrap step",
            extra={"gate": gate, "lock_path": self._lock_path},
        )
        time.sleep(delay)
        return gate


def _publish_online_state() -> None:
    snapshot = _BOOTSTRAP_SCHEDULER.snapshot()
    BOOTSTRAP_ONLINE_STATE.update(snapshot)
    core_ready, lagging_core, degraded_core, degraded_online = minimal_online(snapshot)
    BOOTSTRAP_ONLINE_STATE.update(
        {
            "core_ready": core_ready,
            "core_lagging": sorted(lagging_core),
            "core_degraded": sorted(degraded_core),
            "core_degraded_online": degraded_online,
            "quorum": snapshot.get("quorum", False) or core_ready,
            "online": snapshot.get("online", False) or core_ready,
        }
    )
    optional_warming = lagging_optional_components(snapshot)
    BOOTSTRAP_ONLINE_STATE["optional_warming"] = sorted(optional_warming)
    BOOTSTRAP_ONLINE_STATE["partial_online"] = bool(optional_warming)
    if optional_warming:
        BOOTSTRAP_ONLINE_STATE["optional_degraded"] = sorted(optional_warming)


def _clamp_timeout_floor(timeout: float, *, env_var: str) -> float:
    if timeout < _BOOTSTRAP_TIMEOUT_FLOOR:
        LOGGER.warning(
            "%s below minimum; clamping to %ss",
            env_var,
            _BOOTSTRAP_TIMEOUT_FLOOR,
            extra={
                "requested_timeout": timeout,
                "timeout_floor": _BOOTSTRAP_TIMEOUT_FLOOR,
                "effective_timeout": _BOOTSTRAP_TIMEOUT_FLOOR,
            },
        )
        return _BOOTSTRAP_TIMEOUT_FLOOR
    return timeout


def _apply_timeout_policy_snapshot(policy_snapshot: dict[str, dict[str, Any]]) -> None:
    """Re-apply default timeout resolution after policy enforcement."""

    global _DEFAULT_BOOTSTRAP_STEP_TIMEOUT
    global _DEFAULT_VECTOR_BOOTSTRAP_STEP_TIMEOUT
    global BOOTSTRAP_STEP_TIMEOUT

    def _resolved_value(env_var: str, current: float) -> float:
        entry = policy_snapshot.get(env_var, {})
        value = entry.get("effective")
        return float(value) if isinstance(value, (int, float)) else current

    _DEFAULT_BOOTSTRAP_STEP_TIMEOUT = max(
        _default_step_floor(vector_heavy=False),
        _clamp_timeout_floor(
            _resolved_value("BOOTSTRAP_STEP_TIMEOUT", _DEFAULT_BOOTSTRAP_STEP_TIMEOUT),
            env_var="BOOTSTRAP_STEP_TIMEOUT",
        ),
    )
    _DEFAULT_VECTOR_BOOTSTRAP_STEP_TIMEOUT = max(
        _default_step_floor(vector_heavy=True),
        _clamp_timeout_floor(
            _resolved_value(
                "BOOTSTRAP_VECTOR_STEP_TIMEOUT", _DEFAULT_VECTOR_BOOTSTRAP_STEP_TIMEOUT
            ),
            env_var="BOOTSTRAP_VECTOR_STEP_TIMEOUT",
        ),
    )
    BOOTSTRAP_STEP_TIMEOUT = _resolve_step_timeout(step_name="default")


def _hydrate_vector_bootstrap_env(minimum: float = _VECTOR_ENV_MINIMUM) -> dict[str, float]:
    """Ensure vector-heavy bootstrap env vars exist at or above ``minimum``."""

    resolved: dict[str, float] = {}
    for env_var in (
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS",
        "BOOTSTRAP_VECTOR_STEP_TIMEOUT",
    ):
        raw_value = os.getenv(env_var)
        effective = minimum
        update_env = False

        if raw_value is None:
            update_env = True
        else:
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                update_env = True
            else:
                effective = parsed
                if parsed < minimum:
                    effective = minimum
                    update_env = True

        if update_env:
            os.environ[env_var] = str(effective)
        resolved[env_var] = float(os.getenv(env_var, str(effective)))

    return resolved


def initialize_bootstrap_wait_env(minimum: float = _VECTOR_ENV_MINIMUM) -> dict[str, float]:
    """Clamp bootstrap wait env vars to at least ``minimum`` seconds."""

    resolved: dict[str, float] = {}
    for env_var in (
        "MENACE_BOOTSTRAP_WAIT_SECS",
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS",
    ):
        raw_value = os.getenv(env_var)
        effective = minimum
        update_env = False

        if raw_value is None:
            update_env = True
        else:
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                update_env = True
            else:
                effective = parsed
                if parsed < minimum:
                    effective = minimum
                    update_env = True

        if update_env:
            os.environ[env_var] = str(effective)
        resolved[env_var] = float(os.getenv(env_var, str(effective)))

    LOGGER.info(
        "bootstrap wait env hydrated",
        extra={"minimum": minimum, "bootstrap_waits": resolved},
    )
    return resolved


def _render_timeout_policy(policy_snapshot: dict[str, dict[str, Any]]) -> str:
    parts = []
    for key in (
        "MENACE_BOOTSTRAP_WAIT_SECS",
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS",
        "BOOTSTRAP_STEP_TIMEOUT",
        "BOOTSTRAP_VECTOR_STEP_TIMEOUT",
    ):
        entry = policy_snapshot.get(key, {})
        effective = entry.get("effective")
        requested = entry.get("requested")
        clamped = entry.get("clamped")
        if effective is None:
            continue
        fragment = f"{key}={_describe_timeout(float(effective))}"
        if clamped and requested is not None:
            fragment += f" (requested={_describe_timeout(float(requested))})"
        parts.append(fragment)
    return ", ".join(parts) if parts else "defaults"


_DEFAULT_BOOTSTRAP_STEP_TIMEOUT = max(
    _default_step_floor(vector_heavy=False),
    _clamp_timeout_floor(
        float(os.getenv("BOOTSTRAP_STEP_TIMEOUT", str(_default_step_floor(vector_heavy=False)))),
        env_var="BOOTSTRAP_STEP_TIMEOUT",
    ),
)
_DEFAULT_VECTOR_BOOTSTRAP_STEP_TIMEOUT = max(
    _default_step_floor(vector_heavy=True),
    _clamp_timeout_floor(
        float(os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT", "900.0")),
        env_var="BOOTSTRAP_VECTOR_STEP_TIMEOUT",
    ),
)
BOOTSTRAP_STEP_TIMEOUT = _DEFAULT_BOOTSTRAP_STEP_TIMEOUT
BOOTSTRAP_EMBEDDER_TIMEOUT = float(os.getenv("BOOTSTRAP_EMBEDDER_TIMEOUT", "20.0"))
SELF_CODING_MIN_REMAINING_BUDGET = float(
    os.getenv("SELF_CODING_MIN_REMAINING_BUDGET", "35.0")
)
BOOTSTRAP_DEADLINE_BUFFER = 5.0
_BOOTSTRAP_EMBEDDER_DISABLED = False
_BOOTSTRAP_EMBEDDER_STARTED = False
_BOOTSTRAP_EMBEDDER_ATTEMPTED = False


def _resolve_bootstrap_lock_path() -> str:
    env_path = os.getenv(_BOOTSTRAP_LOCK_PATH_ENV)
    if env_path:
        lock_dir = os.path.dirname(env_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        return env_path

    default_root = os.path.join(os.getcwd(), "sandbox_data")
    os.makedirs(default_root, exist_ok=True)
    return os.path.join(default_root, "bootstrap.lock")


_BOOTSTRAP_CONTENTION_COORDINATOR = _BootstrapContentionCoordinator(
    _resolve_bootstrap_lock_path()
)


def _is_vector_bootstrap_heavy(candidate: Any) -> bool:
    """Identify vector-heavy helpers by their module/qualname."""

    module_name = getattr(candidate, "__module__", "") or ""
    qualname = getattr(candidate, "__qualname__", "") or ""
    text = f"{module_name}:{qualname}".lower()
    heavy_tokens = (
        "vector_service",
        "vectorservice",
        "vector_metrics",
        "vectormetrics",
        "patch_history",
        "patchhistory",
    )
    return any(token in text for token in heavy_tokens)


def _resolve_step_timeout(
    vector_heavy: bool = False,
    step_name: str = "generic",
    *,
    contention_scale: float = 1.0,
) -> float | None:
    """Resolve a bootstrap step timeout with backwards-compatible defaults."""

    resolved_timeout: float | None = None
    timeout_floor = (
        _PREPARE_VECTOR_TIMEOUT_FLOOR if vector_heavy else _PREPARE_STANDARD_TIMEOUT_FLOOR
    )
    fallback_timeout = max(
        timeout_floor,
        (
            _DEFAULT_VECTOR_BOOTSTRAP_STEP_TIMEOUT
            if vector_heavy
            else _DEFAULT_BOOTSTRAP_STEP_TIMEOUT
        ),
    )
    resolver = getattr(_coding_bot_interface, "_resolve_bootstrap_wait_timeout", None)
    if resolver:
        try:
            resolved_timeout = resolver(vector_heavy)
            if resolved_timeout is None:
                LOGGER.debug(
                    "bootstrap wait resolver returned None; using fallback timeout",
                    extra={
                        "vector_heavy": vector_heavy,
                        "fallback": fallback_timeout,
                        "env_override": (
                            "BOOTSTRAP_VECTOR_STEP_TIMEOUT"
                            if vector_heavy
                            else "BOOTSTRAP_STEP_TIMEOUT"
                        ),
                    },
                )
        except Exception:  # pragma: no cover - helper availability best effort
            LOGGER.debug("failed to resolve bootstrap wait timeout", exc_info=True)

    resolved_timeout = fallback_timeout if resolved_timeout is None else resolved_timeout

    if resolved_timeout is not None and resolved_timeout < timeout_floor:
        LOGGER.warning(
            "bootstrap wait timeout below recommended minimum; clamping",
            extra={
                "requested_timeout": resolved_timeout,
                "minimum_timeout": timeout_floor,
                "timeout_floor": _BOOTSTRAP_TIMEOUT_FLOOR,
                "vector_heavy": vector_heavy,
                "effective_timeout": timeout_floor,
            },
        )
        resolved_timeout = timeout_floor

    return _BOOTSTRAP_SCHEDULER.allocate_timeout(
        step_name,
        resolved_timeout,
        vector_heavy=vector_heavy,
        contention_scale=contention_scale,
    )


# Resolve the default timeout eagerly so legacy users retain a stable baseline.
BOOTSTRAP_STEP_TIMEOUT = _resolve_step_timeout(step_name="default")


def _mark_bootstrap_step(step_name: str) -> None:
    """Record the latest bootstrap step for external visibility."""

    global _BOOTSTRAP_TIMELINE_START

    now = time.monotonic()
    with _BOOTSTRAP_TIMELINE_LOCK:
        if _BOOTSTRAP_TIMELINE_START is None:
            _BOOTSTRAP_TIMELINE_START = now

        BOOTSTRAP_STEP_TIMELINE.append((step_name, now))

    BOOTSTRAP_PROGRESS["last_step"] = step_name
    observer = _STEP_START_OBSERVER
    if observer:
        try:
            observer(step_name)
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("step start observer failed", exc_info=True)
    _publish_online_state()


def _format_timestamp(epoch_seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


def _safe_kwargs_summary(kwargs: dict[str, Any]) -> dict[str, str]:
    try:
        redacted = redact_dict(kwargs)
    except Exception:  # pragma: no cover - defensive fallback
        redacted = kwargs

    summary: dict[str, str] = {}
    for key, value in redacted.items():
        try:
            summary[key] = summarise_value(value)
        except Exception:  # pragma: no cover - defensive fallback
            summary[key] = "<unrepresentable>"
    return summary


def _dump_thread_traces(target_thread: threading.Thread) -> None:
    frames = sys._current_frames()
    for thread in threading.enumerate():
        if not thread.ident:
            continue

        marker = f"[bootstrap-timeout][thread={thread.name} id={thread.ident}" + (
            " target]" if thread is target_thread else "]"
        )
        print(f"{marker} stack trace:", flush=True)
        buffer = io.StringIO()
        try:
            faulthandler.dump_traceback(
                file=buffer, all_threads=False, thread_id=thread.ident
            )
            trace = buffer.getvalue()
        except Exception:  # pragma: no cover - fallback for unsupported platforms
            trace = ""

        if not trace:
            frame = frames.get(thread.ident)
            if frame:
                trace = "".join(traceback.format_stack(frame))
        if trace:
            print(trace.rstrip(), flush=True)
        else:
            print(f"{marker} unable to capture stack trace", flush=True)


def _render_bootstrap_timeline(now: float) -> list[str]:
    with _BOOTSTRAP_TIMELINE_LOCK:
        timeline = list(BOOTSTRAP_STEP_TIMELINE)
        start = _BOOTSTRAP_TIMELINE_START

    if not timeline:
        return ["[bootstrap-timeout][timeline] no bootstrap steps recorded"]

    baseline = start if start is not None else timeline[0][1]

    lines = []
    for step, timestamp in timeline:
        elapsed_ms = int((timestamp - baseline) * 1000)
        lines.append(f"[bootstrap-timeout][timeline] {step} → {elapsed_ms}ms")

    return lines


def _resolve_timeout(
    base_timeout: float | None,
    *,
    bootstrap_deadline: float | None,
    heavy_bootstrap: bool = False,
    contention_scale: float = 1.0,
) -> tuple[float | None, dict[str, Any]]:
    """Resolve an effective timeout with deadline- and heavy-aware scaling."""

    effective_timeout = base_timeout
    now = time.monotonic()
    deadline_remaining = bootstrap_deadline - now if bootstrap_deadline else None
    metadata: dict[str, Any] = {
        "base_timeout": base_timeout,
        "heavy_bootstrap": heavy_bootstrap,
        "deadline_remaining": deadline_remaining,
        "deadline_buffer": BOOTSTRAP_DEADLINE_BUFFER,
    }

    if heavy_bootstrap and effective_timeout is not None:
        heavy_scale = float(os.getenv("BOOTSTRAP_HEAVY_TIMEOUT_SCALE", "1.5"))
        effective_timeout = max(effective_timeout, base_timeout * heavy_scale)
        metadata["heavy_scale"] = heavy_scale

    if effective_timeout is not None and contention_scale != 1.0:
        effective_timeout *= contention_scale
        metadata["contention_scale"] = contention_scale

    metadata["effective_timeout"] = effective_timeout
    return effective_timeout, metadata


def _compute_adaptive_budget(
    step_name: str,
    effective_timeout: float | None,
    *,
    bootstrap_deadline: float | None,
    reconciled_deadline: float | None,
    abort_on_timeout: bool,
    vector_heavy: bool = False,
    contention_scale: float = 1.0,
) -> tuple[float | None, dict[str, Any]]:
    """Adapt timeout budgets using history and remaining global window."""

    now = time.monotonic()
    deadline_reference = (
        reconciled_deadline
        if reconciled_deadline is not None
        else bootstrap_deadline
    )
    remaining_global = deadline_reference - now if deadline_reference else None
    adaptive_context: dict[str, Any] = {
        "step": step_name,
        "requested_timeout": effective_timeout,
        "remaining_global_window": remaining_global,
        "abort_on_timeout": abort_on_timeout,
        "vector_heavy": vector_heavy,
        "contention_scale": contention_scale,
        "bootstrap_deadline": bootstrap_deadline,
        "reconciled_deadline": reconciled_deadline,
    }

    predicted_budget = _BOOTSTRAP_SCHEDULER.allocate_timeout(
        step_name,
        effective_timeout,
        vector_heavy=vector_heavy,
        contention_scale=contention_scale,
    )
    adaptive_context["predicted_budget"] = predicted_budget

    adaptive_budget = predicted_budget
    if adaptive_budget is None:
        return adaptive_budget, adaptive_context

    start_reference = _BOOTSTRAP_TIMELINE_START or now
    elapsed_total = max(0.0, now - start_reference)
    allowed_total = None
    deadlines: list[float] = []
    for candidate in (bootstrap_deadline, reconciled_deadline):
        if candidate is not None:
            deadlines.append(max(0.0, candidate - start_reference))
    if deadlines:
        allowed_total = max(deadlines)

    projected_total = elapsed_total + adaptive_budget
    slack_total = None if allowed_total is None else allowed_total - projected_total

    adaptive_context.update(
        {
            "timeline_start": _BOOTSTRAP_TIMELINE_START,
            "elapsed_total": elapsed_total,
            "allowed_total": allowed_total,
            "projected_total": projected_total,
            "slack_total": slack_total,
        }
    )

    if slack_total is not None:
        if slack_total > 0 and not abort_on_timeout:
            extension = min(slack_total, max(adaptive_budget * 0.5, BOOTSTRAP_DEADLINE_BUFFER))
            adaptive_budget += extension
            adaptive_context.update(
                {
                    "extended_with_slack": True,
                    "extension": extension,
                }
            )
        elif slack_total < 0:
            adaptive_context["deadline_guardrail"] = True
            if remaining_global is not None:
                adaptive_budget = max(0.0, remaining_global - BOOTSTRAP_DEADLINE_BUFFER)

    if remaining_global is not None and adaptive_budget is not None:
        adaptive_budget = max(0.0, min(adaptive_budget, remaining_global))
        adaptive_context["adaptive_budget"] = adaptive_budget

    return adaptive_budget, adaptive_context


def _clamp_prepare_timeout_floor(
    resolved_timeout: tuple[float | None, dict[str, Any]],
    *,
    vector_heavy: bool,
    heavy_prepare: bool,
    bootstrap_deadline: float | None,
    reconciled_deadline: float | None,
) -> tuple[float | None, dict[str, Any]]:
    """Clamp prepare timeouts while respecting adaptive stage budgets."""

    effective_timeout, timeout_context = resolved_timeout
    timeout_floor = (
        _PREPARE_VECTOR_TIMEOUT_FLOOR
        if vector_heavy or heavy_prepare
        else _PREPARE_STANDARD_TIMEOUT_FLOOR
    )
    updated_context = dict(timeout_context)
    updated_context.update(
        {
            "timeout_safe_floor": timeout_floor,
            "timeout_floor_applied": timeout_floor,
            "timeout_floor_mode": "vector" if (vector_heavy or heavy_prepare) else "standard",
            "timeout_floor_auto_escalated": False,
            "vector_heavy": vector_heavy,
            "heavy_prepare": heavy_prepare,
        }
    )

    adaptive_timeout, adaptive_context = _compute_adaptive_budget(
        "prepare_pipeline_for_bootstrap",
        effective_timeout,
        bootstrap_deadline=bootstrap_deadline,
        reconciled_deadline=reconciled_deadline,
        abort_on_timeout=True,
        vector_heavy=vector_heavy or heavy_prepare,
    )
    updated_context["adaptive_prepare"] = adaptive_context
    timeout_context = updated_context

    if adaptive_timeout is None:
        return adaptive_timeout, updated_context

    deadline_remaining = None
    for candidate_deadline in (reconciled_deadline, bootstrap_deadline):
        if candidate_deadline is None:
            continue
        remaining = candidate_deadline - time.monotonic()
        if deadline_remaining is None or remaining > deadline_remaining:
            deadline_remaining = remaining

    if adaptive_timeout < timeout_floor:
        updated_context.update(
            {
                "timeout_before_floor": adaptive_timeout,
                "deadline_remaining_now": deadline_remaining,
            }
        )

        shortfall = None
        if deadline_remaining is not None and deadline_remaining < timeout_floor:
            shortfall = round(timeout_floor - deadline_remaining, 3)
            updated_context["deadline_shortfall"] = shortfall

        floor_shortfall = round(timeout_floor - adaptive_timeout, 3)
        updated_context["timeout_floor_shortfall"] = floor_shortfall
        host_load = _host_load_average()
        updated_context["host_load"] = host_load
        violation_payload = {
            "component": _PREPARE_PIPELINE_COMPONENT,
            "floor": timeout_floor,
            "shortfall": floor_shortfall,
            "requested": adaptive_timeout,
            "host_load": host_load,
            "deadline_remaining": deadline_remaining,
            "deadline_shortfall": shortfall,
            "vector_heavy": vector_heavy or heavy_prepare,
        }
        record_component_budget_violation(
            _PREPARE_PIPELINE_COMPONENT,
            floor=timeout_floor,
            shortfall=floor_shortfall,
            requested=adaptive_timeout,
            host_load=host_load,
            context=violation_payload,
        )
        try:
            watchdog = getattr(_coding_bot_interface, "_PREPARE_PIPELINE_WATCHDOG", {})
            if isinstance(watchdog, dict):
                watchdog.setdefault("budget_violations", []).append(violation_payload)
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.debug("failed to snapshot budget violation", exc_info=True)

        LOGGER.warning(
            "prepare_pipeline_for_bootstrap timeout below safe floor; raising to floor",
            extra={"timeout_context": updated_context},
        )
        adaptive_timeout = timeout_floor
        updated_context["effective_timeout"] = adaptive_timeout
        updated_context["timeout_escalated_to_floor"] = True
        updated_context["timeout_floor_auto_escalated"] = True
        if shortfall is not None:
            updated_context["timeout_floor_exceeded_deadline"] = True
        timeout_context = updated_context
    else:
        updated_context["effective_timeout"] = adaptive_timeout

    return adaptive_timeout, timeout_context


def _describe_timeout(value: float | None) -> str:
    return "disabled" if value is None else f"{value:.1f}s"


def _run_with_timeout(
    fn,
    *,
    timeout: float | None,
    bootstrap_deadline: float | None = None,
    description: str,
    abort_on_timeout: bool = True,
    heavy_bootstrap: bool = False,
    resolved_timeout: tuple[float | None, dict[str, Any]] | None = None,
    contention_scale: float = 1.0,
    budget: SharedTimeoutCoordinator | None = None,
    budget_label: str | None = None,
    **kwargs: Any,
):
    """Execute ``fn`` with a timeout to avoid indefinite hangs."""

    start_monotonic = time.monotonic()
    start_wall = time.time()
    if resolved_timeout is None:
        effective_timeout, timeout_context = _resolve_timeout(
            timeout,
            bootstrap_deadline=bootstrap_deadline,
            heavy_bootstrap=heavy_bootstrap,
            contention_scale=contention_scale,
        )
        requested_timeout = timeout
    else:
        effective_timeout, timeout_context = resolved_timeout
        requested_timeout = effective_timeout

    adaptive_timeout, adaptive_context = _compute_adaptive_budget(
        description,
        effective_timeout,
        bootstrap_deadline=bootstrap_deadline,
        reconciled_deadline=bootstrap_deadline,
        abort_on_timeout=abort_on_timeout,
        vector_heavy=heavy_bootstrap,
        contention_scale=contention_scale,
    )
    timeout_context = {**timeout_context, "adaptive": adaptive_context}
    effective_timeout = adaptive_timeout

    budget_context = (
        budget.consume(
            budget_label or description,
            requested=effective_timeout,
            minimum=timeout_context.get("timeout_floor_applied", 0.0) or 0.0,
            metadata={
                "heavy": heavy_bootstrap,
                "contention_scale": contention_scale,
            },
        )
        if budget
        else contextlib.nullcontext((effective_timeout, None))
    )

    with budget_context as (shared_timeout, budget_meta):
        if budget_meta:
            timeout_context["shared_budget"] = budget_meta
        effective_timeout = shared_timeout

        LOGGER.info(
            "%s starting with timeout (requested=%s effective=%s heavy=%s deadline=%s)",
            description,
            _describe_timeout(requested_timeout),
            _describe_timeout(effective_timeout),
            heavy_bootstrap,
            bootstrap_deadline,
            extra={
                "timeout_context": timeout_context,
                "timeout_floor_applied": timeout_context.get("timeout_floor_applied"),
                "timeout_floor_auto_escalated": timeout_context.get(
                    "timeout_floor_auto_escalated"
                ),
                "remaining_global_window": adaptive_context.get("remaining_global_window"),
                "adaptive_budget": adaptive_context.get("adaptive_budget", effective_timeout),
            },
        )

        result: Dict[str, Any] = {}

        fn_signature = None
        try:
            fn_signature = inspect.signature(fn)
        except Exception:
            fn_signature = None
        if fn_signature is not None:
            if "timeout" in fn_signature.parameters:
                kwargs.setdefault("timeout", effective_timeout)
            if "stage_budget" in fn_signature.parameters:
                kwargs.setdefault("stage_budget", effective_timeout)

        def _target() -> None:
            try:
                result["value"] = fn(**kwargs)
            except Exception as exc:  # pragma: no cover - error propagation
                result["exc"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(effective_timeout)

        last_step = BOOTSTRAP_PROGRESS.get("last_step", "unknown")

        if thread.is_alive():
            now = time.monotonic()
            with _BOOTSTRAP_TIMELINE_LOCK:
                timeline = list(BOOTSTRAP_STEP_TIMELINE)
            active_step = timeline[-1][0] if timeline else last_step
            active_started_at = timeline[-1][1] if timeline else None
            active_elapsed_ms = (
                int((now - active_started_at) * 1000)
                if active_started_at is not None
                else None
            )
            end_wall = time.time()
            deadline_remaining_now = (
                bootstrap_deadline - time.monotonic() if bootstrap_deadline else None
            )
            remediation_hints = None
            if description == "prepare_pipeline_for_bootstrap":
                remediation_hints = render_prepare_pipeline_timeout_hints()
            try:
                watchdog_timeline = list(
                    getattr(_coding_bot_interface, "_PREPARE_PIPELINE_WATCHDOG", {}).get(
                        "stages", ()
                    )
                )
            except Exception:  # pragma: no cover - best effort diagnostics
                watchdog_timeline = []

            resolved_bootstrap_wait = None
            resolved_vector_wait = None
            try:
                resolved_bootstrap_wait = _coding_bot_interface._resolve_bootstrap_wait_timeout(
                    False
                )
                resolved_vector_wait = _coding_bot_interface._resolve_bootstrap_wait_timeout(True)
            except Exception:  # pragma: no cover - best effort diagnostics
                LOGGER.debug("failed to resolve MENACE bootstrap waits", exc_info=True)
            metadata = {
                "start_time": _format_timestamp(start_wall),
                "end_time": _format_timestamp(end_wall),
                "elapsed": round(time.monotonic() - start_monotonic, 3),
                "timeout_requested": requested_timeout,
                "timeout_effective": effective_timeout,
                "timeout_effective_after_clamp": timeout_context.get(
                    "effective_timeout", effective_timeout
                ),
                "bootstrap_deadline": bootstrap_deadline,
                "bootstrap_remaining_start": timeout_context.get("deadline_remaining"),
                "bootstrap_remaining_now": deadline_remaining_now,
                "function": getattr(fn, "__name__", fn.__class__.__name__),
                "kwargs": _safe_kwargs_summary(kwargs),
                "thread_name": thread.name,
                "thread_ident": thread.ident,
                "last_step": last_step,
                "active_step": active_step,
                "active_elapsed_ms": active_elapsed_ms,
                "timeout_context": timeout_context,
                "timeout_floor_applied": timeout_context.get("timeout_floor_applied"),
                "timeout_floor_auto_escalated": timeout_context.get(
                    "timeout_floor_auto_escalated"
                ),
                "prepare_watchdog_timeline": watchdog_timeline,
                "env_menace_bootstrap_wait_resolved": resolved_bootstrap_wait,
                "env_menace_bootstrap_vector_wait_resolved": resolved_vector_wait,
                "remediation_hints": remediation_hints,
            }

            LOGGER.error(
                "%s timed out after %s (last_step=%s) metadata=%s",
                description,
                _describe_timeout(effective_timeout),
                last_step,
                metadata,
            )
            active_fragment = (
                f"active_step={active_step} (+{active_elapsed_ms}ms)"
                if active_step is not None and active_elapsed_ms is not None
                else "active_step=unknown"
            )
            print(
                ("[bootstrap-timeout][metadata] %s timed out after %s (%s): %s")
                % (
                    description,
                    _describe_timeout(effective_timeout),
                    active_fragment,
                    metadata,
                ),
                flush=True,
            )

            if remediation_hints:
                for hint in remediation_hints:
                    LOGGER.warning("[bootstrap-timeout][remediation] %s", hint)
                    print(f"[bootstrap-timeout][remediation] {hint}", flush=True)

            for line in _render_bootstrap_timeline(now):
                print(line, flush=True)

            _dump_thread_traces(thread)

            if abort_on_timeout:
                raise TimeoutError(
                    f"{description} timed out after {_describe_timeout(effective_timeout)}"
                )

            LOGGER.warning("skipping %s due to timeout", description)
            return None

        if "exc" in result:
            LOGGER.exception("%s failed", description, exc_info=result["exc"])
            print(
                (
                    "[bootstrap-error] %s failed after %s (last_step=%s)" % (
                        description,
                        _describe_timeout(effective_timeout),
                        last_step,
                    )
                ),
                flush=True,
            )
            raise result["exc"]

        return result.get("value")


def _seed_research_aggregator_context(
    *,
    registry: BotRegistry,
    data_bot: DataBot,
    context_builder: Any,
    engine: SelfCodingEngine,
    pipeline: ModelAutomationPipeline,
    manager: SelfCodingManager,
) -> None:
    try:
        from menace_sandbox import research_aggregator_bot as aggregator
    except Exception:  # pragma: no cover - optional optimisation
        LOGGER.debug("research_aggregator_bot unavailable during seeding", exc_info=True)
        return

    try:
        orchestrator = aggregator.get_orchestrator(
            "ResearchAggregatorBot", data_bot, engine
        )
    except Exception:  # pragma: no cover - orchestrator best effort
        LOGGER.debug("research aggregator orchestrator unavailable", exc_info=True)
        orchestrator = None

    aggregator.registry = registry
    aggregator.data_bot = data_bot
    aggregator._context_builder = context_builder
    aggregator.engine = engine
    aggregator.pipeline = pipeline
    aggregator.manager = manager
    aggregator.evolution_orchestrator = orchestrator
    aggregator._PipelineCls = type(pipeline)

    try:
        runtime_state = aggregator._RuntimeDependencies(
            registry=registry,
            data_bot=data_bot,
            context_builder=context_builder,
            engine=engine,
            pipeline=pipeline,
            evolution_orchestrator=orchestrator,
            manager=manager,
        )
        aggregator._runtime_state = runtime_state
        aggregator._ensure_self_coding_decorated(runtime_state)
    except Exception:  # pragma: no cover - optional runtime hints
        LOGGER.debug("unable to seed research aggregator runtime state", exc_info=True)


def _ensure_not_stopped(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise TimeoutError("initialize_bootstrap_context cancelled via stop event")


def _bootstrap_embedder(
    timeout: float,
    *,
    stop_event: threading.Event | None = None,
    stage_budget: float | None = None,
) -> None:
    """Attempt to initialise the shared embedder without blocking bootstrap."""

    global _BOOTSTRAP_EMBEDDER_DISABLED, _BOOTSTRAP_EMBEDDER_STARTED, _BOOTSTRAP_EMBEDDER_ATTEMPTED

    if timeout <= 0:
        LOGGER.info("bootstrap embedder timeout disabled; skipping embedder preload")
        return

    if _BOOTSTRAP_EMBEDDER_DISABLED:
        LOGGER.info("embedder preload disabled for this bootstrap run; skipping")
        return

    if _BOOTSTRAP_EMBEDDER_ATTEMPTED:
        LOGGER.debug("embedder preload already attempted; refusing to create another thread")
        return

    if _BOOTSTRAP_EMBEDDER_STARTED:
        LOGGER.debug("embedder preload already started; refusing to create another thread")
        return

    _BOOTSTRAP_EMBEDDER_ATTEMPTED = True

    try:
        from menace_sandbox.governed_embeddings import (
            _activate_bundled_fallback,
            apply_bootstrap_timeout_caps,
            cancel_embedder_initialisation,
            get_embedder,
        )
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.debug("governed_embeddings unavailable; skipping embedder bootstrap", exc_info=True)
        return

    result: Dict[str, Any] = {}
    embedder_stop_event = threading.Event()
    timeout_cap = apply_bootstrap_timeout_caps(stage_budget)
    hard_cap = min(timeout_cap, 5.0)
    if timeout is None:
        timeout = hard_cap
    elif timeout > 0:
        timeout = min(timeout, hard_cap)
    elif stage_budget is not None and stage_budget >= 0:
        timeout = min(stage_budget, hard_cap)
    else:
        timeout = 0.0
    soft_wait = min(timeout, 2.0) if timeout > 0 else 0.0
    _BOOTSTRAP_EMBEDDER_STARTED = True

    def _worker() -> None:
        try:
            result["embedder"] = get_embedder(
                timeout=timeout,
                stop_event=embedder_stop_event,
                bootstrap_timeout=stage_budget,
                bootstrap_mode=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            result["error"] = exc

    thread = threading.Thread(target=_worker, name="bootstrap-embedder", daemon=True)
    thread.start()
    deadline = perf_counter() + soft_wait if soft_wait > 0 else perf_counter()
    budget_deadline = (
        perf_counter() + stage_budget if stage_budget is not None and stage_budget >= 0 else None
    )
    next_progress = perf_counter() + 0.5
    join_min_interval = 0.05

    def _handle_embedder_timeout() -> Any:
        LOGGER.warning(
            "embedding model load exceeded %.1fs during bootstrap; deferring to stub",
            soft_wait,
        )
        embedder_stop_event.set()
        cancel_embedder_initialisation(
            embedder_stop_event, reason="bootstrap_timeout", join_timeout=1.0
        )

        timed_out_embedder = result.get("embedder")
        if timed_out_embedder is not None:
            LOGGER.info(
                "bootstrap embedder became ready after timeout: %s",
                type(timed_out_embedder).__name__,
            )
            return timed_out_embedder

        LOGGER.info(
            "bootstrap embedder warmup deferred after %.1fs; exposing stub fallback",
            soft_wait,
        )

        fallback_used = False
        try:
            fallback_used = _activate_bundled_fallback("bootstrap_timeout")
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.warning(
                "failed to activate bundled fallback embedder after timeout", exc_info=True
            )

        _BOOTSTRAP_EMBEDDER_DISABLED = True
        _BOOTSTRAP_EMBEDDER_STARTED = False

        if fallback_used:
            fallback_embedder = get_embedder(timeout=0, stop_event=stop_event)
            if fallback_embedder is not None:
                LOGGER.info(
                    "bootstrap embedder fallback ready: %s",
                    type(fallback_embedder).__name__,
                )
                return fallback_embedder
            LOGGER.info(
                "bundled embedder fallback requested after timeout but none available yet"
            )
        return

    while True:
        if not thread.is_alive():
            break

        now = perf_counter()
        if budget_deadline is not None and now >= budget_deadline:
            LOGGER.warning(
                "embedder preload exceeded stage budget after %.1fs; cancelling warmup",
                stage_budget,
            )
            embedder_stop_event.set()
            cancel_embedder_initialisation(
                embedder_stop_event, reason="bootstrap_budget_exceeded", join_timeout=1.0
            )
            try:
                _activate_bundled_fallback("bootstrap_budget")
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("failed to promote bundled embedder after budget cancel", exc_info=True)
            _BOOTSTRAP_EMBEDDER_DISABLED = True
            _BOOTSTRAP_EMBEDDER_STARTED = False
            return

        if now >= deadline:
            return _handle_embedder_timeout()

        remaining = max(0.0, deadline - now)
        slice_wait = min(0.25, remaining) if remaining > 0 else join_min_interval
        slice_wait = max(slice_wait, join_min_interval)
        thread.join(slice_wait)

        if thread.is_alive() and perf_counter() >= next_progress:
            remaining_budget = (
                max(0.0, budget_deadline - perf_counter()) if budget_deadline is not None else None
            )
            LOGGER.debug(
                "embedder preload still warming (remaining_slice=%.2fs, budget_remaining=%s)",
                remaining or slice_wait,
                f"{remaining_budget:.2f}s" if remaining_budget is not None else "n/a",
            )
            next_progress = perf_counter() + 0.5

    if "error" in result:
        LOGGER.info("embedder preload failed; bootstrap will proceed without embeddings")
        LOGGER.debug("embedder preload error", exc_info=result["error"])
        _BOOTSTRAP_EMBEDDER_DISABLED = True
        _BOOTSTRAP_EMBEDDER_STARTED = False
        return

    embedder = result.get("embedder")
    if embedder is None:
        LOGGER.info(
            "embedder unavailable after %.1fs wait; proceeding without embeddings",
            timeout,
        )
        _BOOTSTRAP_EMBEDDER_DISABLED = True
        _BOOTSTRAP_EMBEDDER_STARTED = False
    else:
        placeholder_reason = getattr(embedder, "_placeholder_reason", None)
        if placeholder_reason:
            LOGGER.warning(
                "bootstrap using degraded embedder placeholder (%s); later calls may retry full initialisation",
                placeholder_reason,
                extra={"timeout": timeout, "stage_budget": stage_budget},
            )
            _BOOTSTRAP_EMBEDDER_STARTED = False
        LOGGER.info("bootstrap embedder ready: %s", type(embedder).__name__)
    _BOOTSTRAP_EMBEDDER_STARTED = False


def initialize_bootstrap_context(
    bot_name: str = "ResearchAggregatorBot",
    *,
    use_cache: bool = True,
    heavy_bootstrap: bool = False,
    stop_event: threading.Event | None = None,
    bootstrap_deadline: float | None = None,
    shared_timeout_coordinator: SharedTimeoutCoordinator | None = None,
    stage_deadlines: Mapping[str, Any] | None = None,
    progress_signal: Callable[[Mapping[str, object]], None] | None = None,
) -> Dict[str, Any]:
    """Build and seed bootstrap helpers for reuse by entry points.

    The returned mapping contains the seeded ``registry``, ``data_bot``,
    ``context_builder``, ``engine``, ``pipeline`` and ``manager`` instances.
    Subsequent invocations return cached instances for the given ``bot_name`` when
    ``use_cache`` is ``True``. Pass ``use_cache=False`` to force a fresh bootstrap
    without populating or reading the shared cache. Enable ``heavy_bootstrap`` (or
    set ``BOOTSTRAP_HEAVY_BOOTSTRAP``) to allow timeouts to scale up when more time
    is available or heavy vector work is expected.
    """

    global _BOOTSTRAP_CACHE

    def _log_step(step_name: str, start_time: float) -> None:
        elapsed = perf_counter() - start_time
        LOGGER.info("%s completed (elapsed=%.3fs)", step_name, elapsed)
        _BOOTSTRAP_SCHEDULER.record_history(step_name, elapsed)
        observer = _STEP_END_OBSERVER
        if observer:
            try:
                observer(step_name, elapsed)
            except Exception:  # pragma: no cover - advisory only
                LOGGER.debug("step end observer failed", exc_info=True)
        _publish_online_state()

    def _timed_callable(func: Any, *, label: str, **func_kwargs: Any) -> Any:
        start = perf_counter()
        LOGGER.debug("starting %s", label)
        try:
            return func(**func_kwargs)
        finally:
            LOGGER.debug("%s completed (elapsed=%.3fs)", label, perf_counter() - start)

    def _apply_vector_env(reason: str) -> dict[str, str | float]:
        vector_env = _hydrate_vector_bootstrap_env()
        os.environ["VECTOR_SERVICE_HEAVY"] = "1"
        snapshot = {
            "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS": os.getenv(
                "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"
            ),
            "BOOTSTRAP_VECTOR_STEP_TIMEOUT": os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT"),
            "VECTOR_SERVICE_HEAVY": os.getenv("VECTOR_SERVICE_HEAVY"),
        }
        LOGGER.info(
            "vector-heavy bootstrap env applied",
            extra={
                "reason": reason,
                "vector_env": snapshot,
                "timeout_policy": enforce_bootstrap_timeout_policy(logger=LOGGER),
            },
        )
        print(f"vector bootstrap env ({reason}): {snapshot}", flush=True)
        return {**snapshot, **vector_env}

    env_heavy = os.getenv("BOOTSTRAP_HEAVY_BOOTSTRAP", "")
    heavy_bootstrap = heavy_bootstrap or env_heavy.lower() in {"1", "true", "yes"}
    vector_bootstrap_hint = False
    vector_env_snapshot: dict[str, str | float] = {}
    LOGGER.info(
        "initialize_bootstrap_context heavy mode=%s", heavy_bootstrap,
        extra={"heavy_env": env_heavy},
    )
    timeout_policy = enforce_bootstrap_timeout_policy(logger=LOGGER)
    _apply_timeout_policy_snapshot(timeout_policy)
    timeout_policy_summary = _render_timeout_policy(timeout_policy)
    LOGGER.info(
        "bootstrap timeout policy applied",
        extra={"timeout_policy": timeout_policy},
    )
    print(f"bootstrap timeout policy: {timeout_policy_summary}", flush=True)
    component_timeout_floors = timeout_policy.get(
        "component_floors", load_component_timeout_floors()
    )
    resolved_stage_deadlines = stage_deadlines
    if resolved_stage_deadlines is None:
        baseline_timeout = float(
            os.getenv("BOOTSTRAP_STEP_TIMEOUT", str(BOOTSTRAP_STEP_TIMEOUT))
            or BOOTSTRAP_STEP_TIMEOUT
        )
        duration_store = load_duration_store()
        stage_stats = compute_stats(duration_store.get("bootstrap_stages", {}))
        base_stage_budgets = {stage.name: baseline_timeout for stage in READINESS_STAGES}
        calibrated_stage_budgets, budget_debug = calibrate_step_budgets(
            base_budgets=base_stage_budgets,
            stats=stage_stats,
            floors={stage.name: _COMPONENT_BASELINES.get(stage.name, 0.0) for stage in READINESS_STAGES},
        )
        guard_context = get_bootstrap_guard_context() or {}
        try:
            guard_scale = max(float(guard_context.get("budget_scale", 1.0) or 1.0), 1.0)
        except Exception:
            guard_scale = 1.0
        if guard_scale > 1.0:
            calibrated_stage_budgets = {
                stage: budget * guard_scale for stage, budget in calibrated_stage_budgets.items()
            }
        calibrated_timeout, timeout_debug = calibrate_overall_timeout(
            base_timeout=baseline_timeout, calibrated_budgets=calibrated_stage_budgets
        )
        calibrated_timeout *= guard_scale
        resolved_stage_deadlines = build_stage_deadlines(
            calibrated_timeout, soft_deadline=True
        )
        LOGGER.info(
            "bootstrap stage deadlines initialised from baseline budget",
            extra={
                "event": "bootstrap-stage-deadlines",
                "baseline_timeout": calibrated_timeout,
                "stage_policy": resolved_stage_deadlines,
                "stage_budgets": calibrated_stage_budgets,
                "budget_adjustments": budget_debug.get("adjusted"),
                "budget_scale": guard_scale,
                "timeout_context": timeout_debug,
            },
        )

    component_budgets = compute_prepare_pipeline_component_budgets(
        component_floors=component_timeout_floors,
        telemetry=collect_timeout_telemetry(),
        host_telemetry=read_bootstrap_heartbeat(),
        pipeline_complexity=resolved_stage_deadlines,
    )
    _BOOTSTRAP_SCHEDULER.set_component_budgets(component_budgets)
    _apply_timeout_policy_snapshot(timeout_policy)
    BOOTSTRAP_ONLINE_STATE["stage_policy"] = resolved_stage_deadlines
    _publish_online_state()

    resolved_bootstrap_window = None
    try:
        resolved_bootstrap_window = _coding_bot_interface._resolve_bootstrap_wait_timeout(
            vector_bootstrap_hint
        )
    except Exception:  # pragma: no cover - diagnostics only
        LOGGER.debug("failed to resolve shared bootstrap window", exc_info=True)
    reconciled_bootstrap_deadline = bootstrap_deadline
    deadline_remaining = None
    now_monotonic = time.monotonic()
    if resolved_bootstrap_window is not None:
        resolved_window_deadline = now_monotonic + resolved_bootstrap_window
        if reconciled_bootstrap_deadline is None:
            reconciled_bootstrap_deadline = resolved_window_deadline
        else:
            reconciled_bootstrap_deadline = max(
                reconciled_bootstrap_deadline, resolved_window_deadline
            )

    if reconciled_bootstrap_deadline is not None:
        deadline_remaining = max(0.0, reconciled_bootstrap_deadline - time.monotonic())
        resolved_bootstrap_window = max(resolved_bootstrap_window or 0.0, deadline_remaining)
        bootstrap_deadline = reconciled_bootstrap_deadline

    if progress_signal is None:
        progress_signal = build_progress_signal_hook(namespace="bootstrap_shared")

    if shared_timeout_coordinator is None:
        shared_timeout_coordinator = SharedTimeoutCoordinator(
            resolved_bootstrap_window,
            logger=LOGGER,
            namespace="bootstrap_shared",
            component_floors=component_timeout_floors,
            component_budgets=component_budgets,
            signal_hook=progress_signal,
            complexity_inputs=resolved_stage_deadlines,
        )
    else:
        LOGGER.info(
            "initialize_bootstrap_context using shared timeout coordinator",
            extra={"shared_timeout": shared_timeout_coordinator.snapshot()},
        )

    stage_controller = _StagedBootstrapController(
        stage_policy=resolved_stage_deadlines,
        coordinator=shared_timeout_coordinator,
        signal_hook=progress_signal,
    )
    global _STEP_START_OBSERVER, _STEP_END_OBSERVER
    _STEP_START_OBSERVER = stage_controller.start_step
    _STEP_END_OBSERVER = stage_controller.complete_step

    set_audit_bootstrap_safe_default(True)
    _ensure_not_stopped(stop_event)

    runner = _BootstrapDagRunner(logger=LOGGER)
    vector_bootstrap_hint_holder = {"vector": vector_bootstrap_hint}

    def _task_embedder(_: dict[str, Any]) -> None:
        _mark_bootstrap_step("embedder_preload")
        embedder_gate = _BOOTSTRAP_CONTENTION_COORDINATOR.negotiate_step(
            "embedder_preload", vector_heavy=True, heavy=True
        )
        embedder_timeout = BOOTSTRAP_EMBEDDER_TIMEOUT * embedder_gate["timeout_scale"]
        _run_with_timeout(
            _bootstrap_embedder,
            timeout=embedder_timeout,
            bootstrap_deadline=bootstrap_deadline,
            description="bootstrap_embedder_preload",
            heavy_bootstrap=True,
            budget=None,
            stop_event=stop_event,
        )

    def _task_context_builder(_: dict[str, Any]) -> Any:
        _ensure_not_stopped(stop_event)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "db_index_load", reason="context_builder_start"
        )
        _mark_bootstrap_step("context_builder")
        ctx_builder_start = perf_counter()
        try:
            builder = create_context_builder(bootstrap_safe=True)
        except Exception:
            LOGGER.exception("context_builder creation failed (step=context_builder)")
            raise
        _log_step("context_builder", ctx_builder_start)
        _BOOTSTRAP_SCHEDULER.mark_ready(
            "db_index_load", reason="context_builder_ready"
        )
        try:
            vector_bootstrap_hint_holder["vector"] = _is_vector_bootstrap_heavy(builder)
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("failed to inspect context builder for vector-heavy hint", exc_info=True)
        if vector_bootstrap_hint_holder["vector"]:
            vector_env_snapshot.update(_apply_vector_env("context_builder_hint"))
            try:
                _coding_bot_interface._BOOTSTRAP_STATE.vector_heavy = True
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("unable to propagate vector-heavy bootstrap flag", exc_info=True)
            timeout_policy = enforce_bootstrap_timeout_policy(logger=LOGGER)
            _apply_timeout_policy_snapshot(timeout_policy)
            timeout_policy_summary = _render_timeout_policy(timeout_policy)
            print(f"bootstrap timeout policy: {timeout_policy_summary}", flush=True)
        return builder

    def _task_registry(_: dict[str, Any]) -> BotRegistry:
        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("bot_registry")
        registry_start = perf_counter()
        try:
            reg = BotRegistry()
        except Exception:
            LOGGER.exception("BotRegistry initialization failed (step=bot_registry)")
            raise
        _log_step("bot_registry", registry_start)
        return reg

    def _task_data_bot(results: dict[str, Any]) -> DataBot:
        _ensure_not_stopped(stop_event)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "retriever_hydration", reason="data_bot_start"
        )
        _mark_bootstrap_step("data_bot")
        data_bot_start = perf_counter()
        try:
            data_bot_instance = DataBot(start_server=False)
        except Exception:
            LOGGER.exception("DataBot setup failed (step=data_bot)")
            raise
        _log_step("data_bot", data_bot_start)
        _BOOTSTRAP_SCHEDULER.mark_ready(
            "retriever_hydration", reason="data_bot_ready"
        )
        return data_bot_instance

    def _task_engine(results: dict[str, Any]) -> SelfCodingEngine:
        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("self_coding_engine")
        engine_start = perf_counter()
        try:
            engine_instance = SelfCodingEngine(
                CodeDB(),
                MenaceMemoryManager(),
                context_builder=results["context_builder"],
            )
        except Exception:
            LOGGER.exception("SelfCodingEngine instantiation failed (step=self_coding_engine)")
            raise
        _log_step("self_coding_engine", engine_start)
        return engine_instance

    if use_cache:
        cached_context = _BOOTSTRAP_CACHE.get(bot_name)
        if cached_context:
            LOGGER.info(
                "reusing preseeded bootstrap context for %s; pipeline/manager already available",
                bot_name,
            )
            return cached_context

    runner.add(
        _BootstrapTask(
            name="vectorizer_preload",
            fn=_task_embedder,
            component="vectorizer_preload",
            critical=False,
            background=True,
        )
    )
    runner.add(
        _BootstrapTask(
            name="context_builder",
            fn=_task_context_builder,
            component="db_index_load",
            requires=(),
        )
    )
    runner.add(
        _BootstrapTask(
            name="bot_registry",
            fn=_task_registry,
            requires=("context_builder",),
        )
    )
    runner.add(
        _BootstrapTask(
            name="data_bot",
            fn=_task_data_bot,
            requires=("bot_registry",),
            component="retriever_hydration",
        )
    )
    runner.add(
        _BootstrapTask(
            name="self_coding_engine",
            fn=_task_engine,
            requires=("context_builder",),
        )
    )

    if use_cache:
        cached_context = _BOOTSTRAP_CACHE.get(bot_name)
        if cached_context:
            LOGGER.info(
                "reusing preseeded bootstrap context for %s; pipeline/manager already available",
                bot_name,
            )
            return cached_context

    with _BOOTSTRAP_CACHE_LOCK:
        _ensure_not_stopped(stop_event)
        if use_cache:
            cached_context = _BOOTSTRAP_CACHE.get(bot_name)
            if cached_context:
                LOGGER.info(
                    "reusing preseeded bootstrap context for %s; pipeline/manager already available",
                    bot_name,
                )
                return cached_context

        _ensure_not_stopped(stop_event)

        dag_results = runner.run(
            critical_gate=("context_builder", "bot_registry", "data_bot", "self_coding_engine"),
        )
        context_builder = dag_results["context_builder"]
        registry = dag_results["bot_registry"]
        data_bot = dag_results["data_bot"]
        engine = dag_results["self_coding_engine"]
        vector_bootstrap_hint = vector_bootstrap_hint_holder["vector"]
        BOOTSTRAP_ONLINE_STATE["critical_ready"] = True
        _publish_online_state()

        remaining_budget = (
            bootstrap_deadline - time.monotonic() if bootstrap_deadline else None
        )
        if remaining_budget is not None and remaining_budget < SELF_CODING_MIN_REMAINING_BUDGET:
            LOGGER.warning(
                "remaining bootstrap budget is low (%.1fs < %.1fs); skipping self-coding bootstrap",
                max(remaining_budget, 0.0),
                SELF_CODING_MIN_REMAINING_BUDGET,
            )
            _ensure_not_stopped(stop_event)
            with fallback_helper_manager(
                bot_registry=registry, data_bot=data_bot
            ) as bootstrap_manager:
                _mark_bootstrap_step("push_final_context")
                push_timeout = _resolve_step_timeout(
                    step_name="push_final_context",
                    vector_heavy=vector_bootstrap_hint,
                )
                _run_with_timeout(
                    _push_bootstrap_context,
                    timeout=push_timeout,
                    bootstrap_deadline=bootstrap_deadline,
                    description="_push_bootstrap_context final",
                    abort_on_timeout=True,
                    heavy_bootstrap=heavy_bootstrap,
                    budget=shared_timeout_coordinator,
                    budget_label="db_indexes",
                    registry=registry,
                    data_bot=data_bot,
                    manager=bootstrap_manager,
                    pipeline=bootstrap_manager,
                )
                _mark_bootstrap_step("seed_final_context")
                seed_timeout = _resolve_step_timeout(
                    step_name="seed_final_context",
                    vector_heavy=True,
                )
                _run_with_timeout(
                    _seed_research_aggregator_context,
                    timeout=seed_timeout,
                    bootstrap_deadline=bootstrap_deadline,
                    description="_seed_research_aggregator_context final",
                    abort_on_timeout=False,
                    heavy_bootstrap=heavy_bootstrap,
                    budget=shared_timeout_coordinator,
                    budget_label="orchestrator_state",
                    registry=registry,
                    data_bot=data_bot,
                    context_builder=context_builder,
                    engine=getattr(bootstrap_manager, "engine", None),
                    pipeline=bootstrap_manager,
                    manager=bootstrap_manager,
                )

            bootstrap_context = {
                "registry": registry,
                "data_bot": data_bot,
                "context_builder": context_builder,
                "engine": getattr(bootstrap_manager, "engine", None),
                "pipeline": bootstrap_manager,
                "manager": bootstrap_manager,
            }
            if use_cache:
                _BOOTSTRAP_CACHE[bot_name] = bootstrap_context
            LOGGER.info(
                "initialize_bootstrap_context completed with disabled self-coding due to budget constraints",
                extra={"remaining_budget": remaining_budget},
            )
            return bootstrap_context

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("self_coding_engine")
        engine_start = perf_counter()
        try:
            engine = SelfCodingEngine(
                CodeDB(),
                MenaceMemoryManager(),
                context_builder=context_builder,
            )
        except Exception:
            LOGGER.exception("SelfCodingEngine instantiation failed (step=self_coding_engine)")
            raise
        _log_step("self_coding_engine", engine_start)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("prepare_pipeline")
        with fallback_helper_manager(
            bot_registry=registry, data_bot=data_bot
        ) as bootstrap_manager:
            LOGGER.info(
                "seeding research aggregator with bootstrap manager before pipeline preparation"
            )
            LOGGER.info(
                "before _push_bootstrap_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            placeholder_push_timeout = _resolve_step_timeout(
                step_name="_push_bootstrap_context", vector_heavy=vector_bootstrap_hint
            )
            placeholder_context = _run_with_timeout(
                _timed_callable,
                timeout=placeholder_push_timeout,
                bootstrap_deadline=bootstrap_deadline,
                description="_push_bootstrap_context placeholder",
                abort_on_timeout=True,
                heavy_bootstrap=heavy_bootstrap,
                budget=shared_timeout_coordinator,
                budget_label="db_indexes",
                func=_push_bootstrap_context,
                label="_push_bootstrap_context placeholder",
                registry=registry,
                data_bot=data_bot,
                manager=bootstrap_manager,
                pipeline=bootstrap_manager,
                bootstrap_safe=True,
                bootstrap_fast=True,
            )
            LOGGER.info(
                "after _push_bootstrap_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            LOGGER.info("_push_bootstrap_context completed (step=push_placeholder)")
            LOGGER.info(
                "before _seed_research_aggregator_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            placeholder_seed_gate = _BOOTSTRAP_CONTENTION_COORDINATOR.negotiate_step(
                "_seed_research_aggregator_context_placeholder",
                vector_heavy=True,
                heavy=True,
            )
            placeholder_seed_timeout = _resolve_step_timeout(
                step_name="_seed_research_aggregator_context",
                vector_heavy=True,
                contention_scale=placeholder_seed_gate["timeout_scale"],
            )
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "vector_seeding", reason="placeholder_seed"
            )
            _run_with_timeout(
                _timed_callable,
                timeout=placeholder_seed_timeout,
                bootstrap_deadline=bootstrap_deadline,
                description="_seed_research_aggregator_context placeholder",
                abort_on_timeout=False,
                heavy_bootstrap=heavy_bootstrap,
                contention_scale=placeholder_seed_gate["timeout_scale"],
                budget=shared_timeout_coordinator,
                budget_label="orchestrator_state",
                func=_seed_research_aggregator_context,
                label="_seed_research_aggregator_context placeholder",
                registry=registry,
                data_bot=data_bot,
                context_builder=context_builder,
                engine=engine,
                pipeline=bootstrap_manager,
                manager=bootstrap_manager,
            )
            LOGGER.info(
                "after _seed_research_aggregator_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            LOGGER.info(
                "starting prepare_pipeline_for_bootstrap (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            prepare_gate = _BOOTSTRAP_CONTENTION_COORDINATOR.negotiate_step(
                "prepare_pipeline_for_bootstrap",
                vector_heavy=vector_bootstrap_hint or vector_heavy,
                heavy=True,
            )
            vector_heavy = False
            vector_timeout = _resolve_step_timeout(
                vector_heavy=True,
                step_name="prepare_pipeline_for_bootstrap",
                contention_scale=prepare_gate["timeout_scale"],
            )
            standard_timeout = _resolve_step_timeout(
                vector_heavy=False,
                step_name="prepare_pipeline_for_bootstrap",
                contention_scale=prepare_gate["timeout_scale"],
            )
            try:
                vector_state = getattr(_coding_bot_interface, "_BOOTSTRAP_STATE", None)
                if vector_state is not None:
                    vector_heavy = bool(getattr(vector_state, "vector_heavy", False))
                if not vector_heavy and _is_vector_bootstrap_heavy(context_builder):
                    vector_heavy = True
                if not vector_heavy and _is_vector_bootstrap_heavy(ModelAutomationPipeline):
                    vector_heavy = True
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("unable to inspect vector_heavy flag", exc_info=True)

            if vector_heavy:
                vector_env_snapshot = vector_env_snapshot or _apply_vector_env(
                    "prepare_pipeline_vector_heavy"
                )

            prepare_timeout = vector_timeout if vector_heavy else standard_timeout
            prepare_timeout_floor = (
                _PREPARE_VECTOR_TIMEOUT_FLOOR
                if vector_heavy or heavy_bootstrap
                else _PREPARE_STANDARD_TIMEOUT_FLOOR
            )
            if prepare_timeout is not None and prepare_timeout < prepare_timeout_floor:
                LOGGER.warning(
                    "prepare_pipeline_for_bootstrap timeout below safe floor; clamping",
                    extra={
                        "vector_heavy": vector_heavy,
                        "requested_timeout": prepare_timeout,
                        "minimum_timeout": prepare_timeout_floor,
                    },
                )
                prepare_timeout = prepare_timeout_floor
            heavy_prepare = heavy_bootstrap or vector_heavy
            resolved_prepare_timeout = _resolve_timeout(
                prepare_timeout,
                bootstrap_deadline=bootstrap_deadline,
                heavy_bootstrap=heavy_prepare,
                contention_scale=prepare_gate["timeout_scale"],
            )
            resolved_prepare_timeout = _clamp_prepare_timeout_floor(
                resolved_prepare_timeout,
                vector_heavy=vector_heavy,
                heavy_prepare=heavy_prepare,
                bootstrap_deadline=bootstrap_deadline,
                reconciled_deadline=bootstrap_deadline,
            )
            effective_prepare_timeout = resolved_prepare_timeout[0]
            LOGGER.info(
                "prepare_pipeline timeout selected",
                extra={
                    "vector_heavy": vector_heavy,
                    "timeout": effective_prepare_timeout,
                    "timeout_requested": effective_prepare_timeout,
                    "timeout_requested_raw": prepare_timeout,
                    "vector_timeout": vector_timeout,
                    "standard_timeout": standard_timeout,
                    "heavy_bootstrap": heavy_prepare,
                    "timeout_context": resolved_prepare_timeout[1],
                    "timeout_policy": timeout_policy,
                    "vector_env": vector_env_snapshot,
                },
            )
            print(
                (
                    "starting prepare_pipeline_for_bootstrap "
                    "(last_step=%s, timeout=%s, elapsed=0.0s, timeouts=%s)"
                )
                % (
                    BOOTSTRAP_PROGRESS["last_step"],
                    _describe_timeout(effective_prepare_timeout),
                    _render_timeout_policy(timeout_policy),
                ),
                flush=True,
            )
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "orchestrator_state", reason="prepare_pipeline_start"
            )
            _set_component_state("orchestrator_state", "warming")
            prepare_start = perf_counter()
            lock_path = _resolve_bootstrap_lock_path()
            bootstrap_lock = SandboxLock(lock_path)
            lock_wait_start = perf_counter()
            LOGGER.info(
                "waiting for bootstrap lock",
                extra={"lock_path": lock_path},
            )
            try:
                with bootstrap_lock:
                    LOGGER.info(
                        "bootstrap lock acquired",
                        extra={
                            "lock_path": lock_path,
                            "wait_time": perf_counter() - lock_wait_start,
                        },
                    )
                    prepare_degraded = False
                    retry_scheduled = False
                    prepare_resume_hook: Callable[[], Any] | None = None
                    prepare_result: tuple[Any, Callable[[Any], None]] | None = None
                    prepare_kwargs = {
                        "func": prepare_pipeline_for_bootstrap,
                        "label": "prepare_pipeline_for_bootstrap",
                        "stop_event": stop_event,
                        "pipeline_cls": ModelAutomationPipeline,
                        "context_builder": context_builder,
                        "bot_registry": registry,
                        "data_bot": data_bot,
                        "bootstrap_runtime_manager": bootstrap_manager,
                        "manager": bootstrap_manager,
                        "bootstrap_safe": True,
                        "bootstrap_fast": True,
                        "component_timeouts": component_budgets,
                    }

                    def _schedule_prepare_retry() -> bool:
                        if shared_timeout_coordinator is None:
                            return False

                        retry_request = effective_prepare_timeout
                        retry_context = _resolve_timeout(
                            retry_request,
                            bootstrap_deadline=bootstrap_deadline,
                            heavy_bootstrap=heavy_prepare,
                            contention_scale=prepare_gate["timeout_scale"],
                        )
                        retry_context = _clamp_prepare_timeout_floor(
                            retry_context,
                            vector_heavy=vector_heavy,
                            heavy_prepare=heavy_prepare,
                            bootstrap_deadline=bootstrap_deadline,
                            reconciled_deadline=bootstrap_deadline,
                        )

                        def _retry_worker() -> None:
                            retry_label = "prepare_pipeline_for_bootstrap.retry"
                            LOGGER.info(
                                "restarting prepare_pipeline_for_bootstrap in background",
                                extra={
                                    "timeout": retry_context[0],
                                    "timeout_context": retry_context[1],
                                    "remaining_budget": getattr(
                                        shared_timeout_coordinator, "remaining_budget", None
                                    ),
                                },
                            )
                            try:
                                retry_pipeline = _run_with_timeout(
                                    _timed_callable,
                                    timeout=retry_context[0],
                                    bootstrap_deadline=bootstrap_deadline,
                                    description=retry_label,
                                    abort_on_timeout=False,
                                    heavy_bootstrap=heavy_prepare,
                                    contention_scale=prepare_gate["timeout_scale"],
                                    resolved_timeout=retry_context,
                                    budget=shared_timeout_coordinator,
                                    budget_label="retrievers.retry",
                                    **prepare_kwargs,
                                )
                            except Exception:
                                LOGGER.exception("background prepare retry failed")
                                return

                            if retry_pipeline:
                                LOGGER.info(
                                    "background prepare pipeline ready after retry",
                                    extra={"timeout": retry_context[0]},
                                )
                                _BOOTSTRAP_SCHEDULER.mark_ready(
                                    "orchestrator_state", reason="prepare_retry_ready"
                                )
                                _set_component_state("orchestrator_state", "ready")

                        thread = threading.Thread(
                            target=_retry_worker,
                            name="bootstrap-prepare-retry",
                            daemon=True,
                        )
                        thread.start()
                        return True

                    try:
                        prepare_result = _run_with_timeout(
                            _timed_callable,
                            timeout=effective_prepare_timeout,
                            bootstrap_deadline=bootstrap_deadline,
                            description="prepare_pipeline_for_bootstrap",
                            abort_on_timeout=True,
                            heavy_bootstrap=heavy_prepare,
                            contention_scale=prepare_gate["timeout_scale"],
                            resolved_timeout=resolved_prepare_timeout,
                            budget=shared_timeout_coordinator,
                            budget_label="retrievers",
                            **prepare_kwargs,
                        )
                    except TimeoutError:
                        prepare_degraded = True
                        retry_scheduled = _schedule_prepare_retry()
                        prepare_resume_hook = lambda: _run_with_timeout(
                            _timed_callable,
                            timeout=effective_prepare_timeout,
                            bootstrap_deadline=bootstrap_deadline,
                            description="prepare_pipeline_for_bootstrap.resume",
                            abort_on_timeout=False,
                            heavy_bootstrap=heavy_prepare,
                            contention_scale=prepare_gate["timeout_scale"],
                            resolved_timeout=resolved_prepare_timeout,
                            budget=shared_timeout_coordinator,
                            budget_label="retrievers.resume",
                            **prepare_kwargs,
                        )
                        LOGGER.warning(
                            "prepare_pipeline_for_bootstrap exceeded budget; entering degraded mode",
                            extra={
                                "timeout": effective_prepare_timeout,
                                "timeout_context": resolved_prepare_timeout[1],
                                "shared_budget": getattr(
                                    shared_timeout_coordinator, "remaining_budget", None
                                ),
                                "retry_scheduled": retry_scheduled,
                            },
                        )
                        _BOOTSTRAP_SCHEDULER.mark_partial(
                            "orchestrator_state", reason="prepare_timeout"
                        )
                        _set_component_state("orchestrator_state", "degraded")
                        prepare_result = None
                    except Exception:
                        LOGGER.exception("prepare_pipeline_for_bootstrap failed (step=prepare_pipeline)")
                        print(
                            (
                                "prepare_pipeline_for_bootstrap failed "
                                "(last_step=%s, timeout=%s, elapsed=%.2fs)"
                            )
                            % (
                                BOOTSTRAP_PROGRESS["last_step"],
                                _describe_timeout(effective_prepare_timeout),
                                perf_counter() - prepare_start,
                            ),
                            flush=True,
                        )
                        raise
                    finally:
                        _pop_bootstrap_context(placeholder_context)

                    if prepare_degraded:
                        degrade_snapshot = _BOOTSTRAP_SCHEDULER.snapshot()
                        LOGGER.info(
                            "continuing bootstrap in reduced-capability mode",
                            extra={
                                "online_state": degrade_snapshot,
                                "retry_scheduled": retry_scheduled,
                            },
                        )
                        _publish_online_state()
                        _log_step("prepare_pipeline_for_bootstrap", prepare_start)
                        bootstrap_context = {
                            "registry": registry,
                            "data_bot": data_bot,
                            "context_builder": context_builder,
                            "engine": engine,
                            "pipeline": bootstrap_manager,
                            "manager": bootstrap_manager,
                            "degraded_prepare": True,
                            "prepare_retry": retry_scheduled,
                            "resume_prepare": prepare_resume_hook,
                            "online_state": degrade_snapshot,
                        }
                        if use_cache:
                            _BOOTSTRAP_CACHE[bot_name] = bootstrap_context
                        return bootstrap_context

                    pipeline, promote_pipeline = prepare_result
            except Exception:
                LOGGER.exception("prepare_pipeline_for_bootstrap failed (step=prepare_pipeline)")
                print(
                    (
                        "prepare_pipeline_for_bootstrap failed "
                        "(last_step=%s, timeout=%s, elapsed=%.2fs)"
                    )
                    % (
                        BOOTSTRAP_PROGRESS["last_step"],
                        _describe_timeout(effective_prepare_timeout),
                        perf_counter() - prepare_start,
                    ),
                    flush=True,
                )
                raise
            finally:
                _pop_bootstrap_context(placeholder_context)
            LOGGER.info(
                "prepare_pipeline_for_bootstrap finished (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            print(
                (
                    "prepare_pipeline_for_bootstrap finished "
                    "(last_step=%s, timeout=%s, elapsed=%.2fs)"
                )
                % (
                    BOOTSTRAP_PROGRESS["last_step"],
                    _describe_timeout(effective_prepare_timeout),
                    perf_counter() - prepare_start,
                ),
                flush=True,
            )
            _log_step("prepare_pipeline_for_bootstrap", prepare_start)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("threshold_persistence")
        thresholds = get_thresholds(bot_name)
        try:
            persist_sc_thresholds(
                bot_name,
                roi_drop=thresholds.roi_drop,
                error_increase=thresholds.error_increase,
                test_failure_increase=thresholds.test_failure_increase,
            )
            LOGGER.info("persist_sc_thresholds completed (step=threshold_persistence)")
        except Exception:  # pragma: no cover - best effort persistence
            LOGGER.debug("failed to persist thresholds for %s", bot_name, exc_info=True)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("internalize_coding_bot")
        internalize_start = perf_counter()
        try:
            LOGGER.info(
                "before internalize_coding_bot (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            manager = internalize_coding_bot(
                bot_name,
                engine,
                pipeline,
                data_bot=data_bot,
                bot_registry=registry,
                threshold_service=ThresholdService(),
                roi_threshold=thresholds.roi_drop,
                error_threshold=thresholds.error_increase,
                test_failure_threshold=thresholds.test_failure_increase,
            )
        except Exception:
            LOGGER.exception("internalize_coding_bot failed (step=internalize_coding_bot)")
            raise
        LOGGER.info(
            "after internalize_coding_bot (last_step=%s)", BOOTSTRAP_PROGRESS["last_step"]
        )
        _log_step("internalize_coding_bot", internalize_start)
        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("promote_pipeline")
        promote_start = perf_counter()
        promote_timeout = _resolve_step_timeout(
            step_name="promote_pipeline", vector_heavy=vector_heavy
        )
        try:
            LOGGER.info(
                "starting promote_pipeline (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            _run_with_timeout(
                _timed_callable,
                timeout=promote_timeout,
                bootstrap_deadline=bootstrap_deadline,
                description="promote_pipeline",
                abort_on_timeout=True,
                heavy_bootstrap=heavy_bootstrap,
                budget=shared_timeout_coordinator,
                budget_label="db_indexes",
                func=promote_pipeline,
                label="promote_pipeline",
                manager=manager,
            )
        except Exception:
            LOGGER.exception("promote_pipeline failed (step=promote_pipeline)")
            raise
        LOGGER.info(
            "promote_pipeline finished (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        _log_step("promote_pipeline", promote_start)
        _BOOTSTRAP_SCHEDULER.mark_ready(
            "orchestrator_state", reason="pipeline_promoted"
        )
        _set_component_state("orchestrator_state", "ready")
        _publish_online_state()

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("seed_final_context")
        LOGGER.info(
            "starting _push_bootstrap_context (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        final_push_timeout = _resolve_step_timeout(
            step_name="_push_bootstrap_context", vector_heavy=vector_heavy
        )
        _run_with_timeout(
            _push_bootstrap_context,
            timeout=final_push_timeout,
            bootstrap_deadline=bootstrap_deadline,
            description="_push_bootstrap_context final",
            abort_on_timeout=True,
            heavy_bootstrap=heavy_bootstrap,
            budget=shared_timeout_coordinator,
            budget_label="db_indexes",
            registry=registry,
            data_bot=data_bot,
            manager=manager,
            pipeline=pipeline,
            bootstrap_safe=True,
            bootstrap_fast=True,
        )
        LOGGER.info(
            "_push_bootstrap_context finished (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        LOGGER.info("_push_bootstrap_context completed (step=push_final_context)")
        LOGGER.info(
            "starting _seed_research_aggregator_context (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        final_seed_gate = _BOOTSTRAP_CONTENTION_COORDINATOR.negotiate_step(
            "_seed_research_aggregator_context", vector_heavy=True, heavy=True
        )
        final_seed_timeout = _resolve_step_timeout(
            step_name="_seed_research_aggregator_context",
            vector_heavy=True,
            contention_scale=final_seed_gate["timeout_scale"],
        )
        _run_with_timeout(
            _seed_research_aggregator_context,
            timeout=final_seed_timeout,
            bootstrap_deadline=bootstrap_deadline,
            description="_seed_research_aggregator_context final",
            abort_on_timeout=False,
            heavy_bootstrap=heavy_bootstrap,
            contention_scale=final_seed_gate["timeout_scale"],
            budget=shared_timeout_coordinator,
            budget_label="orchestrator_state",
            registry=registry,
            data_bot=data_bot,
            context_builder=context_builder,
            engine=engine,
            pipeline=pipeline,
            manager=manager,
        )
        LOGGER.info(
            "_seed_research_aggregator_context finished (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        LOGGER.info("_seed_research_aggregator_context completed (step=seed_final)")
        _BOOTSTRAP_SCHEDULER.mark_ready(
            "vector_seeding", reason="seed_final_context"
        )
        _publish_online_state()

        background_gate = _BOOTSTRAP_CONTENTION_COORDINATOR.negotiate_step(
            "background_loops_start", heavy=False, vector_heavy=False
        )
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason="post_core_ready"
        )
        if background_gate.get("parallelism_scale", 1.0) < 1.0:
            os.environ["MENACE_BOOTSTRAP_PARALLELISM_HINT"] = str(
                background_gate["parallelism_scale"]
            )
        LOGGER.info(
            "scheduling deferred background loops after core readiness",
            extra={"gate": background_gate, "event": "background-loops-deferred"},
        )
        _publish_online_state()

        bootstrap_context = {
            "registry": registry,
            "data_bot": data_bot,
            "context_builder": context_builder,
            "engine": engine,
            "pipeline": pipeline,
            "manager": manager,
        }
        online_snapshot = _BOOTSTRAP_SCHEDULER.snapshot()
        bootstrap_context["online_state"] = online_snapshot
        bootstrap_context["online"] = online_snapshot.get("quorum", False)
        bootstrap_context["warming_components"] = BOOTSTRAP_ONLINE_STATE.get("warming", [])
        if use_cache:
            _BOOTSTRAP_CACHE[bot_name] = bootstrap_context
        _mark_bootstrap_step("bootstrap_complete")
        _BOOTSTRAP_SCHEDULER.mark_ready(
            "background_loops", reason="bootstrap_complete"
        )
        _publish_online_state()
        LOGGER.info(
            "bootstrap online state", extra={"online_state": online_snapshot}
        )
        LOGGER.info(
            "initialize_bootstrap_context completed successfully for %s (step=bootstrap_complete)",
            bot_name,
        )
        LOGGER.info(
            "bootstrap marked complete; returning context to caller (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        LOGGER.info("bootstrap complete; returning to caller")
        LOGGER.info(
            "bootstrap return (last_step=%s)", BOOTSTRAP_PROGRESS["last_step"]
        )
        return bootstrap_context


__all__ = ["initialize_bootstrap_context"]
