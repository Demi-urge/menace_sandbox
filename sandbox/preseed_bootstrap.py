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
import math
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

_bootstrap_dependency_broker = getattr(
    _coding_bot_interface, "_bootstrap_dependency_broker", lambda: lambda *_, **__: None
)
_current_bootstrap_context = getattr(_coding_bot_interface, "_current_bootstrap_context", lambda: None)
_pop_bootstrap_context = getattr(_coding_bot_interface, "_pop_bootstrap_context", lambda *_, **__: None)
_push_bootstrap_context = getattr(_coding_bot_interface, "_push_bootstrap_context", lambda *_, **__: None)
advertise_bootstrap_placeholder = getattr(
    _coding_bot_interface, "advertise_bootstrap_placeholder", lambda **_: (object(), object())
)
fallback_helper_manager = getattr(
    _coding_bot_interface, "fallback_helper_manager", contextlib.nullcontext
)
prepare_pipeline_for_bootstrap = getattr(
    _coding_bot_interface, "prepare_pipeline_for_bootstrap", lambda *_, **__: None
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
_BOOTSTRAP_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_BOOTSTRAP_EMBEDDER_READY = threading.Event()
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
        self._embedder_deferred: bool = False
        self._embedder_deferral_reason: str | None = None
        self._embedder_background_future: Future | None = None
        self._embedder_background_reason: str | None = None
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
            "embedder_deferred": self._embedder_deferred,
            "embedder_deferral_reason": self._embedder_deferral_reason,
            "embedder_background_enqueued": self._embedder_background_future is not None,
            "embedder_background_reason": self._embedder_background_reason,
        }

    def mark_embedder_deferred(self, *, reason: str | None = None) -> None:
        self._embedder_deferred = True
        self._embedder_deferral_reason = reason or self._embedder_deferral_reason

    def clear_embedder_deferral(self) -> None:
        self._embedder_deferred = False
        self._embedder_deferral_reason = None

    def embedder_deferral(self) -> tuple[bool, str | None]:
        return self._embedder_deferred, self._embedder_deferral_reason

    def schedule_embedder_background(
        self, worker: Callable[[], Any], *, reason: str | None = None
    ) -> Future | None:
        if self._embedder_background_future is not None:
            LOGGER.debug(
                "embedder background already enqueued",  # pragma: no cover - telemetry
                extra={
                    "reason": self._embedder_background_reason,
                    "requested_reason": reason,
                },
            )
            return self._embedder_background_future

        self._embedder_background_reason = reason
        try:
            future = _BOOTSTRAP_BACKGROUND_EXECUTOR.submit(worker)
        except Exception:
            LOGGER.debug("failed to enqueue embedder background worker", exc_info=True)
            return None

        self._embedder_background_future = future
        LOGGER.info(
            "embedder background preload enqueued",  # pragma: no cover - telemetry
            extra={
                "reason": reason,
                "event": "embedder-background-enqueue",
            },
        )
        return future


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
        self._stage_deferred_reason: dict[str, str] = {}
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

    def defer_step(self, step_name: str, *, reason: str | None = None) -> None:
        stage = stage_for_step(step_name)
        if not stage:
            return

        remaining = self._remaining_steps.get(stage)
        if remaining is not None:
            remaining.discard(step_name)

        _BOOTSTRAP_SCHEDULER.mark_partial(
            stage, reason=reason or f"{step_name}_deferred"
        )
        if reason:
            self._stage_deferred_reason.setdefault(stage, reason)
        _set_component_state(stage, "deferred")

        if self._coordinator:
            self._coordinator.mark_component_state(stage, "running")

        self._emit_stage_signal(stage, step_name, "deferred")

    def stage_budget(self, *, step_name: str | None = None, stage: str | None = None) -> float | None:
        target_stage = stage or (stage_for_step(step_name) if step_name else None)
        if target_stage is None:
            return None

        window = self._stage_windows.get(target_stage, {}) if self._stage_windows else {}
        remaining = window.get("remaining")
        budget = window.get("budget")
        try:
            if remaining is not None:
                return float(remaining)
        except (TypeError, ValueError):
            LOGGER.debug("invalid remaining budget for stage", exc_info=True)
        try:
            if budget is not None:
                return float(budget)
        except (TypeError, ValueError):
            LOGGER.debug("invalid stage budget", exc_info=True)

        entry = self._stage_policy.get(target_stage, {}) if self._stage_policy else {}
        if isinstance(entry, Mapping):
            for key in ("deadline", "scaled_budget", "soft_budget", "budget"):
                candidate = entry.get(key)
                try:
                    if candidate is not None:
                        return float(candidate)
                except (TypeError, ValueError):
                    continue
        return None

    def stage_deadline(
        self, *, step_name: str | None = None, stage: str | None = None
    ) -> float | None:
        target_stage = stage or (stage_for_step(step_name) if step_name else None)
        if target_stage is None:
            return None

        window = self._stage_windows.get(target_stage, {}) if self._stage_windows else {}
        deadline = window.get("deadline")
        if deadline is not None:
            try:
                return float(deadline)
            except (TypeError, ValueError):
                LOGGER.debug("invalid stage deadline", exc_info=True)

        entry = self._stage_policy.get(target_stage, {}) if self._stage_policy else {}
        if isinstance(entry, Mapping):
            candidate = entry.get("deadline")
            try:
                if candidate is not None:
                    return float(candidate)
            except (TypeError, ValueError):
                LOGGER.debug("invalid stage deadline policy", exc_info=True)
        return None

    def deferred_reason(
        self, *, step_name: str | None = None, stage: str | None = None
    ) -> str | None:
        target_stage = stage or (stage_for_step(step_name) if step_name else None)
        if target_stage is None:
            return None
        return self._stage_deferred_reason.get(target_stage)



def _default_step_floor(*, vector_heavy: bool = False) -> float:
    return _BOOTSTRAP_SCHEDULER.default_budget(vector_heavy=vector_heavy)


def _set_component_state(component: str, state: str) -> None:
    BOOTSTRAP_ONLINE_STATE.setdefault("components", {})[component] = state
    BOOTSTRAP_ONLINE_STATE["warming"] = [
        name for name, status in BOOTSTRAP_ONLINE_STATE["components"].items() if status != "ready"
    ]
    _publish_online_state()


def _derive_warmup_join_timeout(
    *,
    warmup_timebox_cap: float | None,
    enforced_timebox: float | None,
    warmup_started: float,
    stage_guard_timebox: float | None,
    embedder_stage_budget_hint: float | None,
    warmup_hard_cap: float | None,
    warmup_join_ceiling: float | None = None,
) -> tuple[float, float | None]:
    """Determine a finite join timeout for embedder warmup threads."""

    warmup_join_cap = (
        warmup_join_ceiling
        if warmup_join_ceiling is not None
        else BOOTSTRAP_EMBEDDER_WARMUP_JOIN_CAP
    )
    if warmup_join_cap is not None and warmup_join_cap <= 0:
        warmup_join_cap = None

    strict_warmup_remaining = None
    if warmup_timebox_cap is not None:
        strict_warmup_remaining = max(
            0.0, warmup_timebox_cap - (time.monotonic() - warmup_started)
        )

    warmup_budget_remaining = None
    if enforced_timebox is not None:
        warmup_budget_remaining = max(
            0.0, enforced_timebox - (time.monotonic() - warmup_started)
        )

    join_timeout = strict_warmup_remaining
    join_cap = warmup_timebox_cap if warmup_timebox_cap is not None else enforced_timebox
    if join_timeout is None:
        join_timeout = warmup_budget_remaining

    if join_timeout is None:
        fallback_candidates = [
            candidate
            for candidate in (
                stage_guard_timebox,
                embedder_stage_budget_hint,
                warmup_hard_cap,
            )
            if candidate is not None and candidate > 0
        ]
        join_cap = min(fallback_candidates) if fallback_candidates else 1.0
        join_timeout = max(0.0, join_cap)
    else:
        join_cap = join_cap if join_cap is not None else join_timeout

    if warmup_join_cap is not None:
        join_cap = min(join_cap, warmup_join_cap) if join_cap is not None else warmup_join_cap
        join_timeout = (
            min(join_timeout, warmup_join_cap)
            if join_timeout is not None
            else warmup_join_cap
        )
    elif join_timeout is None and join_cap is not None:
        join_timeout = join_cap

    return join_timeout, join_cap


def _embedder_presence_policy(
    *,
    gate_constrained: bool,
    stage_budget: float | None,
    embedder_timeout: float | None,
    force_full_preload: bool,
    fast_or_lite: bool,
    embedder_deferred: bool,
    bootstrap_fast_context: bool,
    warmup_lite_context: bool,
    bootstrap_guard_active: bool,
) -> tuple[bool, str, bool]:
    """Compute whether embedder preload should fall back to a presence probe.

    Returns a tuple of ``(presence_only, reason, budget_guarded)`` to help upstream
    callers annotate state for warmup flows.
    """

    budget_guarded = False
    low_budget_guard = False
    default_presence = not force_full_preload
    guard_presence = bootstrap_guard_active and not force_full_preload
    if stage_budget is not None:
        budget_guarded = stage_budget < 5.0
        tight_budget_threshold = embedder_timeout if embedder_timeout is not None else 0.0
        tight_budget_threshold = max(3.0, min(tight_budget_threshold * 0.75, 15.0))
        low_budget_guard = stage_budget < tight_budget_threshold

    presence_only = default_presence or guard_presence or fast_or_lite or gate_constrained or embedder_deferred
    presence_only = presence_only or budget_guarded or low_budget_guard

    presence_reason = "embedder_presence_default" if default_presence else "embedder_presence_probe"
    if guard_presence:
        presence_reason = "embedder_presence_bootstrap_guard"
    elif gate_constrained:
        presence_reason = "embedder_preload_guarded"
    elif budget_guarded:
        presence_reason = "embedder_budget_guarded"
    elif low_budget_guard:
        presence_reason = "embedder_budget_tight"
    elif fast_or_lite and not force_full_preload:
        presence_reason = (
            "bootstrap_fast_embedder_probe"
            if bootstrap_fast_context
            else "warmup_lite_embedder_probe"
        )

    if force_full_preload and not presence_only:
        presence_reason = "embedder_force_preload"

    return presence_only, presence_reason, budget_guarded or low_budget_guard


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
BOOTSTRAP_EMBEDDER_WARMUP_CAP = float(os.getenv("BOOTSTRAP_EMBEDDER_WARMUP_CAP", "90.0"))
BOOTSTRAP_EMBEDDER_FALLBACK_WARMUP_CAP = float(
    os.getenv("BOOTSTRAP_EMBEDDER_FALLBACK_WARMUP_CAP", "12.0")
)
BOOTSTRAP_EMBEDDER_WARMUP_JOIN_CAP = float(
    os.getenv("BOOTSTRAP_EMBEDDER_WARMUP_JOIN_CAP", "30.0")
)
SELF_CODING_MIN_REMAINING_BUDGET = float(
    os.getenv("SELF_CODING_MIN_REMAINING_BUDGET", "35.0")
)
BOOTSTRAP_DEADLINE_BUFFER = 5.0
_BOOTSTRAP_EMBEDDER_DISABLED = False
_BOOTSTRAP_EMBEDDER_STARTED = False
_BOOTSTRAP_EMBEDDER_ATTEMPTED = False
_BOOTSTRAP_EMBEDDER_JOB: dict[str, Any] | None = None


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
    budget: SharedTimeoutCoordinator | None = None,
    budget_label: str | None = None,
    presence_probe: bool = False,
    presence_reason: str | None = None,
    presence_only_guard: bool = False,
    bootstrap_fast: bool = False,
    schedule_background: bool = False,
    force_placeholder: bool = False,
    bootstrap_deadline: float | None = None,
    precomputed_caps: Mapping[str, float | None] | None = None,
) -> None:
    """Attempt to initialise the shared embedder without blocking bootstrap."""

    global _BOOTSTRAP_EMBEDDER_DISABLED, _BOOTSTRAP_EMBEDDER_STARTED, _BOOTSTRAP_EMBEDDER_ATTEMPTED
    global _BOOTSTRAP_EMBEDDER_JOB

    if timeout <= 0:
        LOGGER.info("bootstrap embedder timeout disabled; skipping embedder preload")
        return

    if _BOOTSTRAP_EMBEDDER_DISABLED:
        LOGGER.info("embedder preload disabled for this bootstrap run; skipping")
        return

    if _BOOTSTRAP_EMBEDDER_ATTEMPTED:
        LOGGER.debug("embedder preload already attempted; refusing to create another thread")
        if _BOOTSTRAP_EMBEDDER_JOB:
            return _BOOTSTRAP_EMBEDDER_JOB.get("result", _BOOTSTRAP_PLACEHOLDER)
        return

    if _BOOTSTRAP_EMBEDDER_STARTED:
        LOGGER.debug("embedder preload already started; refusing to create another thread")
        return

    existing_job = _BOOTSTRAP_EMBEDDER_JOB or {}
    if (
        not schedule_background
        and existing_job.get("awaiting_full_preload")
        and existing_job.get("ready_after_bootstrap")
        and not existing_job.get("ready")
    ):
        LOGGER.debug(
            "embedder preload already scheduled for background; returning placeholder"
        )
        return existing_job.get("result", existing_job.get("placeholder", _BOOTSTRAP_PLACEHOLDER))

    budget_exhausted = stage_budget is not None and stage_budget <= 0
    if budget_exhausted:
        LOGGER.warning(
            "embedder preload skipped because stage budget is exhausted (stage_budget=%.2fs)",
            stage_budget,
        )

    try:
        from menace_sandbox.governed_embeddings import (
            _MAX_EMBEDDER_WAIT,
            _activate_bundled_fallback,
            apply_bootstrap_timeout_caps,
            cancel_embedder_initialisation,
            embedder_cache_present,
            get_embedder,
        )
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.debug("governed_embeddings unavailable; skipping embedder bootstrap", exc_info=True)
        return

    caps: dict[str, float | None] = dict(precomputed_caps or {})
    stage_budget_hint = caps.get("stage_budget_hint")
    result: Dict[str, Any] = {}
    embedder_stop_event = threading.Event()
    now = time.monotonic()
    stage_deadline_hint = caps.get("stage_deadline")
    stage_deadline_remaining = (
        max(0.0, stage_deadline_hint - now) if stage_deadline_hint is not None else None
    )
    remaining_bootstrap_window = (
        max(0.0, bootstrap_deadline - now) if bootstrap_deadline is not None else None
    )
    timeout_cap = apply_bootstrap_timeout_caps(stage_budget)
    warmup_cap = BOOTSTRAP_EMBEDDER_WARMUP_CAP if BOOTSTRAP_EMBEDDER_WARMUP_CAP > 0 else None
    max_duration_cap = min(
        (
            cap
            for cap in (
                stage_budget if stage_budget is not None and stage_budget > 0 else None,
                stage_budget_hint
                if stage_budget_hint is not None and stage_budget_hint > 0
                else None,
                stage_deadline_remaining
                if stage_deadline_remaining is not None and stage_deadline_remaining > 0
                else None,
                remaining_bootstrap_window
                if remaining_bootstrap_window is not None and remaining_bootstrap_window > 0
                else None,
                BOOTSTRAP_EMBEDDER_TIMEOUT if BOOTSTRAP_EMBEDDER_TIMEOUT > 0 else None,
            )
            if cap is not None
        ),
        default=None,
    )
    strict_timeout_candidates = [
        stage_budget if stage_budget is not None and stage_budget > 0 else None,
        stage_budget_hint if stage_budget_hint is not None and stage_budget_hint > 0 else None,
        BOOTSTRAP_EMBEDDER_TIMEOUT if BOOTSTRAP_EMBEDDER_TIMEOUT > 0 else None,
        warmup_cap,
        stage_deadline_remaining if stage_deadline_remaining is not None else None,
        remaining_bootstrap_window if remaining_bootstrap_window is not None else None,
    ]
    strict_timeout_cap = min(
        (candidate for candidate in strict_timeout_candidates if candidate is not None),
        default=None,
    )
    effective_timeout_cap = min(
        (
            candidate
            for candidate in (timeout_cap, strict_timeout_cap)
            if candidate is not None and candidate > 0
        ),
        default=strict_timeout_cap,
    )

    if timeout is None:
        timeout = effective_timeout_cap
    elif timeout > 0 and effective_timeout_cap is not None:
        timeout = min(timeout, effective_timeout_cap)
    elif timeout < 0 and effective_timeout_cap is not None:
        timeout = effective_timeout_cap
    elif timeout < 0:
        timeout = 0.0

    bootstrap_timeout = stage_budget
    if bootstrap_timeout is None:
        bootstrap_timeout = effective_timeout_cap
    elif bootstrap_timeout > 0 and effective_timeout_cap is not None:
        bootstrap_timeout = min(bootstrap_timeout, effective_timeout_cap)
    if remaining_bootstrap_window is not None:
        if bootstrap_timeout is None:
            bootstrap_timeout = remaining_bootstrap_window
        elif bootstrap_timeout > 0:
            bootstrap_timeout = min(bootstrap_timeout, remaining_bootstrap_window)

    stage_wall_cap = (
        effective_timeout_cap
        if effective_timeout_cap is not None and effective_timeout_cap > 0
        else None
    )
    stage_wall_cap_hint = caps.get("stage_wall_cap")
    if stage_wall_cap_hint is not None and stage_wall_cap_hint >= 0:
        stage_wall_cap = (
            stage_wall_cap_hint if stage_wall_cap is None else min(stage_wall_cap, stage_wall_cap_hint)
        )
    if remaining_bootstrap_window is not None:
        stage_wall_cap = (
            remaining_bootstrap_window
            if stage_wall_cap is None
            else min(stage_wall_cap, remaining_bootstrap_window)
        )
    if stage_deadline_remaining is not None:
        stage_wall_cap = (
            stage_deadline_remaining
            if stage_wall_cap is None
            else min(stage_wall_cap, stage_deadline_remaining)
        )
    if max_duration_cap is not None:
        if stage_wall_cap is None:
            stage_wall_cap = max_duration_cap
        else:
            stage_wall_cap = min(stage_wall_cap, max_duration_cap)

        if timeout is None or timeout <= 0:
            timeout = max_duration_cap
        elif timeout > 0:
            timeout = min(timeout, max_duration_cap)

    try:
        placeholder_candidate = _activate_bundled_fallback("bootstrap_placeholder")
    except Exception:  # pragma: no cover - advisory only
        LOGGER.debug("failed to activate embedder placeholder", exc_info=True)
        placeholder_candidate = None
    placeholder = placeholder_candidate or _BOOTSTRAP_PLACEHOLDER

    if stop_event is not None and stop_event.is_set():
        _BOOTSTRAP_EMBEDDER_JOB = {
            "result": placeholder,
            "placeholder": placeholder,
            "placeholder_reason": "bootstrap_stop_signal",
            "ready": True,
        }
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason="bootstrap_stop_signal")
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason="embedder_placeholder:bootstrap_stop_signal"
        )
        return placeholder
    _BOOTSTRAP_EMBEDDER_STARTED = True
    start_time = perf_counter()
    stage_deadline_perf = (
        start_time + stage_deadline_remaining if stage_deadline_remaining is not None else None
    )
    presence_cap = caps.get("presence_cap")
    if presence_cap is None:
        presence_cap = float(os.getenv("BOOTSTRAP_EMBEDDER_PRESENCE_CAP", "0.75"))
    if presence_cap < 0:
        presence_cap = 0.0
    caps["presence_cap"] = presence_cap
    presence_deadline = (
        start_time + presence_cap if presence_cap and presence_cap > 0 else None
    )
    tight_cap_for_presence = (
        max_duration_cap is not None
        and presence_cap > 0
        and max_duration_cap <= presence_cap
        and not presence_probe
    )
    if tight_cap_for_presence:
        presence_probe = True
        schedule_background = True
        presence_reason = presence_reason or "embedder_max_duration_guard"
    if stage_wall_cap is not None:
        cap_deadline = start_time + stage_wall_cap
        presence_deadline = cap_deadline if presence_deadline is None else min(
            presence_deadline, cap_deadline
        )

    caps.update(
        {
            "bootstrap_deadline": bootstrap_deadline,
            "timeout": timeout,
            "stage_budget": stage_budget,
            "stage_budget_hint": stage_budget_hint,
            "warmup_cap": warmup_cap,
            "max_duration_cap": max_duration_cap,
            "effective_timeout_cap": effective_timeout_cap,
            "bootstrap_timeout": bootstrap_timeout,
            "stage_wall_cap": stage_wall_cap,
            "stage_deadline": stage_deadline_hint,
        }
    )

    _ensure_not_stopped(stop_event)
    _BOOTSTRAP_EMBEDDER_ATTEMPTED = True

    if _BOOTSTRAP_EMBEDDER_JOB and _BOOTSTRAP_EMBEDDER_JOB.get("thread"):
        existing = _BOOTSTRAP_EMBEDDER_JOB["thread"]
        existing_stop_event = _BOOTSTRAP_EMBEDDER_JOB.get("stop_event")
        if existing.is_alive():
            cancel_caps = [
                candidate
                for candidate in (
                    stage_budget if stage_budget and stage_budget > 0 else None,
                    warmup_cap,
                    effective_timeout_cap,
                )
                if candidate is not None and candidate > 0
            ]
            cancel_timeout = min(cancel_caps) if cancel_caps else 0.0
            LOGGER.info(
                "embedder warmup already running; cancelling stalled download",
                extra={
                    "cancel_timeout": cancel_timeout,
                    "stage_budget": stage_budget,
                    "warmup_cap": warmup_cap,
                },
            )
            try:
                cancel_embedder_initialisation(
                    existing_stop_event or embedder_stop_event,
                    reason="embedder_preexisting_download_cancelled",
                    join_timeout=cancel_timeout,
                )
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("failed to cancel running embedder warmup", exc_info=True)
            try:
                existing.join(cancel_timeout)
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("join on stalled embedder warmup failed", exc_info=True)
            _BOOTSTRAP_EMBEDDER_JOB = None
            _BOOTSTRAP_EMBEDDER_STARTED = False
            return placeholder
    _BOOTSTRAP_EMBEDDER_JOB = {
        "thread": None,
        "stop_event": embedder_stop_event,
        "placeholder": placeholder,
        "started_at": start_time,
        "result": None,
    }

    if stop_event is not None:
        def _propagate_stop_event() -> None:
            stop_event.wait()
            if stop_event.is_set():
                embedder_stop_event.set()

        threading.Thread(
            target=_propagate_stop_event, name="embedder-external-stop", daemon=True
        ).start()
    _BOOTSTRAP_SCHEDULER.mark_partial("background_loops", reason="embedder_warmup_start")

    if budget_exhausted:
        result["embedder"] = placeholder
        result["placeholder_reason"] = "stage_budget_exhausted"
        result["deferred"] = True
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason="stage_budget_exhausted")
        _BOOTSTRAP_EMBEDDER_JOB.update(
            {
                "result": placeholder,
                "placeholder_reason": "stage_budget_exhausted",
                "deferred": True,
            }
        )
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason="embedder_placeholder:stage_budget_exhausted"
        )
        _finalize_embedder_job(
            placeholder,
            placeholder_reason="stage_budget_exhausted",
            aborted=True,
            deferred=True,
        )
        return placeholder

    if presence_only_guard:
        guard_reason = presence_reason or "embedder_presence_guard"
        result.update(
            {
                "embedder": placeholder,
                "placeholder_reason": guard_reason,
                "deferred": True,
            }
        )
        _BOOTSTRAP_EMBEDDER_JOB.update(
            {
                "result": placeholder,
                "placeholder_reason": guard_reason,
                "deferred": True,
                "warmup_placeholder_reason": guard_reason,
                "computed_caps": caps,
            }
        )
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=guard_reason)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{guard_reason}"
        )
        LOGGER.info(
            "embedder warmup guarded; returning placeholder and deferring",  # pragma: no cover - telemetry
            extra={
                "stage_budget": stage_budget,
                "bootstrap_deadline": bootstrap_deadline,
                "presence_cap": presence_cap,
                "guard_reason": guard_reason,
            },
        )
        _finalize_embedder_job(
            placeholder,
            placeholder_reason=guard_reason,
            aborted=False,
            deferred=True,
        )
        return placeholder

    def _signal_stop(reason: str) -> None:
        LOGGER.debug("signalling embedder stop (%s)", reason)
        embedder_stop_event.set()

    background_download_requested = schedule_background or force_placeholder
    background_budget_available = stage_budget is None or stage_budget > 0

    def _enqueue_background_download() -> None:
        if not background_budget_available:
            LOGGER.debug("embedder background download blocked by stage budget")
            return

        if presence_probe and not background_download_requested:
            LOGGER.debug(
                "presence probe requested; skipping embedder background download"
            )
            return

        if _BOOTSTRAP_EMBEDDER_JOB and _BOOTSTRAP_EMBEDDER_JOB.get("background_scheduled"):
            LOGGER.debug("embedder background download already scheduled")
            return

        if _BOOTSTRAP_EMBEDDER_JOB is not None:
            _BOOTSTRAP_EMBEDDER_JOB["background_scheduled"] = True
            _BOOTSTRAP_EMBEDDER_JOB["deferred"] = True
            _BOOTSTRAP_EMBEDDER_JOB["awaiting_full_preload"] = True

        def _background_worker() -> None:
            def _remaining_budget() -> tuple[float | None, str | None]:
                deadlines: list[tuple[float, str]] = []
                if budget_deadline is not None:
                    deadlines.append((budget_deadline, "bootstrap_budget_exceeded"))
                if budget_deadline_hint is not None:
                    deadlines.append((budget_deadline_hint, "embedder_stage_budget_deadline"))
                if wall_clock_deadline is not None:
                    deadlines.append((wall_clock_deadline, "bootstrap_wall_clock_exceeded"))
                if bootstrap_wall_deadline is not None:
                    deadlines.append((bootstrap_wall_deadline, "bootstrap_deadline_exceeded"))
                if stage_deadline_perf is not None:
                    deadlines.append((stage_deadline_perf, "embedder_stage_deadline_exhausted"))
                if max_wait_deadline is not None:
                    deadlines.append((max_wait_deadline, max_wait_reason))
                if presence_deadline is not None:
                    deadlines.append((presence_deadline, "embedder_presence_timeout"))

                if not deadlines:
                    return None, None

                deadline, reason = min(deadlines, key=lambda item: item[0])
                return deadline - perf_counter(), reason

            def _abort_background(reason: str) -> None:
                result["deferred"] = True
                _signal_stop(reason)
                try:
                    cancel_embedder_initialisation(
                        embedder_stop_event, reason=reason, join_timeout=0.0
                    )
                except Exception:  # pragma: no cover - diagnostics only
                    LOGGER.debug(
                        "embedder background cancel failed", exc_info=True
                    )

                result.setdefault("embedder", placeholder)
                result.setdefault("placeholder_reason", reason)
                if _BOOTSTRAP_EMBEDDER_JOB is not None:
                    _BOOTSTRAP_EMBEDDER_JOB["result"] = placeholder
                    _BOOTSTRAP_EMBEDDER_JOB["placeholder_reason"] = reason
                    _BOOTSTRAP_EMBEDDER_JOB["deferred"] = True
                _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=reason)
                _BOOTSTRAP_SCHEDULER.mark_partial(
                    "background_loops", reason=f"embedder_placeholder:{reason}"
                )
                _finalize_embedder_job(
                    placeholder,
                    placeholder_reason=reason,
                    aborted=True,
                    deferred=True,
                )

            def _budget_guard() -> None:
                while not embedder_stop_event.is_set():
                    remaining, reason = _remaining_budget()
                    if remaining is None:
                        return
                    if remaining <= 0:
                        _abort_background(reason or "embedder_background_timeout")
                        return
                    if embedder_stop_event.wait(min(remaining, 0.1)):
                        return

            threading.Thread(
                target=_budget_guard, name="embedder-background-guard", daemon=True
            ).start()

            try:
                ready_timeout = (
                    stage_wall_cap if stage_wall_cap is not None else max_duration_cap
                )
                remaining_budget, remaining_reason = _remaining_budget()
                if remaining_budget is not None and remaining_budget < 0:
                    _abort_background(remaining_reason or "embedder_background_timeout")
                    return
                if ready_timeout is None:
                    _BOOTSTRAP_EMBEDDER_READY.wait()
                else:
                    ready = _BOOTSTRAP_EMBEDDER_READY.wait(ready_timeout)
                    if not ready:
                        LOGGER.debug(
                            "embedder background preload skipped after readiness wait timeout"
                        )
                        return
                with contextlib.suppress(Exception):
                    os.nice(5)
                remaining_budget, remaining_reason = _remaining_budget()
                embedder_timeout = _MAX_EMBEDDER_WAIT
                if remaining_budget is not None:
                    if remaining_budget <= 0:
                        _abort_background(
                            remaining_reason or "embedder_background_timeout"
                        )
                        return
                    embedder_timeout = min(embedder_timeout, max(remaining_budget, 0.0))

                embedder_obj = get_embedder(
                    timeout=embedder_timeout,
                    stop_event=embedder_stop_event,
                    bootstrap_timeout=bootstrap_timeout,
                    bootstrap_mode=True,
                )
                if _BOOTSTRAP_EMBEDDER_JOB is not None:
                    _BOOTSTRAP_EMBEDDER_JOB["result"] = embedder_obj
                    _BOOTSTRAP_EMBEDDER_JOB.setdefault(
                        "placeholder_reason", "background_download"
                    )
            except Exception:
                LOGGER.debug("background embedder download failed", exc_info=True)
                if embedder_stop_event.is_set():
                    _abort_background("embedder_background_cancelled")
            
        future = _BOOTSTRAP_BACKGROUND_EXECUTOR.submit(_background_worker)
        if _BOOTSTRAP_EMBEDDER_JOB is not None:
            _BOOTSTRAP_EMBEDDER_JOB["background_future"] = future

    def _finalize_embedder_job(
        embedder_obj: Any,
        *,
        placeholder_reason: str | None = None,
        aborted: bool = False,
        deferred: bool = False,
    ) -> None:
        global _BOOTSTRAP_EMBEDDER_JOB, _BOOTSTRAP_EMBEDDER_STARTED

        job = _BOOTSTRAP_EMBEDDER_JOB or {}
        job["result"] = embedder_obj
        job["placeholder"] = job.get("placeholder", placeholder)
        job["ready"] = not aborted or bool(placeholder_reason)
        if placeholder_reason:
            job["placeholder_reason"] = placeholder_reason
        if aborted:
            job["aborted"] = True
        if deferred:
            job["deferred"] = True
        if not aborted and not deferred and not placeholder_reason:
            _BOOTSTRAP_SCHEDULER.clear_embedder_deferral()
        job.pop("thread", None)
        job.pop("stop_event", None)
        _BOOTSTRAP_EMBEDDER_JOB = job
        _BOOTSTRAP_EMBEDDER_STARTED = False

        if job.get("ready"):
            _BOOTSTRAP_SCHEDULER.mark_ready(
                "background_loops", reason=placeholder_reason or "embedder_ready"
            )
            _set_component_state("vector_seeding", "ready")

    def _record_abort(
        reason: str, *, deferred: bool = False, schedule_background: bool = False
    ) -> None:
        nonlocal placeholder
        global _BOOTSTRAP_EMBEDDER_DISABLED
        result["aborted"] = True
        result["deferred"] = deferred
        _signal_stop(reason)
        if deferred:
            _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=reason)
        elapsed = perf_counter() - start_time
        abort_metadata = {
            "elapsed": round(elapsed, 3),
            "stage_budget": stage_budget,
            "max_wait": _MAX_EMBEDDER_WAIT,
            "reason": reason,
        }
        if budget_label and budget:
            try:
                budget.record_progress(
                    budget_label,
                    elapsed=elapsed,
                    remaining=0.0,
                    metadata={"abort_reason": reason},
                )
                budget.mark_component_state(budget_label, "blocked")
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("failed to publish embedder abort to budget", exc_info=True)
        try:
            promoted = _activate_bundled_fallback(reason)
            if promoted:
                placeholder = _BOOTSTRAP_EMBEDDER_JOB.get("placeholder", placeholder)
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("failed to promote fallback embedder", exc_info=True)
        result.setdefault("embedder", placeholder)
        result.setdefault("placeholder_reason", reason)
        _BOOTSTRAP_EMBEDDER_JOB["result"] = placeholder
        if deferred:
            _BOOTSTRAP_EMBEDDER_JOB["deferred"] = True
            if schedule_background:
                _enqueue_background_download()
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{reason}"
        )
        cancel_embedder_initialisation(
            embedder_stop_event,
            reason=reason,
            join_timeout=0.25,
        )
        joined = False
        still_running = False
        if thread is not None:
            try:
                thread.join(0.1)
                joined = True
                still_running = thread.is_alive()
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("embedder warmup thread join failed", exc_info=True)
        abort_metadata["thread_joined"] = joined
        abort_metadata["thread_alive"] = still_running
        LOGGER.warning(
            "background embedder warmup aborted", extra=abort_metadata
        )
        _BOOTSTRAP_EMBEDDER_DISABLED = True
        _finalize_embedder_job(
            placeholder,
            placeholder_reason=reason,
            aborted=True,
            deferred=deferred,
        )

    tight_budget_threshold = float(
        os.getenv("BOOTSTRAP_EMBEDDER_DEFERRAL_THRESHOLD", "1.0")
    )
    tight_budget = False
    if tight_budget_threshold > 0:
        tight_budget = any(
            candidate is not None and candidate < tight_budget_threshold
            for candidate in (stage_budget, warmup_cap)
        )

    cache_available = False
    try:
        cache_available = bool(embedder_cache_present())
    except Exception:  # pragma: no cover - diagnostics only
        LOGGER.debug("embedder cache presence probe failed", exc_info=True)

    deferral_reason = None
    if not presence_probe:
        if cache_available:
            deferral_reason = "embedder_cache_present"
        elif tight_budget:
            deferral_reason = "embedder_budget_short_circuit"

    if deferral_reason:
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=deferral_reason)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{deferral_reason}"
        )
        placeholder_path = summarise_value(
            getattr(placeholder, "model_path", None)
            or getattr(placeholder, "path", None)
            or getattr(placeholder, "name", None)
            or placeholder
        )
        result.update(
            {
                "embedder": placeholder,
                "placeholder_reason": deferral_reason,
                "placeholder_path": placeholder_path,
                "deferred": True,
                "presence_available": cache_available,
            }
        )
        _BOOTSTRAP_EMBEDDER_JOB.update(
            {
                "result": placeholder,
                "placeholder_reason": deferral_reason,
                "placeholder_path": placeholder_path,
                "deferred": True,
                "presence_available": cache_available,
                "background_enqueue_reason": deferral_reason,
            }
        )
        LOGGER.info(
            "embedder warmup deferred prior to thread launch",  # pragma: no cover - telemetry
            extra={
                "stage_budget": stage_budget,
                "warmup_cap": warmup_cap,
                "deferral_reason": deferral_reason,
                "placeholder_path": placeholder_path,
                "cache_available": cache_available,
                "deferral_threshold": tight_budget_threshold,
            },
        )
        _enqueue_background_download()
        _finalize_embedder_job(
            placeholder,
            placeholder_reason=deferral_reason,
            aborted=False,
            deferred=True,
        )
        return placeholder

    if presence_probe:
        probe_reason = presence_reason or "embedder_presence_probe"
        budget_deadline = (
            start_time + stage_budget if stage_budget is not None and stage_budget > 0 else None
        )
        if budget_deadline is None and stage_budget_hint is not None and stage_budget_hint > 0:
            budget_deadline = start_time + stage_budget_hint
        try:
            cache_available = embedder_cache_present()
        except Exception:  # pragma: no cover - diagnostics only
            cache_available = False
            LOGGER.debug("embedder presence probe failed", exc_info=True)

        if budget_deadline is not None and perf_counter() >= budget_deadline:
            probe_reason = "embedder_presence_budget_exhausted"
            try:
                placeholder = _activate_bundled_fallback(probe_reason) or placeholder
            except Exception:  # pragma: no cover - advisory only
                LOGGER.debug(
                    "failed to promote fallback embedder during presence probe", exc_info=True
                )

        result.update(
            {
                "embedder": placeholder,
                "placeholder_reason": probe_reason,
                "deferred": True,
                "presence_available": cache_available,
            }
        )
        if _BOOTSTRAP_EMBEDDER_JOB is not None:
            _BOOTSTRAP_EMBEDDER_JOB.update(
                {
                    "result": placeholder,
                    "placeholder_reason": probe_reason,
                    "deferred": True,
                    "presence_available": cache_available,
                }
            )
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=probe_reason)
        _BOOTSTRAP_SCHEDULER.mark_ready(
            "background_loops", reason=f"embedder_placeholder:{probe_reason}"
        )
        _set_component_state("vector_seeding", "ready")
        try:
            cancel_embedder_initialisation(
                embedder_stop_event,
                reason=probe_reason,
                join_timeout=0.25,
            )
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.debug("embedder presence probe cancel failed", exc_info=True)
        if background_download_requested:
            _enqueue_background_download()
        _finalize_embedder_job(
            placeholder,
            placeholder_reason=probe_reason,
            aborted=False,
            deferred=True,
        )
        return placeholder

    def _worker() -> None:
        try:
            with contextlib.suppress(Exception):
                os.nice(5)
            result["embedder"] = get_embedder(
                timeout=timeout,
                stop_event=embedder_stop_event,
                bootstrap_timeout=bootstrap_timeout,
                bootstrap_mode=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            result["error"] = exc
        finally:
            elapsed = perf_counter() - start_time
            embedder = result.get("embedder")
            placeholder_reason = result.get("placeholder_reason") or getattr(
                embedder, "_placeholder_reason", None
            )
            if not placeholder_reason and result.get("aborted"):
                placeholder_reason = "aborted"
            if result.get("error") and not placeholder_reason:
                placeholder_reason = "error"
            if embedder is None and placeholder_reason:
                embedder = placeholder
                result["embedder"] = embedder
            elif embedder is None:
                embedder = placeholder
                placeholder_reason = placeholder_reason or "missing"
                result["embedder"] = embedder

            if embedder and not placeholder_reason:
                LOGGER.info(
                    "background embedder warmup completed",
                    extra={
                        "elapsed": round(elapsed, 3),
                        "stage_budget": stage_budget,
                        "max_wait": _MAX_EMBEDDER_WAIT,
                    },
                )
                _BOOTSTRAP_SCHEDULER.mark_ready(
                    "background_loops", reason="embedder_warmup_complete"
                )
            elif placeholder_reason:
                LOGGER.info(
                    "background embedder warmup completed with placeholder",
                    extra={
                        "elapsed": round(elapsed, 3),
                        "stage_budget": stage_budget,
                        "max_wait": _MAX_EMBEDDER_WAIT,
                        "placeholder_reason": placeholder_reason,
                    },
                )
                _BOOTSTRAP_EMBEDDER_JOB["result"] = embedder
                _BOOTSTRAP_SCHEDULER.mark_partial(
                    "background_loops", reason=f"embedder_placeholder:{placeholder_reason}"
                )
            elif result.get("aborted"):
                _BOOTSTRAP_SCHEDULER.mark_partial(
                    "background_loops", reason="embedder_warmup_aborted"
                )
            elif result.get("error"):
                LOGGER.info(
                    "embedder preload failed; proceeding without embeddings",
                    extra={
                        "elapsed": round(elapsed, 3),
                        "stage_budget": stage_budget,
                        "max_wait": _MAX_EMBEDDER_WAIT,
                    },
                )
                LOGGER.debug("embedder preload error", exc_info=result["error"])
                _BOOTSTRAP_EMBEDDER_DISABLED = True
                _BOOTSTRAP_SCHEDULER.mark_partial(
                    "background_loops", reason="embedder_warmup_failed"
                )
            else:
                _BOOTSTRAP_SCHEDULER.mark_partial(
                    "background_loops", reason="embedder_warmup_empty"
                )
            _signal_stop("completed")
            _finalize_embedder_job(
                embedder,
                placeholder_reason=placeholder_reason,
                aborted=result.get("aborted", False),
                deferred=result.get("deferred", False),
            )

    thread = threading.Thread(target=_worker, name="bootstrap-embedder", daemon=True)
    thread.start()
    _BOOTSTRAP_EMBEDDER_JOB["thread"] = thread

    if force_placeholder:
        _BOOTSTRAP_EMBEDDER_JOB["deferred"] = True
        _BOOTSTRAP_EMBEDDER_JOB["placeholder_reason"] = "embedder_tight_budget"
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason="embedder_tight_budget")
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason="embedder_placeholder:embedder_tight_budget"
        )
        return placeholder

    budget_window = stage_budget if stage_budget is not None and stage_budget >= 0 else None
    if stage_budget_hint is not None and stage_budget_hint >= 0:
        budget_window = (
            stage_budget_hint if budget_window is None else min(budget_window, stage_budget_hint)
        )
    if stage_wall_cap is not None and budget_window is not None:
        budget_window = min(stage_wall_cap, budget_window)
    budget_deadline = start_time + budget_window if budget_window is not None else None
    budget_deadline_hint = (
        start_time + stage_budget_hint
        if stage_budget_hint is not None and stage_budget_hint >= 0
        else None
    )
    wall_clock_deadline = (
        start_time + stage_wall_cap if stage_wall_cap is not None else None
    )
    bootstrap_wall_deadline = (
        start_time + remaining_bootstrap_window
        if remaining_bootstrap_window is not None
        else None
    )
    max_wait_cap_candidates = []
    for candidate in (_MAX_EMBEDDER_WAIT, stage_wall_cap, warmup_cap):
        if candidate is None:
            continue
        if candidate >= 0:
            max_wait_cap_candidates.append(candidate)
    max_wait_cap = min(max_wait_cap_candidates) if max_wait_cap_candidates else _MAX_EMBEDDER_WAIT
    max_wait_deadline = start_time + max_wait_cap if max_wait_cap >= 0 else None
    max_wait_reason = "max_wait_exceeded"
    if warmup_cap is not None and warmup_cap >= 0:
        try:
            if abs(max_wait_cap - warmup_cap) < 1e-6:
                max_wait_reason = "warmup_cap_exceeded"
        except TypeError:
            pass
    hard_join_deadline = None
    hard_join_caps: list[float] = []
    for cap_candidate in (warmup_cap, presence_cap):
        if cap_candidate is None:
            continue
        if cap_candidate > 0:
            hard_join_caps.append(cap_candidate)
    if hard_join_caps:
        hard_join_deadline = start_time + min(hard_join_caps)

    hard_timeout_triggered = threading.Event()

    def _hard_timeout_watchdog() -> None:
        if thread is None:
            return
        hard_timeout: float | None = None
        hard_timeout_reason = "embedder_hard_timeout"
        timeout_candidates: list[tuple[float, str]] = []
        if stage_wall_cap is not None and stage_wall_cap > 0:
            timeout_candidates.append((stage_wall_cap, "bootstrap_wall_clock_exceeded"))
        if timeout is not None and timeout > 0:
            timeout_candidates.append((timeout, hard_timeout_reason))
        if timeout_candidates:
            hard_timeout, hard_timeout_reason = min(timeout_candidates, key=lambda item: item[0])

        if hard_timeout is None:
            return

        remaining = (start_time + hard_timeout) - perf_counter()
        if remaining <= 0:
            remaining = 0

        if embedder_stop_event.wait(remaining):
            return

        if thread.is_alive():
            hard_timeout_triggered.set()
            try:
                cancel_embedder_initialisation(
                    embedder_stop_event,
                    reason=hard_timeout_reason,
                    join_timeout=0.0,
                )
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("embedder hard-timeout cancel failed", exc_info=True)
            _record_abort(
                hard_timeout_reason, deferred=True, schedule_background=True
            )

    def _stop_event_watchdog() -> None:
        if stop_event is None:
            return
        stop_event.wait()
        if stop_event.is_set() and (
            thread.is_alive() or not _BOOTSTRAP_EMBEDDER_JOB.get("result")
        ):
            _record_abort(
                "bootstrap_stop_signal", deferred=False, schedule_background=False
            )

    def _budget_watchdog() -> None:
        while thread.is_alive():
            now = perf_counter()
            if budget_deadline is not None and now >= budget_deadline:
                _record_abort(
                    "bootstrap_budget_exceeded",
                    deferred=True,
                    schedule_background=True,
                )
                return
            if budget_deadline_hint is not None and now >= budget_deadline_hint:
                _record_abort(
                    "embedder_stage_budget_deadline",
                    deferred=True,
                    schedule_background=True,
                )
                return
            if wall_clock_deadline is not None and now >= wall_clock_deadline:
                _record_abort(
                    "bootstrap_wall_clock_exceeded",
                    deferred=True,
                    schedule_background=True,
                )
                return
            if stage_deadline_perf is not None and now >= stage_deadline_perf:
                _record_abort(
                    "embedder_stage_deadline_exhausted",
                    deferred=True,
                    schedule_background=True,
                )
                return
            if bootstrap_wall_deadline is not None and now >= bootstrap_wall_deadline:
                _record_abort(
                    "bootstrap_deadline_exceeded",
                    deferred=True,
                    schedule_background=True,
                )
                return
            if max_wait_deadline is not None and now >= max_wait_deadline:
                _record_abort(
                    max_wait_reason,
                    deferred=True,
                    schedule_background=True,
                )
                return
            time.sleep(0.05)

    if not presence_probe:
        threading.Thread(
            target=_hard_timeout_watchdog, name="embedder-hard-timeout", daemon=True
        ).start()
        threading.Thread(
            target=_stop_event_watchdog, name="embedder-stop", daemon=True
        ).start()
        threading.Thread(
            target=_budget_watchdog, name="embedder-budget", daemon=True
        ).start()
    else:
        LOGGER.debug("skipping embedder watchdogs for presence probe")

    join_window = stage_wall_cap if stage_wall_cap is not None else 0.1
    if presence_deadline is not None:
        join_window = min(join_window, presence_cap) if join_window is not None else presence_cap
    if bootstrap_fast:
        join_window = min(join_window or 0.0, 0.05)
    try:
        thread.join(join_window)
    except Exception:
        LOGGER.debug("embedder warmup join probe failed", exc_info=True)

    if thread.is_alive() and presence_deadline is not None:
        if perf_counter() >= presence_deadline:
            try:
                placeholder = _activate_bundled_fallback(
                    "embedder_presence_timeout"
                ) or placeholder
            except Exception:  # pragma: no cover - advisory only
                LOGGER.debug("failed to promote fallback after presence probe timeout", exc_info=True)
        _BOOTSTRAP_EMBEDDER_JOB["deferred"] = True
        _BOOTSTRAP_EMBEDDER_JOB["placeholder_reason"] = "presence_check_deferred"
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(
            reason="presence_check_deferred"
        )
        LOGGER.info(
            "embedder presence check deferred to background",  # pragma: no cover - telemetry
            extra={
                "presence_cap": presence_cap,
                "stage_budget": stage_budget,
                "timeout": timeout,
                "probe": presence_probe,
                "bootstrap_fast": bootstrap_fast,
            },
        )
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason="embedder_presence_check_deferred"
        )
        if bootstrap_fast:
            _enqueue_background_download()
        return placeholder

    if thread.is_alive() and hard_join_deadline is not None:
        remaining = hard_join_deadline - perf_counter()
        if remaining > 0:
            try:
                thread.join(remaining)
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("embedder hard join window failed", exc_info=True)
        if thread.is_alive():
            join_reason = "embedder_join_cap"
            _record_abort(join_reason, deferred=True, schedule_background=True)
            return placeholder

    if thread.is_alive() and stage_wall_cap is not None:
        _record_abort(
            "stage_wall_cap_exceeded", deferred=True, schedule_background=True
        )
        return placeholder

    if hard_timeout_triggered.is_set():
        return placeholder

    return placeholder


def start_embedder_warmup(
    *,
    timeout: float | None = None,
    stage_budget: float | None = None,
    stop_event: threading.Event | None = None,
    budget: SharedTimeoutCoordinator | None = None,
    budget_label: str | None = None,
    bootstrap_deadline: float | None = None,
    allow_full_preload: bool | None = None,
) -> Any:
    """Launch embedder warmup quickly and defer heavy downloads when budgets are tight."""

    global _BOOTSTRAP_EMBEDDER_ATTEMPTED

    resolved_timeout = BOOTSTRAP_EMBEDDER_TIMEOUT if timeout is None else timeout
    presence_cap = float(os.getenv("BOOTSTRAP_EMBEDDER_PRESENCE_CAP", "0.75"))
    if presence_cap < 0:
        presence_cap = 0.0
    bootstrap_context_active = False
    warmup_lite_context = False
    resume_embedder_download_default = False
    try:
        bootstrap_context_active = bool(
            _current_bootstrap_context()
            or getattr(_coding_bot_interface, "_BOOTSTRAP_STATE", None)
        )
        warmup_lite_context = bool(
            getattr(
                getattr(_coding_bot_interface, "_BOOTSTRAP_STATE", None),
                "warmup_lite",
                False,
            )
        )
    except Exception:  # pragma: no cover - advisory only
        LOGGER.debug(
            "unable to resolve bootstrap context for embedder warmup", exc_info=True
        )
    force_full_preload = bool(allow_full_preload)
    if not force_full_preload:
        try:
            force_full_preload = (
                os.getenv("MENACE_FORCE_EMBEDDER_PRELOAD", "")
                .strip()
                .lower()
                in {"1", "true", "yes"}
            )
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("failed to read MENACE_FORCE_EMBEDDER_PRELOAD", exc_info=True)
    now = perf_counter()
    deadline_remaining = (bootstrap_deadline - now) if bootstrap_deadline else None
    placeholder = _BOOTSTRAP_PLACEHOLDER
    existing_job = _BOOTSTRAP_EMBEDDER_JOB or {}
    placeholder = existing_job.get("placeholder", placeholder)
    placeholder_reason = existing_job.get("warmup_placeholder_reason") or existing_job.get(
        "placeholder_reason"
    )
    if existing_job.get("ready_after_bootstrap") and existing_job.get("deferred"):
        placeholder = existing_job.get("placeholder", _BOOTSTRAP_PLACEHOLDER)
        join_timeout = existing_job.get("background_join_timeout")
        background_future = existing_job.get("background_future")
        capped_join: float | None = None
        try:
            if join_timeout is not None:
                capped_join = max(0.0, min(float(join_timeout), 5.0))
        except (TypeError, ValueError):
            capped_join = None
        if background_future is not None and capped_join is not None:
            with contextlib.suppress(Exception):
                background_future.result(timeout=capped_join)
        if not placeholder_reason:
            placeholder_reason = existing_job.get("background_enqueue_reason")
        existing_job["warmup_placeholder_reason"] = placeholder_reason
        _BOOTSTRAP_EMBEDDER_ATTEMPTED = True
        LOGGER.info(
            "embedder warmup deferred during bootstrap; returning placeholder",
            extra={
                "placeholder_reason": placeholder_reason,
                "stage_budget": stage_budget,
                "join_timeout": capped_join,
            },
        )
        return placeholder
    guard_reason = None
    if stage_budget is None:
        guard_reason = "embedder_stage_budget_missing"
    elif stage_budget >= _BOOTSTRAP_TIMEOUT_FLOOR:
        guard_reason = "embedder_stage_budget_large"
    contextual_guard = (
        (bootstrap_context_active or warmup_lite_context)
        and not force_full_preload
    )
    if guard_reason is None and contextual_guard:
        guard_reason = "embedder_presence_context_guard"
    if guard_reason is None:
        if bootstrap_deadline is None:
            guard_reason = "embedder_deadline_missing"
        elif deadline_remaining is not None and deadline_remaining >= resolved_timeout:
            guard_reason = "embedder_deadline_large"
    generous_budget_guard = (
        guard_reason is None
        and not force_full_preload
        and (
            stage_budget is None
            or (resolved_timeout is not None and stage_budget >= resolved_timeout)
            or (deadline_remaining is not None and deadline_remaining >= resolved_timeout)
        )
    )
    if guard_reason is None and generous_budget_guard:
        guard_reason = "embedder_generous_budget_guard"
    if guard_reason:
        _BOOTSTRAP_EMBEDDER_ATTEMPTED = True
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=guard_reason)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{guard_reason}"
        )
        guard_timebox = presence_cap if presence_cap > 0 else None
        telemetry = {
            "stage_budget": stage_budget,
            "presence_cap": presence_cap,
            "bootstrap_deadline": bootstrap_deadline,
            "deadline_remaining": deadline_remaining,
            "resolved_timeout": resolved_timeout,
            "guard_reason": guard_reason,
            "bootstrap_context": bootstrap_context_active,
            "warmup_lite_context": warmup_lite_context,
            "force_full_preload": force_full_preload,
            "guard_timebox": guard_timebox,
        }
        LOGGER.info(
            "embedder warmup guarded upstream; returning placeholder",  # pragma: no cover - telemetry
            extra=telemetry,
        )
        job_snapshot = _BOOTSTRAP_EMBEDDER_JOB or {}
        job_snapshot.update(
            {
                "result": placeholder,
                "placeholder": placeholder,
                "placeholder_reason": guard_reason,
                "warmup_placeholder_reason": guard_reason,
                "deferred": True,
                "background_enqueue_reason": guard_reason,
                "deferral_reason": guard_reason,
                "strict_timebox": guard_timebox,
                "presence_only": True,
            }
        )
        _BOOTSTRAP_EMBEDDER_JOB = job_snapshot

        def _background_presence() -> None:
            try:
                _bootstrap_embedder(
                    resolved_timeout,
                    stop_event=stop_event,
                    stage_budget=stage_budget,
                    budget=budget,
                    budget_label=budget_label,
                    presence_probe=True,
                    presence_reason=guard_reason,
                    presence_only_guard=True,
                    bootstrap_fast=True,
                    force_placeholder=True,
                    bootstrap_deadline=bootstrap_deadline,
                    precomputed_caps={
                        "presence_cap": presence_cap,
                        "stage_budget": stage_budget,
                        "bootstrap_deadline": bootstrap_deadline,
                        "timeout": resolved_timeout,
                        "strict_timebox": guard_timebox,
                    },
                )
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug(
                    "background embedder guard presence probe failed", exc_info=True
                )

        background_future = _BOOTSTRAP_BACKGROUND_EXECUTOR.submit(_background_presence)
        _BOOTSTRAP_EMBEDDER_JOB["background_future"] = background_future
        return placeholder
    tight_budget = (
        stage_budget is not None
        and stage_budget > 0
        and presence_cap > 0
        and stage_budget <= presence_cap
    )
    if stage_budget is not None and stage_budget > 0 and resolved_timeout > 0:
        resolved_timeout = min(resolved_timeout, stage_budget)

    stage_budget_cap_candidates = [
        candidate
        for candidate in (
            stage_budget if stage_budget is not None and stage_budget > 0 else None,
            deadline_remaining if deadline_remaining is not None and deadline_remaining > 0 else None,
        )
        if candidate is not None
    ]
    stage_budget_cap = min(stage_budget_cap_candidates) if stage_budget_cap_candidates else None

    def _run_bootstrap_embedder(local_stop_event: threading.Event | None) -> Any:
        return _bootstrap_embedder(
            resolved_timeout,
            stop_event=local_stop_event,
            stage_budget=stage_budget,
            budget=budget,
            budget_label=budget_label,
            bootstrap_fast=tight_budget,
            force_placeholder=tight_budget,
            bootstrap_deadline=bootstrap_deadline,
        )

    if stage_budget_cap is None:
        return _run_bootstrap_embedder(stop_event)

    def _remaining_stage_budget() -> float:
        elapsed = time.monotonic() - warmup_started
        return stage_budget_cap - elapsed

    budget_remaining = _remaining_stage_budget()
    presence_fastpath_threshold = 8.0
    try:
        presence_fastpath_threshold = float(
            os.getenv("BOOTSTRAP_EMBEDDER_PRESENCE_FASTPATH", presence_fastpath_threshold)
        )
    except Exception:  # pragma: no cover - advisory only
        LOGGER.debug("unable to parse presence fastpath threshold", exc_info=True)
    if presence_fastpath_threshold < 0:
        presence_fastpath_threshold = 0.0

    def _schedule_presence_fastpath(reason: str) -> Any:
        budget_hint = budget_remaining if budget_remaining is not None else stage_budget_cap
        job_snapshot = _schedule_background_preload(
            reason, strict_timebox=budget_hint
        )
        job_snapshot.update(
            {
                "presence_only": True,
                "deferral_reason": reason,
                "warmup_placeholder_reason": job_snapshot.get(
                    "warmup_placeholder_reason", reason
                ),
                "budget_remaining": budget_remaining,
                "stage_budget_cap": stage_budget_cap,
                "bootstrap_context": bootstrap_context_active,
                "warmup_lite": warmup_lite_context,
                "background_join_timeout": budget_hint,
            }
        )
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=reason)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{reason}"
        )
        _BOOTSTRAP_SCHEDULER.mark_partial("vectorizer_preload", reason=reason)
        _BOOTSTRAP_EMBEDDER_JOB = job_snapshot
        LOGGER.info(
            "embedder warmup running as background presence-only preload",
            extra={
                "reason": reason,
                "budget_hint": budget_hint,
                "bootstrap_context": bootstrap_context_active,
                "warmup_lite": warmup_lite_context,
                "presence_threshold": presence_fastpath_threshold,
            },
        )
        return job_snapshot.get("placeholder", _BOOTSTRAP_PLACEHOLDER)

    if (warmup_lite_context or bootstrap_context_active) and not force_full_preload:
        if budget_remaining is None or budget_remaining <= presence_fastpath_threshold:
            return _schedule_presence_fastpath("embedder_presence_fastpath")

    if budget_remaining <= 0:
        timeout_reason = "embedder_stage_budget_timeout"
        LOGGER.info(
            "embedder preload budget exhausted before launch; deferring",  # pragma: no cover - telemetry
            extra={
                "event": "embedder-preload-stage-timeout",
                "elapsed": round(time.monotonic() - warmup_started, 3),
                "stage_budget": stage_budget,
                "bootstrap_window": deadline_remaining,
                "budget_remaining": budget_remaining,
                "presence_only": presence_only,
                "budget_guarded": True,
            },
        )
        job_snapshot = _schedule_background_preload(timeout_reason)
        job_snapshot.update(
            {
                "timed_out": True,
                "elapsed": round(time.monotonic() - warmup_started, 3),
                "stage_budget": stage_budget,
                "bootstrap_window": deadline_remaining,
                "budget_remaining": budget_remaining,
            }
        )
        _BOOTSTRAP_EMBEDDER_JOB = job_snapshot
        stage_controller.defer_step("embedder_preload", reason=timeout_reason)
        stage_controller.complete_step("embedder_preload", 0.0)
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=timeout_reason)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{timeout_reason}"
        )
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "vectorizer_preload", reason=timeout_reason
        )
        return job_snapshot.get("placeholder", _BOOTSTRAP_PLACEHOLDER)

    embedder_stop_event = threading.Event()
    if stop_event is not None:
        def _propagate_stop_to_embedder() -> None:
            stop_event.wait()
            if stop_event.is_set():
                embedder_stop_event.set()

        threading.Thread(
            target=_propagate_stop_to_embedder,
            name="embedder-stage-stop",
            daemon=True,
        ).start()

    embedder_future = _BOOTSTRAP_BACKGROUND_EXECUTOR.submit(
        _run_bootstrap_embedder, embedder_stop_event
    )

    join_timeout = 5.0 if budget_remaining is None else min(budget_remaining, 5.0)

    try:
        return embedder_future.result(timeout=join_timeout)
    except FuturesTimeoutError:
        timeout_reason = presence_reason or "embedder_stage_budget_timeout"
        elapsed = time.monotonic() - warmup_started
        embedder_stop_event.set()
        with contextlib.suppress(Exception):
            from menace_sandbox.governed_embeddings import cancel_embedder_initialisation

            cancel_embedder_initialisation(
                embedder_stop_event, reason=timeout_reason, join_timeout=0.0
            )
        LOGGER.info(
            "embedder preload timed out against stage budget; deferring",  # pragma: no cover - telemetry
            extra={
                "event": "embedder-preload-stage-timeout",
                "elapsed": round(elapsed, 3),
                "stage_budget": stage_budget,
                "bootstrap_window": deadline_remaining,
                "budget_remaining": round(_remaining_stage_budget(), 3),
                "presence_only": presence_only,
                "budget_guarded": True,
                "timeout_reason": timeout_reason,
            },
        )
        stage_controller.defer_step("embedder_preload", reason=timeout_reason)
        stage_controller.complete_step("embedder_preload", 0.0)
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=timeout_reason)
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{timeout_reason}"
        )
        _BOOTSTRAP_SCHEDULER.mark_partial("vectorizer_preload", reason=timeout_reason)
        job_snapshot = _schedule_background_preload(timeout_reason)
        job_snapshot.update(
            {
                "timed_out": True,
                "elapsed": round(elapsed, 3),
                "stage_budget": stage_budget,
                "bootstrap_window": deadline_remaining,
                "budget_remaining": round(_remaining_stage_budget(), 3),
                "timeout_reason": timeout_reason,
            }
        )
        _BOOTSTRAP_EMBEDDER_JOB = job_snapshot
        return job_snapshot.get("placeholder", _BOOTSTRAP_PLACEHOLDER)


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
    full_embedder_preload: bool | None = None,
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

    def _env_flag(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

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
    bootstrap_mode_env = os.getenv("MENACE_BOOTSTRAP_MODE", "").strip().lower()
    bootstrap_fast_env = _env_flag("MENACE_BOOTSTRAP_FAST")
    bootstrap_lite_mode = bootstrap_mode_env in {"lite", "fast"}
    bootstrap_fast_context = bootstrap_fast_env or bootstrap_lite_mode
    warmup_lite_context = False
    try:
        warmup_lite_context = bool(
            getattr(getattr(_coding_bot_interface, "_BOOTSTRAP_STATE", None), "warmup_lite", False)
        )
    except Exception:  # pragma: no cover - advisory only
        LOGGER.debug("unable to resolve warmup_lite flag from bootstrap state", exc_info=True)
    force_vector_warmup = _env_flag("MENACE_FORCE_HEAVY_VECTOR_WARMUP")
    force_embedder_preload = _env_flag("MENACE_FORCE_EMBEDDER_PRELOAD")
    full_embedder_preload = (
        full_embedder_preload
        if full_embedder_preload is not None
        else _env_flag("MENACE_EMBEDDER_FULL_PRELOAD")
    )
    if bootstrap_lite_mode:
        resume_embedder_download_default = True

    if not full_embedder_preload and not warmup_lite_context:
        LOGGER.info(
            "embedder full preload not requested; enforcing warmup-lite presence path",
            extra={
                "event": "embedder-preload-default-lite",
                "bootstrap_fast": bootstrap_fast_context,
            },
        )
        warmup_lite_context = True
    vector_force_warmup = force_vector_warmup or force_embedder_preload
    vector_bootstrap_hint = False
    vector_warmup_requested = False
    if heavy_bootstrap or vector_force_warmup:
        if warmup_lite_context:
            LOGGER.info(
                "warmup-lite active; skipping heavy vector bootstrap hints",
                extra={
                    "event": "vector-warmup-lite-skip",
                    "heavy_bootstrap": heavy_bootstrap,
                    "force_vector_warmup": force_vector_warmup,
                    "force_embedder_preload": force_embedder_preload,
                },
            )
        elif not full_embedder_preload:
            LOGGER.info(
                "full embedder preload not requested; deferring heavy vector bootstrap hints",
                extra={
                    "event": "vector-warmup-preload-skip",
                    "heavy_bootstrap": heavy_bootstrap,
                    "force_vector_warmup": force_vector_warmup,
                    "force_embedder_preload": force_embedder_preload,
                },
            )
        else:
            vector_bootstrap_hint = True
            vector_warmup_requested = True
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
    guard_context: Mapping[str, object] = get_bootstrap_guard_context() or {}
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
    embedder_stage_budget_hint = stage_controller.stage_budget(
        step_name="embedder_preload"
    )
    embedder_stage_deadline_hint = stage_controller.stage_deadline(
        step_name="embedder_preload"
    )
    embedder_warmup_lite_budget_guard = bool(
        embedder_stage_budget_hint is not None
        and math.isfinite(embedder_stage_budget_hint)
        and embedder_stage_budget_hint > 0
    )
    embedder_warmup_lite_deadline_guard = bool(
        embedder_stage_deadline_hint is not None and math.isfinite(embedder_stage_deadline_hint)
    )
    if (embedder_warmup_lite_budget_guard or embedder_warmup_lite_deadline_guard) and not warmup_lite_context:
        LOGGER.info(
            "forcing embedder warmup-lite path due to finite stage window",
            extra={
                "event": "embedder-preload-budget-lite",
                "stage_budget": embedder_stage_budget_hint,
                "stage_deadline": embedder_stage_deadline_hint,
            },
        )
        warmup_lite_context = True
    embedder_preload_enabled = True
    embedder_preload_stage_blocked = bool(
        embedder_stage_budget_hint is not None and embedder_stage_budget_hint <= 0.0
    )
    embedder_preload_skip_reason = None
    embedder_preload_deferred = False

    def _task_embedder(_: dict[str, Any]) -> None:
        global _BOOTSTRAP_EMBEDDER_JOB
        _mark_bootstrap_step("embedder_preload")
        existing_job_snapshot = _BOOTSTRAP_EMBEDDER_JOB or {}
        existing_background = existing_job_snapshot.get("background_future") or getattr(
            _BOOTSTRAP_SCHEDULER, "_embedder_background_future", None
        )
        if existing_background is not None:
            resume_reason = existing_job_snapshot.get("deferral_reason") or existing_job_snapshot.get(
                "background_enqueue_reason", "embedder_background_resume"
            )
            LOGGER.info(
                "embedder preload resuming from background state",
                extra={
                    "event": "embedder-preload-background",
                    "reason": resume_reason,
                    "bootstrap_fast": bootstrap_fast_context,
                    "warmup_lite": warmup_lite_context,
                },
            )
            stage_controller.defer_step("embedder_preload", reason=resume_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial("vectorizer_preload", reason=resume_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "background_loops", reason=f"embedder_placeholder:{resume_reason}"
            )
            placeholder_obj = existing_job_snapshot.get("placeholder", _BOOTSTRAP_PLACEHOLDER)
            return existing_job_snapshot.get("result", placeholder_obj)
        embedder_gate = _BOOTSTRAP_CONTENTION_COORDINATOR.negotiate_step(
            "embedder_preload", vector_heavy=True, heavy=True
        )
        embedder_timeout = BOOTSTRAP_EMBEDDER_TIMEOUT * embedder_gate["timeout_scale"]
        embedder_stage_budget = stage_controller.stage_budget(step_name="embedder_preload")
        embedder_stage_deadline = stage_controller.stage_deadline(
            step_name="embedder_preload"
        )
        bootstrap_guard_active = False
        guard_budget_scale = 1.0
        guard_delay = 0.0
        if guard_context:
            try:
                guard_budget_scale = float(guard_context.get("budget_scale", 1.0) or 1.0)
            except Exception:
                guard_budget_scale = 1.0
            try:
                guard_delay = float(guard_context.get("delay", 0.0) or 0.0)
            except Exception:
                guard_delay = 0.0
            bootstrap_guard_active = bool(guard_delay > 0 or guard_budget_scale > 1.0 or guard_context)
        embedder_stage_deadline_remaining = (
            max(0.0, embedder_stage_deadline - time.monotonic())
            if embedder_stage_deadline is not None
            else None
        )
        stage_deadline_hint_remaining = (
            max(0.0, embedder_stage_deadline_hint - time.monotonic())
            if embedder_stage_deadline_hint is not None
            and math.isfinite(embedder_stage_deadline_hint)
            else None
        )
        embedder_warmup_cap = (
            BOOTSTRAP_EMBEDDER_WARMUP_CAP if BOOTSTRAP_EMBEDDER_WARMUP_CAP > 0 else None
        )
        remaining_bootstrap_window = (
            max(0.0, bootstrap_deadline - time.monotonic())
            if bootstrap_deadline is not None
            else None
        )
        coordinator_remaining_budget = getattr(
            shared_timeout_coordinator, "remaining_budget", None
        )
        bootstrap_context_active = False
        try:
            bootstrap_context_active = bool(
                _current_bootstrap_context()
                or getattr(_coding_bot_interface, "_BOOTSTRAP_STATE", None)
            )
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("failed to detect active bootstrap context", exc_info=True)
        estimated_preload_cost = float(
            _COMPONENT_BASELINES.get("vector_seeding")
            or embedder_warmup_cap
            or embedder_timeout
            or BOOTSTRAP_EMBEDDER_TIMEOUT
        )
        budget_window_missing = not (
            (embedder_stage_budget is not None and embedder_stage_budget > 0)
            and (embedder_timeout is not None and embedder_timeout > 0)
        )
        presence_only = False
        presence_reason = None
        budget_guarded = False
        guard_blocks_preload = False
        non_blocking_presence_probe = False
        fast_presence_reason = None
        fast_presence_enforced = False
        finite_stage_window = bool(
            (
                embedder_stage_budget_hint is not None
                and math.isfinite(embedder_stage_budget_hint)
            )
            or (
                embedder_stage_deadline_hint is not None
                and math.isfinite(embedder_stage_deadline_hint)
            )
        )
        if bootstrap_fast_context:
            fast_presence_reason = "embedder_presence_bootstrap_fast"
        elif warmup_lite_context:
            fast_presence_reason = "embedder_presence_warmup_lite"
        elif finite_stage_window:
            fast_presence_reason = "embedder_presence_finite_stage_window"
        if fast_presence_reason:
            presence_only = True
            budget_guarded = True
            guard_blocks_preload = True
            non_blocking_presence_probe = True
            presence_reason = fast_presence_reason
            fast_presence_enforced = True
        if bootstrap_fast_context:
            presence_reason = presence_reason or "embedder_presence_bootstrap_fast_guard"
        elif warmup_lite_context:
            presence_reason = presence_reason or "embedder_presence_warmup_lite_guard"
        if budget_window_missing:
            presence_only = True
            budget_guarded = True
            guard_blocks_preload = True
            presence_reason = presence_reason or "embedder_budget_unavailable"
        guard_presence_only = presence_only
        guard_presence_reason = presence_reason
        guard_budget_guarded = budget_guarded
        presence_default_enforced = False
        full_preload_requested = bool(full_embedder_preload or force_embedder_preload)
        if bootstrap_lite_mode:
            presence_only = True
            budget_guarded = True
            guard_blocks_preload = True
            non_blocking_presence_probe = True
            presence_reason = presence_reason or "embedder_presence_bootstrap_lite"
        if guard_blocks_preload:
            guard_presence_only = True
            guard_budget_guarded = True
            guard_presence_reason = presence_reason
        else:
            presence_default_enforced = (
                (bootstrap_fast_context or warmup_lite_context or bootstrap_guard_active)
                and not full_preload_requested
            )
            if not presence_default_enforced:
                presence_default_enforced = not full_preload_requested
        embedder_deferred, embedder_deferral_reason = _BOOTSTRAP_SCHEDULER.embedder_deferral()
        fast_or_lite = bootstrap_fast_context or warmup_lite_context
        gate_constrained = bool(embedder_gate.get("contention"))
        presence_only, presence_reason, budget_guarded = _embedder_presence_policy(
            gate_constrained=gate_constrained,
            stage_budget=embedder_stage_budget,
            embedder_timeout=embedder_timeout,
            force_full_preload=full_preload_requested,
            fast_or_lite=fast_or_lite,
            embedder_deferred=embedder_deferred,
            bootstrap_fast_context=bootstrap_fast_context,
            warmup_lite_context=warmup_lite_context,
            bootstrap_guard_active=bootstrap_guard_active,
        )
        presence_only = presence_only or guard_presence_only
        budget_guarded = budget_guarded or guard_budget_guarded
        presence_reason = presence_reason or guard_presence_reason
        if not full_preload_requested and not embedder_deferred:
            presence_only = True
            budget_guarded = True
            non_blocking_presence_probe = True
            presence_default_enforced = True
            presence_reason = presence_reason or "embedder_presence_bootstrap_default"
        fast_presence_reason = None
        if bootstrap_fast_context:
            fast_presence_reason = "embedder_presence_bootstrap_fast"
        elif warmup_lite_context:
            fast_presence_reason = "embedder_presence_warmup_lite"
        elif bootstrap_guard_active:
            fast_presence_reason = "embedder_presence_bootstrap_guard"
        if bootstrap_context_active and not full_preload_requested:
            presence_only = True
            budget_guarded = True
            presence_reason = presence_reason or "embedder_presence_bootstrap_context_active"
        if presence_default_enforced and not presence_reason:
            presence_reason = "embedder_presence_default_enforced"
        if fast_presence_reason:
            presence_only = True
            budget_guarded = True
            presence_reason = fast_presence_reason
        minimum_presence_window = 5.0
        if budget_window_missing:
            presence_only = True
            budget_guarded = True
            non_blocking_presence_probe = True
            presence_reason = presence_reason or "embedder_budget_unavailable"
        explicit_budget_available = any(
            candidate is not None and candidate > 0
            for candidate in (
                embedder_stage_budget,
                embedder_timeout,
                embedder_stage_deadline_remaining,
                remaining_bootstrap_window,
            )
        )
        if full_preload_requested and not explicit_budget_available:
            presence_only = True
            budget_guarded = True
            non_blocking_presence_probe = True
            presence_reason = presence_reason or "embedder_full_preload_budget_required"
        near_deadline = False
        for deadline_value, deadline_reason in (
            (
                embedder_stage_deadline_remaining,
                "embedder_stage_deadline_near_exhaustion",
            ),
            (
                remaining_bootstrap_window,
                "embedder_bootstrap_deadline_near_exhaustion",
            ),
        ):
            if deadline_value is not None and deadline_value <= max(
                minimum_presence_window, estimated_preload_cost
            ):
                near_deadline = True
                presence_reason = presence_reason or deadline_reason
        if near_deadline:
            presence_only = True
            budget_guarded = True
            non_blocking_presence_probe = True
        deadline_shortfall = (
            remaining_bootstrap_window is None
            or remaining_bootstrap_window < estimated_preload_cost
        )
        if deadline_shortfall:
            presence_only = True
            presence_reason = presence_reason or (
                "embedder_deadline_unavailable"
                if remaining_bootstrap_window is None
                else "embedder_deadline_tight"
            )
        skip_heavy = presence_only or embedder_deferred
        presence_reason = embedder_deferral_reason or presence_reason

        timebox_candidates = [
            candidate
            for candidate in (
                embedder_stage_budget,
                embedder_timeout,
                embedder_warmup_cap,
                embedder_stage_deadline_remaining,
                remaining_bootstrap_window,
                coordinator_remaining_budget,
            )
            if candidate is not None and candidate > 0
        ]
        strict_timebox = min(timebox_candidates) if timebox_candidates else None
        default_missing_budget_timebox = 12.0
        stage_guard_timebox = None
        if embedder_stage_budget_hint is not None:
            stage_guard_timebox = max(5.0, min(embedder_stage_budget_hint, 10.0))
        elif embedder_stage_budget is not None:
            stage_guard_timebox = max(5.0, min(embedder_stage_budget, 10.0))
        elif bootstrap_fast_context or warmup_lite_context or budget_window_missing:
            stage_guard_timebox = 10.0
        hard_cap_candidates = [
            candidate
            for candidate in (embedder_stage_budget, embedder_timeout)
            if candidate is not None and candidate > 0
        ]
        warmup_hard_cap = min(hard_cap_candidates) if hard_cap_candidates else None
        if warmup_hard_cap is not None:
            strict_timebox = warmup_hard_cap if strict_timebox is None else min(strict_timebox, warmup_hard_cap)
        if stage_guard_timebox is not None:
            strict_timebox = (
                stage_guard_timebox if strict_timebox is None else min(strict_timebox, stage_guard_timebox)
            )
        if budget_window_missing and strict_timebox is None:
            strict_timebox = default_missing_budget_timebox
        background_dispatch_gate = _BOOTSTRAP_CONTENTION_COORDINATOR.negotiate_step(
            "embedder_background_dispatch", vector_heavy=True, heavy=True
        )
        guard_timebox = (
            strict_timebox or embedder_stage_budget or embedder_stage_budget_hint
        )
        guard_forced_background = (
            (bootstrap_fast_context or warmup_lite_context)
            or (
                embedder_stage_budget_hint is not None
                and math.isfinite(embedder_stage_budget_hint)
            )
        )
        if guard_forced_background and not full_preload_requested:
            guard_reason = "embedder_backgrounded_guard"
            if bootstrap_fast_context:
                guard_reason = "embedder_backgrounded_bootstrap_fast"
            elif warmup_lite_context:
                guard_reason = "embedder_backgrounded_warmup_lite"
            elif embedder_stage_budget_hint is not None:
                guard_reason = "embedder_backgrounded_finite_budget"
            return _defer_to_presence(
                guard_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=True,
                strict_timebox=guard_timebox,
                non_blocking_probe=True,
                resume_download=True,
            )
        warmup_summary: dict[str, Any] | None = None
        if (
            (embedder_stage_budget is not None and embedder_stage_budget < minimum_presence_window)
            or (strict_timebox is not None and strict_timebox < minimum_presence_window)
        ):
            presence_only = True
            budget_guarded = True
            non_blocking_presence_probe = True
            presence_reason = presence_reason or "embedder_preload_min_budget"

        def _defer_to_presence(
            reason: str,
            *,
            budget_guarded: bool,
            budget_window_missing: bool,
            forced_background: bool = False,
            strict_timebox: float | None = None,
            non_blocking_probe: bool = False,
            resume_download: bool | None = None,
        ) -> Any:
            resolved_resume_download = (
                resume_download
                if resume_download is not None
                else resume_embedder_download_default
            )
            LOGGER.info(
                "embedder preload guarded; scheduling presence probe",
                extra={
                    "gate": embedder_gate,
                    "budget": embedder_stage_budget,
                    "force_preload": full_preload_requested,
                    "full_preload_flag": full_embedder_preload,
                    "bootstrap_fast": bootstrap_fast_context,
                    "warmup_lite": warmup_lite_context,
                    "bootstrap_guard_active": bootstrap_guard_active,
                    "guard_budget_scale": guard_budget_scale,
                    "guard_delay": guard_delay,
                    "embedder_deferred": embedder_deferred,
                    "budget_guarded": budget_guarded,
                    "presence_default": presence_default_enforced,
                    "presence_reason": reason,
                    "budget_window_missing": budget_window_missing,
                    "forced_background": forced_background,
                    "background_gate": background_dispatch_gate,
                    "background_dispatch": True,
                    "strict_timebox": strict_timebox,
                    "resume_download": resolved_resume_download,
                    "event": "embedder-preload-deferred",
                },
            )
            stage_controller.defer_step("embedder_preload", reason=reason)
            _BOOTSTRAP_SCHEDULER.mark_partial("vectorizer_preload", reason=reason)
            _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "background_loops", reason=f"embedder_placeholder:{reason}"
            )
            probe_timeout_candidates = [
                candidate
                for candidate in (
                    embedder_timeout,
                    embedder_stage_budget,
                    embedder_stage_budget_hint,
                    0.75,
                )
                if candidate is not None and candidate > 0
            ]
            probe_timeout = min(probe_timeout_candidates) if probe_timeout_candidates else 0.75
            probe_timeout_hard_cap = 1.5
            presence_probe_stage_cap = 0.65
            probe_timeout = min(probe_timeout, probe_timeout_hard_cap, presence_probe_stage_cap)
            probe_stop_event = threading.Event()
            presence_available = False
            probe_timed_out = False
            stage_budget_cap = (
                embedder_stage_budget_hint
                if embedder_stage_budget_hint is not None
                else embedder_stage_budget
            )
            if stage_budget_cap is not None and stage_budget_cap <= 0:
                probe_timed_out = True
            elif stage_budget_cap is not None and probe_timeout > stage_budget_cap:
                probe_timed_out = True
                LOGGER.info(
                    "presence probe capped by stage budget; scheduling background",
                    extra={
                        "event": "embedder-preload-deferred",
                        "stage_budget": stage_budget_cap,
                        "probe_timeout": probe_timeout,
                        "reason": reason,
                    },
                )
                background_join_timeout_candidates = [
                    candidate
                    for candidate in (
                        embedder_timeout,
                        embedder_stage_budget,
                        embedder_stage_budget_hint,
                        5.0,
                    )
                    if candidate is not None and candidate > 0
                ]
                background_join_timeout = (
                    min(background_join_timeout_candidates)
                    if background_join_timeout_candidates
                    else 1.0
                )
    
                try:
                    from menace_sandbox.governed_embeddings import embedder_cache_present
                except Exception:
                    LOGGER.debug(
                        "embedder presence probe unavailable; skipping cache check",
                        exc_info=True,
                    )
                else:
                    def _probe_worker() -> None:
                        nonlocal presence_available
                        if stop_event and stop_event.is_set():
                            return
                        try:
                            presence_available = bool(embedder_cache_present())
                        except Exception:  # pragma: no cover - diagnostics only
                            LOGGER.debug("embedder presence probe failed", exc_info=True)
    
                    if not probe_timed_out:
                        probe_thread = threading.Thread(
                            target=_probe_worker,
                            name="embedder-presence-probe",
                            daemon=True,
                        )
                        probe_thread.start()
    
                        if not non_blocking_probe:
                            if stop_event is not None:
    
                                def _propagate_stop() -> None:
                                    stop_event.wait(probe_timeout)
                                    if stop_event.is_set():
                                        probe_stop_event.set()
    
                                threading.Thread(
                                    target=_propagate_stop,
                                    name="embedder-presence-probe-stop",
                                    daemon=True,
                                ).start()
    
                            probe_thread.join(probe_timeout)
                            if probe_thread.is_alive() or probe_stop_event.is_set():
                                probe_timed_out = True
                                probe_stop_event.set()
    
                if warmup_summary is not None:
                    warmup_summary.setdefault("deferred", True)
                    warmup_summary.setdefault("deferred_reason", reason)
                    warmup_summary.setdefault("stage", "deferred-timebox")
                    job_snapshot = _BOOTSTRAP_EMBEDDER_JOB or {}
                    job_snapshot["warmup_summary"] = warmup_summary
                else:
                    job_snapshot = _BOOTSTRAP_EMBEDDER_JOB or {}
    
                job_snapshot.setdefault("placeholder", _BOOTSTRAP_PLACEHOLDER)
                job_snapshot.setdefault("placeholder_reason", reason)
                job_snapshot.setdefault(
                    "budget_hints",
                    {
                        "strict_timebox": strict_timebox,
                        "stage_budget": embedder_stage_budget,
                        "stage_budget_hint": embedder_stage_budget_hint,
                        "stage_deadline": embedder_stage_deadline,
                        "bootstrap_deadline": bootstrap_deadline,
                    },
                )
                job_snapshot.update(
                    {
                        "deferred": True,
                        "ready_after_bootstrap": True,
                        "presence_default": presence_default_enforced,
                        "presence_available": presence_available,
                        "presence_probe_timeout": probe_timed_out,
                        "background_enqueue_reason": reason,
                        "background_join_timeout": background_join_timeout,
                        "background_gate": background_dispatch_gate,
                        "background_dispatch": True,
                        "budget_guarded": budget_guarded,
                        "warmup_placeholder_reason": reason,
                        "presence_only": True,
                        "budget_window_missing": budget_window_missing,
                        "deferral_reason": reason,
                        "strict_timebox": strict_timebox,
                        "background_full_warmup": True,
                        "resume_embedder_download": bool(resolved_resume_download),
                    }
                )
    
                def _background_preload() -> None:
                    try:
                        background_cap = (
                            strict_timebox
                            if strict_timebox is not None
                            else embedder_stage_budget
                        )
                        cap_deadline = (
                            time.monotonic() + background_cap
                            if background_cap is not None and background_cap > 0
                            else None
                        )
    
                        def _record_background_placeholder(reason: str) -> None:
                            nonlocal job_snapshot
                            placeholder_obj = job_snapshot.get(
                                "placeholder", _BOOTSTRAP_PLACEHOLDER
                            )
                            job_snapshot.update(
                                {
                                    "result": placeholder_obj,
                                    "placeholder_reason": reason,
                                    "warmup_placeholder_reason": job_snapshot.get(
                                        "warmup_placeholder_reason", reason
                                    ),
                                    "background_full_warmup": False,
                                    "strict_timebox": background_cap
                                    if background_cap is not None
                                    else strict_timebox,
                                }
                            )
                            _BOOTSTRAP_SCHEDULER.mark_partial(
                                "background_loops", reason=f"embedder_placeholder:{reason}"
                            )
    
                        if background_cap is not None and background_cap <= 0:
                            _record_background_placeholder("embedder_background_cap_exhausted")
                            return
    
                        shared_budget_remaining = getattr(
                            shared_timeout_coordinator, "remaining_budget", None
                        )
                        if (
                            shared_budget_remaining is not None
                            and shared_budget_remaining <= 0
                        ):
                            _record_background_placeholder(
                                "embedder_background_shared_budget_exhausted"
                            )
                            return
    
                        ready_timeout = background_cap if background_cap is not None else None
                        ready = _BOOTSTRAP_EMBEDDER_READY.wait(ready_timeout)
                        if not ready:
                            _record_background_placeholder(
                                "embedder_background_cap_exhausted"
                            )
                            return
                        if stop_event is not None and stop_event.is_set():
                            return
                        resolved_timeout = embedder_timeout
                        if resolved_timeout is None or resolved_timeout <= 0:
                            resolved_timeout = BOOTSTRAP_EMBEDDER_TIMEOUT
                        if job_snapshot.get("resume_embedder_download"):
                            try:
                                from menace_sandbox.vector_service.lazy_bootstrap import (
                                    _queue_background_model_download,
                                )

                                _queue_background_model_download(
                                    LOGGER,
                                    download_timeout=background_cap
                                    if background_cap is not None
                                    else strict_timebox,
                                )
                            except Exception:
                                LOGGER.debug(
                                    "failed to enqueue background embedding model download",
                                    exc_info=True,
                                )
                        LOGGER.info(
                            "starting deferred embedder preload after readiness",
                            extra={
                                "event": "embedder-preload-background",
                                "presence_only": True,
                                "bootstrap_fast": bootstrap_fast_context,
                                "warmup_lite": warmup_lite_context,
                                "reason": reason,
                                "background_gate": background_dispatch_gate,
                            },
                        )
                        cap_hints = {
                            "stage_budget": embedder_stage_budget,
                            "stage_budget_hint": embedder_stage_budget_hint,
                            "stage_deadline": embedder_stage_deadline,
                            "stage_wall_cap": strict_timebox,
                            "bootstrap_deadline": bootstrap_deadline,
                        }
                        background_result = _bootstrap_embedder(
                            resolved_timeout,
                            stop_event=stop_event,
                            stage_budget=embedder_stage_budget,
                            budget=shared_timeout_coordinator,
                            budget_label="vector_seeding",
                            presence_probe=False,
                            presence_reason=reason,
                            bootstrap_fast=False,
                            schedule_background=True,
                            bootstrap_deadline=bootstrap_deadline,
                            precomputed_caps={
                                **{k: v for k, v in cap_hints.items() if v is not None},
                                "stage_budget": embedder_stage_budget,
                                "stage_deadline": cap_deadline,
                                "stage_wall_cap": background_cap,
                            },
                        )
                        placeholder_obj = job_snapshot.get(
                            "placeholder", _BOOTSTRAP_PLACEHOLDER
                        )
                        if background_result is None or background_result is placeholder_obj:
                            job_snapshot["background_full_warmup"] = False
                    except Exception:  # pragma: no cover - background safety
                        LOGGER.debug("deferred embedder preload failed", exc_info=True)
    
                background_future = _BOOTSTRAP_SCHEDULER.schedule_embedder_background(
                    _background_preload, reason=reason
                )
                if background_future is not None:
                    job_snapshot["background_future"] = background_future
                    job_snapshot["background_enqueued"] = True
                    LOGGER.info(
                        "embedder preload deferred to background",  # pragma: no cover - telemetry
                        extra={
                            "event": "embedder-preload-background",
                            "reason": reason,
                            "bootstrap_fast": bootstrap_fast_context,
                            "warmup_lite": warmup_lite_context,
                        },
                    )
    
                _BOOTSTRAP_EMBEDDER_JOB = job_snapshot
                _set_component_state("vector_seeding", "deferred")
                stage_controller.complete_step("embedder_preload", 0.0)
                return _BOOTSTRAP_PLACEHOLDER

        constrained_presence_reason = None
        constrained_presence_only = False
        if warmup_lite_context:
            constrained_presence_only = True
            constrained_presence_reason = "embedder_presence_warmup_lite_hint"
        elif embedder_stage_budget_hint is not None and math.isfinite(embedder_stage_budget_hint):
            constrained_presence_only = True
            constrained_presence_reason = "embedder_presence_budget_hint"
        elif stage_deadline_hint_remaining is not None:
            constrained_presence_only = True
            constrained_presence_reason = "embedder_presence_deadline_hint"

        if constrained_presence_only and not full_preload_requested:
            warmup_summary = warmup_summary or {}
            warmup_summary.setdefault("deferred", True)
            warmup_summary.setdefault("deferred_reason", constrained_presence_reason)
            warmup_summary.setdefault("deferral_reason", constrained_presence_reason)
            strict_timebox_hint = (
                embedder_stage_budget_hint
                if embedder_stage_budget_hint is not None
                else stage_deadline_hint_remaining
            )
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "vectorizer_preload", reason=constrained_presence_reason
            )
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "background_loops",
                reason=f"embedder_placeholder:{constrained_presence_reason}",
            )
            return _defer_to_presence(
                constrained_presence_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=True,
                strict_timebox=strict_timebox_hint,
                non_blocking_probe=True,
                resume_download=True,
            )

        early_background_required = presence_only or guard_blocks_preload or not explicit_budget_available
        if early_background_required:
            skip_reason = presence_reason or (
                "embedder_budget_unavailable" if not explicit_budget_available else "embedder_presence_guarded"
            )
            strict_timebox_hint = None
            if embedder_stage_budget_hint is not None:
                strict_timebox_hint = embedder_stage_budget_hint
            elif embedder_stage_budget is not None:
                strict_timebox_hint = embedder_stage_budget
            LOGGER.info(
                "embedder preload skipped; deferring to background",
                extra={
                    "event": "embedder-preload-skipped",
                    "presence_only": presence_only,
                    "guard_blocks_preload": guard_blocks_preload,
                    "explicit_budget_available": explicit_budget_available,
                    "bootstrap_fast": bootstrap_fast_context,
                    "warmup_lite": warmup_lite_context,
                    "reason": skip_reason,
                },
            )
            return _defer_to_presence(
                skip_reason,
                budget_guarded=budget_guarded,
                budget_window_missing=budget_window_missing or not explicit_budget_available,
                forced_background=True,
                strict_timebox=strict_timebox_hint,
                non_blocking_probe=True,
                resume_download=True,
            )

        warmup_presence_cap: float | None = None
        try:
            warmup_presence_cap = float(
                os.getenv("BOOTSTRAP_EMBEDDER_PRESENCE_CAP", "0.75")
            )
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("unable to parse warmup presence cap", exc_info=True)
        if warmup_presence_cap is not None and warmup_presence_cap < 0:
            warmup_presence_cap = 0.0

        warmup_timebox_cap = strict_timebox
        if warmup_presence_cap is not None and warmup_presence_cap > 0:
            warmup_timebox_cap = (
                warmup_presence_cap
                if warmup_timebox_cap is None
                else min(warmup_timebox_cap, warmup_presence_cap)
            )

        enforced_timebox = (
            warmup_timebox_cap if warmup_timebox_cap is not None else strict_timebox
        )

        warmup_join_ceiling = (
            BOOTSTRAP_EMBEDDER_WARMUP_JOIN_CAP
            if BOOTSTRAP_EMBEDDER_WARMUP_JOIN_CAP > 0
            else None
        )

        fallback_timebox_reason = None
        fallback_timebox_applied = False
        fallback_warmup_cap = (
            BOOTSTRAP_EMBEDDER_FALLBACK_WARMUP_CAP
            if BOOTSTRAP_EMBEDDER_FALLBACK_WARMUP_CAP > 0
            else None
        )
        if warmup_timebox_cap is None and fallback_warmup_cap is not None:
            missing_budget_hints = all(
                candidate is None
                for candidate in (
                    embedder_stage_budget_hint,
                    embedder_timeout,
                    embedder_stage_deadline,
                    bootstrap_deadline,
                )
            )
            if missing_budget_hints:
                warmup_timebox_cap = fallback_warmup_cap
                enforced_timebox = warmup_timebox_cap
                fallback_timebox_reason = "embedder_preload_fallback_timebox"
                fallback_timebox_applied = True

        timebox_insufficient = (
            warmup_timebox_cap is None or warmup_timebox_cap < estimated_preload_cost
        )
        if (
            coordinator_remaining_budget is not None
            and coordinator_remaining_budget <= 0
        ):
            deadline_reason = embedder_deferral_reason or "embedder_bootstrap_budget_exhausted"
            warmup_summary = warmup_summary or {}
            warmup_summary.update(
                {
                    "deferred": True,
                    "deferred_reason": deadline_reason,
                    "deferral_reason": deadline_reason,
                    "strict_timebox": strict_timebox,
                }
            )
            return _defer_to_presence(
                deadline_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=0.0 if strict_timebox is None else strict_timebox,
                non_blocking_probe=True,
            )
        if embedder_stage_deadline_remaining is not None and embedder_stage_deadline_remaining <= 0:
            deadline_reason = embedder_deferral_reason or "embedder_stage_deadline_elapsed"
            warmup_summary = warmup_summary or {}
            warmup_summary.update(
                {
                    "deferred": True,
                    "deferred_reason": deadline_reason,
                    "deferral_reason": deadline_reason,
                    "strict_timebox": strict_timebox,
                }
            )
            return _defer_to_presence(
                deadline_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=0.0 if strict_timebox is None else strict_timebox,
                non_blocking_probe=True,
            )
        if remaining_bootstrap_window is not None and remaining_bootstrap_window <= 0:
            deadline_reason = embedder_deferral_reason or "embedder_bootstrap_deadline_elapsed"
            warmup_summary = warmup_summary or {}
            warmup_summary.update(
                {
                    "deferred": True,
                    "deferred_reason": deadline_reason,
                    "deferral_reason": deadline_reason,
                    "strict_timebox": strict_timebox,
                }
            )
            return _defer_to_presence(
                deadline_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=0.0 if strict_timebox is None else strict_timebox,
                non_blocking_probe=True,
            )
        budget_shortfall_candidates: list[tuple[float, str]] = []
        for candidate, reason in (
            (embedder_stage_deadline_remaining, "embedder_stage_budget_shortfall"),
            (remaining_bootstrap_window, "embedder_bootstrap_budget_shortfall"),
            (coordinator_remaining_budget, "embedder_bootstrap_budget_shortfall"),
        ):
            if candidate is not None and candidate >= 0:
                budget_shortfall_candidates.append((candidate, reason))
        if budget_shortfall_candidates:
            remaining_budget, shortfall_reason = min(
                budget_shortfall_candidates, key=lambda item: item[0]
            )
            if remaining_budget < estimated_preload_cost:
                warmup_summary = warmup_summary or {}
                warmup_summary.update(
                    {
                        "deferred": True,
                        "deferred_reason": shortfall_reason,
                        "deferral_reason": shortfall_reason,
                        "strict_timebox": strict_timebox,
                    }
                )
                return _defer_to_presence(
                    shortfall_reason,
                    budget_guarded=True,
                    budget_window_missing=budget_window_missing,
                    forced_background=full_preload_requested,
                    strict_timebox=remaining_budget,
                    non_blocking_probe=True,
                )
        if timebox_insufficient:
            presence_reason = (
                "embedder_preload_timebox_missing"
                if warmup_timebox_cap is None
                else "embedder_preload_timebox_short"
            )
            warmup_summary = warmup_summary or {}
            warmup_summary.update(
                {
                    "deferred": True,
                    "deferral_reason": presence_reason,
                    "deferred_reason": presence_reason,
                    "strict_timebox": warmup_timebox_cap,
                    "stage_budget": embedder_stage_budget_hint,
                }
            )
            return _defer_to_presence(
                presence_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=warmup_timebox_cap,
                non_blocking_probe=non_blocking_presence_probe,
            )

        warmup_started = time.monotonic()
        warmup_stop_reason: str | None = None
        warmup_join_cap_reason = "embedder_preload_warmup_ceiling"

        def _warmup_budget_remaining() -> float | None:
            remaining_candidates = []
            if embedder_stage_budget_hint is not None:
                remaining_candidates.append(
                    embedder_stage_budget_hint - (time.monotonic() - warmup_started)
                )
            if embedder_timeout is not None:
                remaining_candidates.append(embedder_timeout - (time.monotonic() - warmup_started))
            if embedder_stage_deadline is not None:
                remaining_candidates.append(embedder_stage_deadline - time.monotonic())
            if bootstrap_deadline is not None:
                remaining_candidates.append(bootstrap_deadline - time.monotonic())
            if warmup_timebox_cap is not None:
                remaining_candidates.append(
                    warmup_timebox_cap - (time.monotonic() - warmup_started)
                )

            remaining_filtered = [candidate for candidate in remaining_candidates if candidate is not None]
            if not remaining_filtered:
                return None
            return max(0.0, min(remaining_filtered))

        warmup_stage_timeouts: dict[str, float] | float | None = None
        warmup_budget = _warmup_budget_remaining()
        if warmup_budget is not None:
            warmup_stage_timeouts = {"budget": warmup_budget, "model": warmup_budget}
        elif embedder_stage_budget_hint is not None or embedder_timeout is not None:
            warmup_stage_timeouts = {
                "budget": embedder_stage_budget_hint
                if embedder_stage_budget_hint is not None
                else embedder_timeout,
                "model": embedder_timeout,
            }
        if isinstance(warmup_stage_timeouts, dict) and warmup_timebox_cap is not None:
            for key in ("budget", "model"):
                existing = warmup_stage_timeouts.get(key)
                if existing is None or existing <= 0:
                    warmup_stage_timeouts[key] = warmup_timebox_cap
                else:
                    warmup_stage_timeouts[key] = min(existing, warmup_timebox_cap)
        elif warmup_timebox_cap is not None:
            warmup_stage_timeouts = {"budget": warmup_timebox_cap, "model": warmup_timebox_cap}

        warmup_cap_exceeded = False
        if warmup_join_ceiling is not None:
            warmup_cap_exceeded = any(
                candidate is not None and candidate > warmup_join_ceiling
                for candidate in (estimated_preload_cost, warmup_budget)
            )
        if warmup_cap_exceeded:
            warmup_summary = warmup_summary or {}
            warmup_summary.update(
                {
                    "deferred": True,
                    "deferral_reason": warmup_join_cap_reason,
                    "deferred_reason": warmup_join_cap_reason,
                    "strict_timebox": (
                        min(warmup_timebox_cap, warmup_join_ceiling)
                        if warmup_timebox_cap is not None
                        else warmup_join_ceiling
                    ),
                    "stage_budget": embedder_stage_budget_hint,
                }
            )
            return _defer_to_presence(
                warmup_join_cap_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=(
                    min(warmup_timebox_cap, warmup_join_ceiling)
                    if warmup_timebox_cap is not None
                    else warmup_join_ceiling
                ),
                non_blocking_probe=non_blocking_presence_probe,
                resume_download=True,
            )

        try:
            from menace_sandbox.vector_service.lazy_bootstrap import warmup_vector_service
        except Exception:
            LOGGER.debug("vector warmup unavailable during embedder preload", exc_info=True)
        else:
            try:
                warmup_summary = dict(
                    warmup_vector_service(
                        logger=LOGGER,
                        probe_model=True,
                        warmup_lite=True,
                        stage_timeouts=warmup_stage_timeouts,
                        budget_remaining=_warmup_budget_remaining,
                    )
                )
            except Exception:  # pragma: no cover - advisory warmup only
                LOGGER.debug("embedder preload warmup-lite failed", exc_info=True)
            else:
                if warmup_summary:
                    _BOOTSTRAP_EMBEDDER_JOB = (_BOOTSTRAP_EMBEDDER_JOB or {}) | {
                        "warmup_summary": warmup_summary
                    }

            def _schedule_background_preload(
                reason: str, *, strict_timebox: float | None = None
            ) -> dict[str, Any]:
                if strict_timebox is None:
                    strict_timebox = warmup_timebox_cap
                _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=reason)
                job_snapshot = _BOOTSTRAP_EMBEDDER_JOB or {}
                job_snapshot.setdefault("placeholder", _BOOTSTRAP_PLACEHOLDER)
                job_snapshot.setdefault("placeholder_reason", reason)
                job_snapshot.setdefault(
                    "budget_hints",
                    {
                        "strict_timebox": strict_timebox,
                        "stage_budget": embedder_stage_budget,
                        "stage_budget_hint": embedder_stage_budget_hint,
                        "stage_deadline": embedder_stage_deadline,
                        "bootstrap_deadline": bootstrap_deadline,
                    },
                )
                job_snapshot.update(
                    {
                        "deferred": True,
                        "ready_after_bootstrap": True,
                        "background_enqueue_reason": reason,
                        "warmup_placeholder_reason": reason,
                        "deferral_reason": reason,
                        "background_full_warmup": True,
                        "strict_timebox": strict_timebox,
                        "background_gate": background_dispatch_gate,
                        "background_dispatch": True,
                        "warmup_timebox_cap": strict_timebox,
                    },
                )

            def _background_preload() -> None:
                try:
                    background_cap = (
                        strict_timebox
                        if strict_timebox is not None
                        else embedder_stage_budget
                    )
                    cap_deadline = (
                        time.monotonic() + background_cap
                        if background_cap is not None and background_cap > 0
                        else None
                    )

                    def _record_background_placeholder(reason: str) -> None:
                        nonlocal job_snapshot
                        placeholder_obj = job_snapshot.get(
                            "placeholder", _BOOTSTRAP_PLACEHOLDER
                        )
                        job_snapshot.update(
                            {
                                "result": placeholder_obj,
                                "placeholder_reason": reason,
                                "warmup_placeholder_reason": job_snapshot.get(
                                    "warmup_placeholder_reason", reason
                                ),
                                "background_full_warmup": False,
                                "strict_timebox": background_cap
                                if background_cap is not None
                                else strict_timebox,
                            }
                        )
                        _BOOTSTRAP_SCHEDULER.mark_partial(
                            "background_loops", reason=f"embedder_placeholder:{reason}"
                        )

                    if background_cap is not None and background_cap <= 0:
                        _record_background_placeholder("embedder_background_cap_exhausted")
                        return

                    shared_budget_remaining = getattr(
                        shared_timeout_coordinator, "remaining_budget", None
                    )
                    if (
                        shared_budget_remaining is not None
                        and shared_budget_remaining <= 0
                    ):
                        _record_background_placeholder(
                            "embedder_background_shared_budget_exhausted"
                        )
                        return

                    ready_timeout = background_cap if background_cap is not None else None
                    ready = _BOOTSTRAP_EMBEDDER_READY.wait(ready_timeout)
                    if not ready:
                        _record_background_placeholder(
                            "embedder_background_cap_exhausted"
                        )
                        return
                    if stop_event is not None and stop_event.is_set():
                        return
                    resolved_timeout = embedder_timeout
                    if resolved_timeout is None or resolved_timeout <= 0:
                        resolved_timeout = BOOTSTRAP_EMBEDDER_TIMEOUT
                    if job_snapshot.get("resume_embedder_download"):
                        try:
                            from menace_sandbox.vector_service.lazy_bootstrap import (
                                _queue_background_model_download,
                            )

                            _queue_background_model_download(
                                LOGGER,
                                download_timeout=
                                    background_cap if background_cap is not None else strict_timebox,
                            )
                        except Exception:
                            LOGGER.debug(
                                "failed to enqueue background embedding model download",
                                exc_info=True,
                            )
                    cap_hints = {
                        "stage_budget": embedder_stage_budget,
                        "stage_budget_hint": embedder_stage_budget_hint,
                        "bootstrap_deadline": bootstrap_deadline,
                        "stage_deadline": embedder_stage_deadline,
                        "stage_wall_cap": strict_timebox,
                    }
                    resolved_caps = {
                        key: value
                        for key, value in cap_hints.items()
                        if value is not None and (not isinstance(value, (int, float)) or value >= 0)
                    }
                    LOGGER.info(
                        "starting deferred embedder preload after readiness",  # pragma: no cover - telemetry
                        extra={
                            "event": "embedder-preload-background",
                            "presence_only": True,
                            "bootstrap_fast": bootstrap_fast_context,
                            "warmup_lite": warmup_lite_context,
                            "reason": reason,
                            "background_gate": background_dispatch_gate,
                        },
                    )
                    background_result = _bootstrap_embedder(
                        resolved_timeout,
                        stop_event=stop_event,
                        stage_budget=embedder_stage_budget,
                        budget=shared_timeout_coordinator,
                        budget_label="vector_seeding",
                        presence_probe=False,
                        presence_reason=reason,
                        bootstrap_fast=False,
                        schedule_background=True,
                        bootstrap_deadline=bootstrap_deadline,
                        precomputed_caps={
                            **resolved_caps,
                            "stage_budget": embedder_stage_budget,
                            "stage_deadline": cap_deadline,
                            "stage_wall_cap": background_cap,
                        },
                    )
                    placeholder_obj = job_snapshot.get(
                        "placeholder", _BOOTSTRAP_PLACEHOLDER
                    )
                    if background_result is None or background_result is placeholder_obj:
                        job_snapshot["background_full_warmup"] = False
                except Exception:  # pragma: no cover - background safety
                    LOGGER.debug("deferred embedder preload failed", exc_info=True)

            background_future = _BOOTSTRAP_SCHEDULER.schedule_embedder_background(
                _background_preload, reason=reason
            )
            if background_future is not None:
                job_snapshot["background_future"] = background_future
                job_snapshot["background_enqueued"] = True
                LOGGER.info(
                    "embedder preload deferred to background",  # pragma: no cover - telemetry
                    extra={
                        "event": "embedder-preload-background",
                        "reason": reason,
                        "bootstrap_fast": bootstrap_fast_context,
                        "warmup_lite": warmup_lite_context,
                    },
                )
            return job_snapshot

        if budget_window_missing and embedder_stage_budget is None and embedder_timeout is None:
            missing_budget_reason = presence_reason or "embedder_budget_unavailable"
            warmup_summary = warmup_summary or {}
            warmup_summary.update(
                {
                    "deferred": True,
                    "deferred_reason": missing_budget_reason,
                    "deferral_reason": missing_budget_reason,
                    "strict_timebox": warmup_timebox_cap,
                }
            )
            stage_controller.defer_step("embedder_preload", reason=missing_budget_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "vectorizer_preload", reason=missing_budget_reason
            )
            _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=missing_budget_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "background_loops", reason=f"embedder_placeholder:{missing_budget_reason}"
            )
            job_snapshot = _schedule_background_preload(
                missing_budget_reason, strict_timebox=warmup_timebox_cap
            )
            job_snapshot.update(
                {
                    "presence_available": False,
                    "presence_probe_timeout": False,
                    "budget_guarded": True,
                    "presence_only": True,
                    "budget_window_missing": True,
                    "forced_background": full_preload_requested,
                    "deferral_reason": missing_budget_reason,
                }
            )
            if warmup_summary:
                job_snapshot.setdefault("warmup_summary", warmup_summary)
            _BOOTSTRAP_EMBEDDER_JOB = job_snapshot
            _set_component_state("vector_seeding", "deferred")
            stage_controller.complete_step("embedder_preload", 0.0)
            return _BOOTSTRAP_PLACEHOLDER

        if fast_presence_enforced:
            strict_timebox_hint = embedder_stage_budget_hint
            if strict_timebox_hint is None:
                strict_timebox_hint = embedder_stage_budget
            return _defer_to_presence(
                presence_reason or "embedder_presence_fast_path",
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=True,
                strict_timebox=strict_timebox_hint,
                non_blocking_probe=True,
                resume_download=True,
            )

        budget_shortfall = any(
            candidate is not None
            and candidate > 0
            and candidate < estimated_preload_cost
            for candidate in (
                embedder_stage_budget,
                embedder_timeout,
                embedder_warmup_cap,
                remaining_bootstrap_window,
            )
        )

        exhausted_budget = any(
            candidate is not None and candidate <= 0
            for candidate in (embedder_stage_budget, embedder_timeout)
        )

        if exhausted_budget:
            exhausted_reason = presence_reason or "embedder_budget_exhausted"
            LOGGER.info(
                "embedder preload budget exhausted; deferring to background",
                extra={
                    "stage_budget": embedder_stage_budget,
                    "timeout": embedder_timeout,
                    "force_preload": full_preload_requested,
                    "bootstrap_fast": bootstrap_fast_context,
                    "warmup_lite": warmup_lite_context,
                    "event": "embedder-preload-deferred-budget-exhausted",
                },
            )
            stage_controller.defer_step("embedder_preload", reason=exhausted_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "vectorizer_preload", reason=exhausted_reason
            )
            _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=exhausted_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "background_loops", reason=f"embedder_placeholder:{exhausted_reason}"
            )
            job_snapshot = _schedule_background_preload(
                exhausted_reason, strict_timebox=warmup_timebox_cap
            )
            job_snapshot.update(
                {
                    "presence_available": False,
                    "presence_probe_timeout": False,
                    "budget_guarded": True,
                    "presence_only": True,
                    "budget_window_missing": budget_window_missing,
                    "forced_background": full_preload_requested,
                }
            )
            _BOOTSTRAP_EMBEDDER_JOB = job_snapshot
            _set_component_state("vector_seeding", "deferred")
            stage_controller.complete_step("embedder_preload", 0.0)
            return _BOOTSTRAP_PLACEHOLDER

        if budget_shortfall:
            shortfall_reason = "embedder_preload_budget_shortfall"
            LOGGER.info(
                "embedder preload budget shortfall detected; deferring to background",
                extra={
                    "stage_budget": embedder_stage_budget,
                    "timeout": embedder_timeout,
                    "warmup_cap": embedder_warmup_cap,
                    "estimated_preload_cost": estimated_preload_cost,
                    "event": "embedder-preload-deferred-budget",
                },
            )
            stage_controller.defer_step("embedder_preload", reason=shortfall_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "vectorizer_preload", reason=shortfall_reason
            )
            _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=shortfall_reason)
            _BOOTSTRAP_SCHEDULER.mark_partial(
                "background_loops", reason=f"embedder_placeholder:{shortfall_reason}"
            )
            job_snapshot = _schedule_background_preload(
                shortfall_reason, strict_timebox=warmup_timebox_cap
            )
            job_snapshot.update(
                {
                    "presence_available": False,
                    "presence_probe_timeout": False,
                    "budget_guarded": True,
                    "presence_only": True,
                    "budget_window_missing": budget_window_missing,
                }
            )
            _BOOTSTRAP_EMBEDDER_JOB = job_snapshot
            _set_component_state("vector_seeding", "deferred")
            stage_controller.complete_step("embedder_preload", 0.0)
            return _BOOTSTRAP_PLACEHOLDER

        if presence_only or skip_heavy:
            presence_reason = presence_reason or "embedder_presence_guard"
            return _defer_to_presence(
                presence_reason,
                budget_guarded=budget_guarded,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=warmup_timebox_cap,
                non_blocking_probe=non_blocking_presence_probe,
            )

        LOGGER.info(
            "embedder preload proceeding with full warmup",  # pragma: no cover - telemetry
            extra={
                "budget": embedder_stage_budget,
                "timeout": embedder_timeout,
                "bootstrap_fast": bootstrap_fast_context,
                "warmup_lite": warmup_lite_context,
                "event": "embedder-preload-full",
            },
        )

        def _build_guarded_stop_event() -> threading.Event:
            combined_stop_event = threading.Event()
            if stop_event is not None:

                def _mirror_stop() -> None:
                    stop_event.wait()
                    combined_stop_event.set()

                threading.Thread(
                    target=_mirror_stop,
                    name="embedder-preload-stop-forwarder",
                    daemon=True,
                ).start()

            deadlines: list[tuple[float, str]] = []
            now = time.monotonic()
            if enforced_timebox is not None:
                deadlines.append(
                    (now + enforced_timebox, "embedder_preload_timebox_expired")
                )
            if coordinator_remaining_budget is not None:
                deadlines.append(
                    (
                        now + max(coordinator_remaining_budget, 0.0),
                        "embedder_bootstrap_budget_exhausted",
                    )
                )
            if embedder_stage_budget_hint is not None and embedder_stage_budget_hint > 0:
                deadlines.append(
                    (
                        warmup_started + embedder_stage_budget_hint,
                        "embedder_stage_budget_deadline",
                    )
                )
            if embedder_stage_deadline is not None:
                deadlines.append(
                    (
                        embedder_stage_deadline,
                        "embedder_stage_deadline_elapsed",
                    )
                )
            if bootstrap_deadline is not None:
                deadlines.append((bootstrap_deadline, "embedder_bootstrap_deadline"))
            if stage_guard_timebox is not None:
                deadlines.append(
                    (
                        warmup_started + stage_guard_timebox,
                        "embedder_stage_timebox_guard",
                    )
                )

            if deadlines:

                def _deadline_guard() -> None:
                    nonlocal warmup_stop_reason
                    while not combined_stop_event.is_set():
                        deadline, reason = min(deadlines, key=lambda item: item[0])
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            warmup_stop_reason = warmup_stop_reason or reason
                            combined_stop_event.set()
                            return
                        if combined_stop_event.wait(min(remaining, 0.1)):
                            return

                threading.Thread(
                    target=_deadline_guard,
                    name="embedder-preload-stop-deadline",
                    daemon=True,
                ).start()

            return combined_stop_event

    def _guarded_embedder_warmup(
        *, join_timeout: float | None, join_cap: float | None
    ) -> tuple[Any, str | None]:
            combined_stop_event = _build_guarded_stop_event()
            warmup_result: dict[str, Any] = {}
            warmup_exc: list[BaseException] = []

            strict_warmup_cap = warmup_timebox_cap
            if strict_warmup_cap is None:
                strict_warmup_cap = enforced_timebox
            if join_cap is not None:
                strict_warmup_cap = (
                    join_cap
                    if strict_warmup_cap is None
                    else min(strict_warmup_cap, join_cap)
                )

            lite_default = bootstrap_context_active or budget_window_missing
            tight_cap = strict_warmup_cap is not None and strict_warmup_cap <= minimum_presence_window
            missing_stage_budget = embedder_stage_budget is None
            stage_budget_tight = (
                embedder_stage_budget is not None
                and strict_warmup_cap is not None
                and embedder_stage_budget <= strict_warmup_cap
            )
            lite_default = lite_default or missing_stage_budget or tight_cap or stage_budget_tight
            heavy_requested = bool(full_embedder_preload or force_embedder_preload)
            heavy_budget_available = (
                (embedder_stage_budget is not None and embedder_stage_budget > (strict_warmup_cap or 0))
                or (embedder_timeout is not None and embedder_timeout > (strict_warmup_cap or 0))
            )

            if lite_default and not (heavy_requested and heavy_budget_available):
                warmup_stop_reason_local = (
                    warmup_stop_reason
                    or (
                        "embedder_preload_bootstrap_context"
                        if bootstrap_context_active
                        else "embedder_preload_warmup_lite_default"
                    )
                )
                warmup_stop_reason = warmup_stop_reason or warmup_stop_reason_local
                return None, warmup_stop_reason

            if (
                embedder_stage_budget is None
                and embedder_timeout is None
                and embedder_stage_budget_hint is not None
            ):
                warmup_stop_reason_local = "embedder_preload_budget_hint_only"
                warmup_stop_reason = warmup_stop_reason or warmup_stop_reason_local
                return None, warmup_stop_reason

            def _run_warmup() -> None:
                try:
                    effective_stage_budget = embedder_stage_budget
                    if (
                        warmup_timebox_cap is not None
                        and (
                            effective_stage_budget is None
                            or effective_stage_budget > warmup_timebox_cap
                        )
                    ):
                        effective_stage_budget = warmup_timebox_cap
                    warmup_result["result"] = _bootstrap_embedder(
                        embedder_timeout,
                        stop_event=combined_stop_event,
                        stage_budget=effective_stage_budget,
                        budget=shared_timeout_coordinator,
                        budget_label="vector_seeding",
                        bootstrap_fast=bootstrap_fast_context,
                        bootstrap_deadline=bootstrap_deadline,
                    )
                except BaseException as exc:  # pragma: no cover - defensive guard
                    warmup_exc.append(exc)

            warmup_thread = threading.Thread(
                target=_run_warmup,
                name="embedder-preload-warmup",
                daemon=True,
            )
            warmup_thread.start()

            if join_timeout is not None and join_timeout <= 0:
                warmup_stop_reason = warmup_stop_reason or (
                    warmup_join_cap_reason
                    if (
                        warmup_join_ceiling is not None
                        and join_cap is not None
                        and join_cap <= warmup_join_ceiling
                    )
                    else "embedder_preload_warmup_cap_exceeded"
                )
                combined_stop_event.set()
                warmup_thread.join(0.05)
                return None, warmup_stop_reason
            warmup_thread.join(join_timeout)

            if warmup_thread.is_alive():
                combined_stop_event.set()
                warmup_stop_reason = warmup_stop_reason or (
                    warmup_join_cap_reason
                    if (
                        warmup_join_ceiling is not None
                        and join_cap is not None
                        and join_cap <= warmup_join_ceiling
                    )
                    else "embedder_preload_warmup_cap_exceeded"
                )
                warmup_thread.join(0.05)
                return None, warmup_stop_reason

            if warmup_exc:
                raise warmup_exc[0]

            return warmup_result.get("result"), None

        warmup_join_timeout, warmup_join_cap = _derive_warmup_join_timeout(
            warmup_timebox_cap=warmup_timebox_cap,
            enforced_timebox=enforced_timebox,
            warmup_started=warmup_started,
            stage_guard_timebox=stage_guard_timebox,
            embedder_stage_budget_hint=embedder_stage_budget_hint,
            warmup_hard_cap=warmup_hard_cap,
            warmup_join_ceiling=warmup_join_ceiling,
        )
        strict_join_cap_candidates = [
            candidate
            for candidate in (
                warmup_join_cap,
                stage_guard_timebox,
                embedder_stage_budget_hint,
                embedder_stage_deadline_remaining,
                stage_deadline_hint_remaining,
                warmup_timebox_cap,
                enforced_timebox,
                warmup_hard_cap,
            )
            if candidate is not None and candidate > 0
        ]
        strict_join_cap = min(strict_join_cap_candidates) if strict_join_cap_candidates else None
        if strict_join_cap is not None:
            if warmup_join_timeout is None or warmup_join_timeout > strict_join_cap:
                warmup_summary = warmup_summary or {}
                warmup_summary.setdefault("deferred", True)
                warmup_summary.setdefault(
                    "deferral_reason", "embedder_preload_join_timebox_exceeded"
                )
                warmup_summary.setdefault(
                    "deferred_reason", "embedder_preload_join_timebox_exceeded"
                )
                warmup_summary.setdefault("stage", "deferred-timebox")
                warmup_summary.setdefault("strict_timebox", strict_join_cap)
                warmup_summary.setdefault("stage_budget", embedder_stage_budget_hint)
                return _defer_to_presence(
                    "embedder_preload_join_timebox_exceeded",
                    budget_guarded=True,
                    budget_window_missing=budget_window_missing,
                    forced_background=full_preload_requested,
                    strict_timebox=strict_join_cap,
                    non_blocking_probe=non_blocking_presence_probe,
                    resume_download=True,
                )
            warmup_join_cap = (
                min(strict_join_cap, warmup_join_cap)
                if warmup_join_cap is not None
                else strict_join_cap
            )
            warmup_join_timeout = (
                strict_join_cap
                if warmup_join_timeout is None
                else min(warmup_join_timeout, strict_join_cap)
            )
        elif warmup_join_timeout is None:
            warmup_summary = warmup_summary or {}
            warmup_summary.setdefault("deferred", True)
            warmup_summary.setdefault(
                "deferral_reason", "embedder_preload_join_timebox_missing"
            )
            warmup_summary.setdefault(
                "deferred_reason", "embedder_preload_join_timebox_missing"
            )
            warmup_summary.setdefault("stage", "deferred-timebox")
            warmup_summary.setdefault("stage_budget", embedder_stage_budget_hint)
            fallback_timebox_hint = (
                enforced_timebox
                if enforced_timebox is not None
                else (warmup_timebox_cap or stage_guard_timebox)
            )
            warmup_summary.setdefault("strict_timebox", fallback_timebox_hint)
            return _defer_to_presence(
                "embedder_preload_join_timebox_missing",
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=fallback_timebox_hint,
                non_blocking_probe=non_blocking_presence_probe,
                resume_download=True,
            )
        warmup_result, warmup_timeout_reason = _guarded_embedder_warmup(
            join_timeout=warmup_join_timeout, join_cap=warmup_join_cap
        )
        if warmup_timeout_reason:
            warmup_summary = warmup_summary or {}
            warmup_summary.setdefault("deferred", True)
            strict_timebox_value = warmup_hard_cap or enforced_timebox or warmup_join_cap
            warmup_summary.setdefault("strict_timebox", strict_timebox_value)
            warmup_deferral_reason = (
                warmup_timeout_reason
                if warmup_timeout_reason != "embedder_preload_timebox_expired"
                else "embedder_preload_warmup_cap_exceeded"
            )
            if fallback_timebox_applied and warmup_deferral_reason in {
                "embedder_preload_warmup_cap_exceeded",
                "embedder_preload_timebox_expired",
            }:
                warmup_deferral_reason = (
                    fallback_timebox_reason or "embedder_preload_fallback_warmup_cap_exceeded"
                )

            warmup_summary["deferral_reason"] = warmup_deferral_reason
            warmup_summary.setdefault("deferred_reason", warmup_deferral_reason)
            warmup_summary.setdefault("stage_budget", embedder_stage_budget_hint)
            warmup_summary.setdefault("stage", "deferred-timebox")
            warmup_summary.setdefault("presence_available", False)
            warmup_summary.setdefault("presence_probe_timeout", False)
            resume_download = warmup_deferral_reason in {
                "embedder_stage_timebox_guard",
                "embedder_preload_timebox_expired",
                "embedder_preload_warmup_cap_exceeded",
                "embedder_preload_fallback_timebox",
                "embedder_preload_fallback_warmup_cap_exceeded",
            }
            return _defer_to_presence(
                warmup_deferral_reason,
                budget_guarded=True,
                budget_window_missing=budget_window_missing,
                forced_background=full_preload_requested,
                strict_timebox=strict_timebox_value,
                non_blocking_probe=non_blocking_presence_probe,
                resume_download=resume_download,
            )

        _BOOTSTRAP_EMBEDDER_JOB = (_BOOTSTRAP_EMBEDDER_JOB or {}) | {
            "result": warmup_result
        }
        _BOOTSTRAP_SCHEDULER.mark_ready(
            "vector_seeding", reason="embedder_preload_complete"
        )
        _set_component_state("vector_seeding", "ready")
        stage_controller.complete_step("embedder_preload", time.monotonic() - warmup_started)
        return warmup_result

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
            vector_bootstrap_hint_holder["vector"] = bool(
                vector_warmup_requested and _is_vector_bootstrap_heavy(builder)
            )
        except Exception:  # pragma: no cover - advisory only
            LOGGER.debug("failed to inspect context builder for vector-heavy hint", exc_info=True)
        if vector_bootstrap_hint_holder["vector"] and vector_warmup_requested:
            vector_env_snapshot.update(_apply_vector_env("context_builder_hint"))
            try:
                _coding_bot_interface._BOOTSTRAP_STATE.vector_heavy = True
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("unable to propagate vector-heavy bootstrap flag", exc_info=True)
            timeout_policy = enforce_bootstrap_timeout_policy(logger=LOGGER)
            _apply_timeout_policy_snapshot(timeout_policy)
            timeout_policy_summary = _render_timeout_policy(timeout_policy)
            print(f"bootstrap timeout policy: {timeout_policy_summary}", flush=True)
        elif vector_bootstrap_hint_holder["vector"] and not vector_warmup_requested:
            LOGGER.info(
                "vector-heavy hint detected but heavy warmup not requested; remaining in warmup-lite mode",
                extra={
                    "event": "vector-warmup-lite-default",
                    "force_vector_warmup": force_vector_warmup,
                    "bootstrap_fast": bootstrap_fast_context,
                },
            )
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

    if embedder_preload_stage_blocked and embedder_preload_enabled and not force_embedder_preload:
        embedder_preload_enabled = False
        embedder_preload_skip_reason = "embedder_preload_stage_budget_exhausted"
        LOGGER.info(
            "embedder preload skipped due to exhausted stage budget",
            extra={
                "stage_budget": embedder_stage_budget_hint,
                "event": embedder_preload_skip_reason,
            },
        )
        embedder_preload_deferred = True
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(
            reason=embedder_preload_skip_reason
        )
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "vectorizer_preload", reason=embedder_preload_skip_reason
        )
        _BOOTSTRAP_SCHEDULER.mark_partial(
            "background_loops", reason=f"embedder_placeholder:{embedder_preload_skip_reason}"
        )
        stage_controller.defer_step(
            "embedder_preload", reason=embedder_preload_skip_reason
        )

    if embedder_preload_enabled:
        runner.add(
            _BootstrapTask(
                name="vectorizer_preload",
                fn=_task_embedder,
                component="vectorizer_preload",
                critical=False,
                background=True,
            )
        )
    else:
        skip_reason = embedder_preload_skip_reason or "embedder_preload_deferred_bootstrap_fast"
        LOGGER.info(
            "embedder preload deferred; using lazy activation",
            extra={
                "bootstrap_mode": bootstrap_mode_env,
                "bootstrap_fast": bootstrap_fast_context,
                "force_vector_warmup": force_vector_warmup,
                "stage_budget": embedder_stage_budget_hint,
                "skip_reason": skip_reason,
            },
        )
        _BOOTSTRAP_SCHEDULER.mark_embedder_deferred(reason=skip_reason)
        if not embedder_preload_deferred:
            stage_controller.defer_step("embedder_preload", reason=skip_reason)
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
                if not vector_heavy and vector_warmup_requested:
                    vector_heavy = _is_vector_bootstrap_heavy(context_builder)
                if not vector_heavy and vector_warmup_requested:
                    vector_heavy = _is_vector_bootstrap_heavy(ModelAutomationPipeline)
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("unable to inspect vector_heavy flag", exc_info=True)

            vector_heavy = bool(vector_heavy and vector_warmup_requested)

            if vector_heavy and vector_warmup_requested:
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
        _BOOTSTRAP_EMBEDDER_READY.set()
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
