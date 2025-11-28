"""Shared timeout policy helpers for bootstrap entry points."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, MutableMapping, Any

_SHARED_EVENT_BUS = None

LOGGER = logging.getLogger(__name__)
_ADAPTIVE_TIMEOUT_CONTEXT: dict[str, object] = {}

_BOOTSTRAP_TIMEOUT_MINIMUMS: dict[str, float] = {
    "MENACE_BOOTSTRAP_WAIT_SECS": 240.0,
    "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS": 240.0,
    "BOOTSTRAP_STEP_TIMEOUT": 240.0,
    "BOOTSTRAP_VECTOR_STEP_TIMEOUT": 240.0,
    "PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS": 420.0,
    "PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS": 300.0,
    "PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS": 240.0,
    "PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS": 240.0,
}
_COMPONENT_TIMEOUT_MINIMUMS: dict[str, float] = {
    "vectorizers": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS"],
    "retrievers": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS"],
    "db_indexes": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS"],
    "orchestrator_state": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS"],
    "pipeline_config": _BOOTSTRAP_TIMEOUT_MINIMUMS["BOOTSTRAP_STEP_TIMEOUT"],
    "background_loops": _BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_WAIT_SECS"],
}
_LAST_BOOTSTRAP_GUARD: dict[str, object] | None = None
_OVERRIDE_ENV = "MENACE_BOOTSTRAP_TIMEOUT_ALLOW_UNSAFE"
_TIMEOUT_STATE_ENV = "MENACE_BOOTSTRAP_TIMEOUT_STATE"
_TIMEOUT_STATE_PATH = Path(
    os.getenv(_TIMEOUT_STATE_ENV, os.path.expanduser("~/.menace_bootstrap_timeout_state.json"))
)
_COMPONENT_ENV_MAPPING = {
    "vectorizers": "PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS",
    "retrievers": "PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS",
    "db_indexes": "PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS",
    "orchestrator_state": "PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS",
    "pipeline_config": "BOOTSTRAP_STEP_TIMEOUT",
    "background_loops": "MENACE_BOOTSTRAP_WAIT_SECS",
}
_OVERRUN_TOLERANCE = 1.05
_OVERRUN_STREAK_THRESHOLD = 2
_DECAY_RATIO = 0.9
_DEFAULT_HEARTBEAT_MAX_AGE = 120.0
_DEFAULT_LOAD_THRESHOLD = 1.35
_DEFAULT_GUARD_MAX_DELAY = 90.0
_DEFAULT_GUARD_INTERVAL = 5.0
_BOOTSTRAP_HEARTBEAT_ENV = "MENACE_BOOTSTRAP_WATCHDOG_PATH"
_BOOTSTRAP_HEARTBEAT_MAX_AGE_ENV = "MENACE_BOOTSTRAP_HEARTBEAT_MAX_AGE"
_BOOTSTRAP_LOAD_THRESHOLD_ENV = "MENACE_BOOTSTRAP_LOAD_THRESHOLD"
_BOOTSTRAP_HEARTBEAT_PATH = Path(
    os.getenv(_BOOTSTRAP_HEARTBEAT_ENV, "/tmp/menace_bootstrap_watchdog.json")
)


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _host_load_average() -> float | None:
    """Return the normalized 1-minute load average when available."""

    try:
        load = os.getloadavg()[0]
    except (AttributeError, OSError):
        return None
    cpus = os.cpu_count() or 1
    return load / max(float(cpus), 1.0)


def _heartbeat_path() -> Path:
    override = os.getenv(_BOOTSTRAP_HEARTBEAT_ENV)
    if override:
        return Path(override)
    return _BOOTSTRAP_HEARTBEAT_PATH


def emit_bootstrap_heartbeat(payload: Mapping[str, Any]) -> None:
    """Persist and broadcast a bootstrap watchdog heartbeat."""

    timestamp = time.time()
    enriched = dict(payload)
    enriched.setdefault("ts", timestamp)
    enriched.setdefault("pid", os.getpid())
    enriched.setdefault("host", socket.gethostname())
    enriched.setdefault("host_load", _host_load_average())

    try:
        path = _heartbeat_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(enriched, indent=2, sort_keys=True))
    except Exception:
        LOGGER.debug("failed to persist bootstrap heartbeat", exc_info=True)

    global _SHARED_EVENT_BUS
    if _SHARED_EVENT_BUS is None:
        try:  # pragma: no cover - optional runtime import
            from shared_event_bus import event_bus as _SHARED_EVENT_BUS  # type: ignore
        except Exception:
            _SHARED_EVENT_BUS = None
    if _SHARED_EVENT_BUS is not None:
        try:
            _SHARED_EVENT_BUS.publish("bootstrap.watchdog", dict(enriched))
        except Exception:
            LOGGER.debug("failed to broadcast bootstrap heartbeat", exc_info=True)


def read_bootstrap_heartbeat(max_age: float | None = None) -> Mapping[str, Any] | None:
    """Return the most recent heartbeat when it is fresh."""

    max_age = max_age if max_age is not None else _parse_float(
        os.getenv(_BOOTSTRAP_HEARTBEAT_MAX_AGE_ENV)
    )
    if not max_age:
        max_age = _DEFAULT_HEARTBEAT_MAX_AGE
    try:
        payload = json.loads(_heartbeat_path().read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None

    ts = payload.get("ts")
    if not isinstance(ts, (float, int)):
        return None
    if time.time() - float(ts) > max_age:
        return None
    return payload


def _record_bootstrap_guard(delay: float, budget_scale: float, *, source: str) -> None:
    """Persist guard telemetry for downstream budget scaling."""

    global _LAST_BOOTSTRAP_GUARD
    _LAST_BOOTSTRAP_GUARD = {
        "delay": float(delay),
        "budget_scale": float(budget_scale),
        "source": source,
        "ts": time.time(),
    }


def broadcast_timeout_floors(
    *,
    source: str,
    component_floors: Mapping[str, float] | None = None,
    timeout_floors: Mapping[str, float] | None = None,
    guard_context: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Persist the latest timeout floors so peers can inherit escalations."""

    state = _load_timeout_state()
    host_key = _state_host_key()
    host_state = state.get(host_key, {}) if isinstance(state, dict) else {}
    if not isinstance(host_state, dict):  # pragma: no cover - defensive
        host_state = {}

    timeout_floors = dict(timeout_floors or load_escalated_timeout_floors())
    component_floors = dict(component_floors or load_component_timeout_floors())
    guard_context = dict(guard_context or get_bootstrap_guard_context())
    snapshot = {
        "host": host_key,
        "pid": os.getpid(),
        "ts": time.time(),
        "source": source,
        "guard_delay": guard_context.get("delay") or guard_context.get("guard_delay"),
        "guard_budget_scale": guard_context.get("budget_scale"),
        "floors": timeout_floors,
        "component_floors": component_floors,
    }

    host_state.update(timeout_floors)
    host_state["component_floors"] = component_floors
    host_state["last_broadcast"] = snapshot
    host_state["updated_at"] = snapshot["ts"]
    state = state if isinstance(state, dict) else {}
    state[host_key] = host_state
    _save_timeout_state(state)

    LOGGER.info(
        "shared bootstrap timeout floors broadcast",
        extra={"broadcast": snapshot, "state_path": str(_TIMEOUT_STATE_PATH)},
    )

    return snapshot


def get_bootstrap_guard_context() -> Mapping[str, object]:
    """Return the most recent guard metadata when available."""

    return dict(_LAST_BOOTSTRAP_GUARD or {})


def wait_for_bootstrap_quiet_period(
    logger: logging.Logger,
    *,
    max_delay: float | None = None,
    load_threshold: float | None = None,
    poll_interval: float = _DEFAULT_GUARD_INTERVAL,
    ignore_pid: int | None = None,
) -> tuple[float, float]:
    """Delay heavy bootstrap stages when peers are active or host load is high.

    Returns a tuple of ``(sleep_seconds, budget_scale)`` where ``budget_scale``
    can be applied to stage budgets to proactively extend deadlines when the
    guard was forced to wait.
    """

    target_delay = max_delay if max_delay is not None else _parse_float(
        os.getenv("MENACE_BOOTSTRAP_GUARD_MAX_DELAY")
    )
    if not target_delay:
        target_delay = _DEFAULT_GUARD_MAX_DELAY
    threshold = load_threshold if load_threshold is not None else _parse_float(
        os.getenv(_BOOTSTRAP_LOAD_THRESHOLD_ENV)
    )
    if threshold is None:
        threshold = _DEFAULT_LOAD_THRESHOLD
    ignore_pid = os.getpid() if ignore_pid is None else ignore_pid
    deadline = time.monotonic() + target_delay
    slept = 0.0
    budget_scale = 1.0

    while time.monotonic() < deadline:
        heartbeat = read_bootstrap_heartbeat()
        normalized_load = _host_load_average()
        peer_active = False
        if heartbeat:
            peer_active = int(heartbeat.get("pid", -1)) != int(ignore_pid)
            normalized_load = heartbeat.get("host_load", normalized_load)
        overloaded = normalized_load is not None and normalized_load > threshold

        if not peer_active and not overloaded:
            break

        remaining_window = max(deadline - time.monotonic(), 0.0)
        sleep_for = min(poll_interval, remaining_window)
        slept += sleep_for
        if normalized_load and threshold:
            budget_scale = max(budget_scale, min(normalized_load / threshold, 2.0))

        logger.info(
            "delaying bootstrap to avoid contention",
            extra={
                "event": "bootstrap-guard-delay",
                "sleep_for": round(sleep_for, 2),
                "peer_active": peer_active,
                "normalized_load": normalized_load,
                "threshold": threshold,
                "budget_scale": round(budget_scale, 3),
            },
        )
        time.sleep(sleep_for)

    _record_bootstrap_guard(slept, budget_scale, source="bootstrap_guard")
    return slept, budget_scale


def build_progress_signal_hook(
    *, namespace: str, run_id: str | None = None
) -> Callable[[Mapping[str, object]], None]:
    """Return a hook that broadcasts SharedTimeoutCoordinator progress."""

    def _signal(record: Mapping[str, object]) -> None:
        enriched: dict[str, Any] = {
            "namespace": namespace,
            "run_id": run_id or f"{namespace}-{os.getpid()}",
            **record,
        }
        emit_bootstrap_heartbeat(enriched)

    return _signal


def _load_timeout_state() -> dict:
    try:
        return json.loads(_TIMEOUT_STATE_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return {}


def _save_timeout_state(state: Mapping[str, object]) -> None:
    try:
        _TIMEOUT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TIMEOUT_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))
    except OSError:
        LOGGER.exception("failed to persist timeout state", extra={"path": str(_TIMEOUT_STATE_PATH)})


def _state_host_key() -> str:
    return socket.gethostname() or "unknown-host"


def _record_adaptive_context(context: Mapping[str, object]) -> None:
    _ADAPTIVE_TIMEOUT_CONTEXT.clear()
    _ADAPTIVE_TIMEOUT_CONTEXT.update(context)


def get_adaptive_timeout_context() -> Mapping[str, object]:
    """Return the last computed adaptive timeout context."""

    return dict(_ADAPTIVE_TIMEOUT_CONTEXT)


def load_escalated_timeout_floors() -> dict[str, float]:
    """Return timeout floors that include host-scoped persisted escalations."""

    minimums = dict(_BOOTSTRAP_TIMEOUT_MINIMUMS)
    state = _load_timeout_state()
    host_state = state.get(_state_host_key(), {}) if isinstance(state, dict) else {}

    component_overruns = (
        host_state.get("component_overruns", {}) if isinstance(host_state, dict) else {}
    )
    decay_streak = int(host_state.get("success_streak", 0) or 0)
    adaptive_notes: dict[str, object] = {
        "component_overruns": component_overruns,
        "decay_streak": decay_streak,
    }
    state_updated = False

    for env_var, default_minimum in list(minimums.items()):
        try:
            host_value = float(host_state.get(env_var, default_minimum))
        except (TypeError, ValueError):
            host_value = default_minimum
        minimums[env_var] = max(default_minimum, host_value)

    for component, env_var in _COMPONENT_ENV_MAPPING.items():
        overrun_state = component_overruns.get(component, {}) if isinstance(component_overruns, dict) else {}
        overruns = int(overrun_state.get("overruns", 0) or 0)
        suggested_floor = _parse_float(str(overrun_state.get("suggested_floor")))
        if overruns >= _OVERRUN_STREAK_THRESHOLD and suggested_floor:
            adjusted_floor = max(minimums.get(env_var, 0.0), float(suggested_floor))
            if adjusted_floor > minimums.get(env_var, 0.0):
                minimums[env_var] = adjusted_floor
                adaptive_notes.setdefault("adaptive_floors", {})[env_var] = {
                    "source": component,
                    "reason": "repeated_overruns",
                    "overruns": overruns,
                    "suggested_floor": adjusted_floor,
                }

    if decay_streak > 0:
        for env_var, baseline in _BOOTSTRAP_TIMEOUT_MINIMUMS.items():
            current = minimums.get(env_var, baseline)
            decayed = max(baseline, current * (_DECAY_RATIO**min(decay_streak, 3)))
            if decayed < current:
                minimums[env_var] = decayed
                state_updated = True
        adaptive_notes["decayed"] = True
        host_state["success_streak"] = max(decay_streak - 1, 0)

    if state_updated:
        state = state if isinstance(state, dict) else {}
        state[_state_host_key()] = host_state
        _save_timeout_state(state)
    _record_adaptive_context(adaptive_notes)

    adaptive_floors = adaptive_notes.get("adaptive_floors")
    if adaptive_floors:
        LOGGER.info(
            "adaptive bootstrap timeout floors applied",
            extra={
                "event": "adaptive-timeout-floors",
                "adaptive_floors": adaptive_floors,
                "state_path": str(_TIMEOUT_STATE_PATH),
            },
        )

    return minimums


def load_component_timeout_floors() -> dict[str, float]:
    """Return per-component timeout floors with host-level overrides applied."""

    component_floors = dict(_COMPONENT_TIMEOUT_MINIMUMS)
    state = _load_timeout_state()
    host_state = state.get(_state_host_key(), {}) if isinstance(state, dict) else {}
    host_component_floors = (
        host_state.get("component_floors", {}) if isinstance(host_state, dict) else {}
    )

    for component, default_minimum in list(component_floors.items()):
        try:
            host_value = float(host_component_floors.get(component, default_minimum))
        except (TypeError, ValueError):
            host_value = default_minimum
        component_floors[component] = max(default_minimum, host_value)

    return component_floors


def _recent_peer_activity(max_age: float = 900.0) -> list[Mapping[str, object]]:
    state = _load_timeout_state()
    if not isinstance(state, Mapping):
        return []

    now = time.time()
    peers: list[Mapping[str, object]] = []
    for host, meta in state.items():
        if not isinstance(meta, Mapping):
            continue
        broadcast = meta.get("last_broadcast") if isinstance(meta, Mapping) else {}
        if not isinstance(broadcast, Mapping):
            broadcast = {}
        ts = broadcast.get("ts") or meta.get("updated_at")
        try:
            ts_val = float(ts) if ts is not None else None
        except (TypeError, ValueError):
            ts_val = None
        if ts_val is None or now - ts_val > max_age:
            continue
        peers.append({
            "host": host,
            "pid": broadcast.get("pid"),
            "ts": ts_val,
            "guard_delay": broadcast.get("guard_delay"),
            "budget_scale": broadcast.get("guard_budget_scale"),
            "source": broadcast.get("source"),
        })

    return peers


def load_persisted_bootstrap_wait(vector_heavy: bool = False) -> float | None:
    """Return the last persisted adaptive bootstrap wait window when available."""

    state = _load_timeout_state()
    host_state = state.get(_state_host_key(), {}) if isinstance(state, dict) else {}
    windows = host_state.get("bootstrap_wait_windows", {}) if isinstance(host_state, dict) else {}
    key = "vector" if vector_heavy else "general"
    window = windows.get(key, {}) if isinstance(windows, Mapping) else {}
    try:
        value = window.get("timeout")  # type: ignore[assignment]
    except AttributeError:  # pragma: no cover - defensive against malformed state
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def persist_bootstrap_wait_window(
    timeout: float | None,
    *,
    vector_heavy: bool,
    source: str,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Persist the most recent adaptive bootstrap wait decision for this host."""

    state = _load_timeout_state()
    host_key = _state_host_key()
    host_state = state.get(host_key, {}) if isinstance(state, dict) else {}
    if not isinstance(host_state, dict):
        host_state = {}

    windows = host_state.get("bootstrap_wait_windows", {})
    if not isinstance(windows, dict):  # pragma: no cover - defensive
        windows = {}

    key = "vector" if vector_heavy else "general"
    window_state: dict[str, object] = {
        "timeout": timeout,
        "source": source,
        "ts": time.time(),
    }
    if metadata:
        window_state.update({f"meta.{k}": v for k, v in metadata.items()})

    windows[key] = window_state
    host_state["bootstrap_wait_windows"] = windows
    state = state if isinstance(state, dict) else {}
    state[host_key] = host_state
    _save_timeout_state(state)



def compute_prepare_pipeline_component_budgets(
    *,
    component_floors: Mapping[str, float] | None = None,
    telemetry: Mapping[str, object] | None = None,
    load_average: float | None = None,
    pipeline_complexity: Mapping[str, object] | None = None,
    host_telemetry: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Return proactive per-component budgets for prepare_pipeline gates."""

    telemetry = telemetry or {}
    complexity = pipeline_complexity or {}
    guard_context = get_bootstrap_guard_context()
    host_state = _load_timeout_state()
    host_key = _state_host_key()
    floors = dict(component_floors or load_component_timeout_floors())
    guard_scale = float(guard_context.get("budget_scale") or 1.0)
    if guard_scale != 1.0:
        floors = {key: value * guard_scale for key, value in floors.items()}
    host_telemetry = dict(host_telemetry or {})
    if isinstance(host_state, Mapping):
        host_floors = (host_state.get(host_key, {}) or {}).get("component_floors", {})
        previous_budgets = (host_state.get(host_key, {}) or {}).get(
            "last_component_budgets", {}
        )
        for key, value in (host_floors or {}).items():
            try:
                floors[key] = max(floors.get(key, 0.0), float(value))
            except (TypeError, ValueError):
                continue
        for key, value in (previous_budgets or {}).items():
            try:
                floors[key] = max(floors.get(key, 0.0), float(value))
            except (TypeError, ValueError):
                continue

    if load_average is None:
        load_average = _parse_float(str(host_telemetry.get("host_load")))
    scale = _host_load_scale(load_average)

    adaptive_floors: dict[str, dict[str, float | int | str]] = {}

    def _apply_adaptive_floor(
        component: str,
        suggested_floor: float,
        *,
        reason: str,
        overruns: int | None = None,
    ) -> None:
        baseline = floors.get(component, _COMPONENT_TIMEOUT_MINIMUMS.get(component, 0.0))
        adjusted_floor = max(baseline, suggested_floor)
        if adjusted_floor > baseline:
            floors[component] = adjusted_floor
            adaptive_floors[component] = {
                "previous": baseline,
                "floor": adjusted_floor,
                "reason": reason,
            }
            if overruns is not None:
                adaptive_floors[component]["overruns"] = overruns

    def _count(value: object) -> int:
        if isinstance(value, (list, tuple, set, frozenset)):
            return len(value)
        if isinstance(value, Mapping):
            return len(value)
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    def _size_scale(bytes_size: object) -> float:
        try:
            value = float(bytes_size)
        except (TypeError, ValueError):
            return 1.0
        gigabytes = max(value, 0.0) / (1024**3)
        return 1.0 + min(gigabytes / 5.0, 0.75)

    component_complexity: dict[str, float] = {}
    component_complexity["vectorizers"] = 1.0 + max(0, _count(complexity.get("vectorizers")) - 1) * 0.35
    component_complexity["retrievers"] = 1.0 + max(0, _count(complexity.get("retrievers")) - 1) * 0.25
    component_complexity["db_indexes"] = 1.0 + max(0, _count(complexity.get("db_indexes")) - 1) * 0.2
    if "db_index_bytes" in complexity:
        component_complexity["db_indexes"] *= _size_scale(complexity.get("db_index_bytes"))
    component_complexity["background_loops"] = 1.0 + max(0, _count(complexity.get("background_loops"))) * 0.15
    component_complexity["pipeline_config"] = 1.0 + max(
        0, _count(complexity.get("pipeline_config_sections")) - 1
    ) * 0.1
    component_complexity["orchestrator_state"] = 1.0 + max(
        0, _count(complexity.get("orchestrator_components")) - 1
    ) * 0.1

    host_overruns = {}
    host_state_details = host_state.get(host_key, {}) if isinstance(host_state, dict) else {}
    if isinstance(host_state, dict):
        host_overruns = host_state_details.get("component_overruns", {}) or {}

    if isinstance(host_overruns, Mapping):
        for component, meta in host_overruns.items():
            streak = int(meta.get("overruns", 0) or 0)
            suggested_floor = _parse_float(str(meta.get("suggested_floor")))
            if streak >= _OVERRUN_STREAK_THRESHOLD and suggested_floor is not None:
                _apply_adaptive_floor(
                    component,
                    float(suggested_floor),
                    reason="persisted_overruns",
                    overruns=streak,
                )

    budgets = {
        key: value * scale * component_complexity.get(key, 1.0)
        for key, value in floors.items()
    }

    def _apply_overruns(overruns: Mapping[str, Mapping[str, float | int]]) -> None:
        for component, meta in overruns.items():
            base = budgets.get(component, floors.get(component, 0.0))
            max_elapsed = float(meta.get("max_elapsed", 0.0) or 0.0)
            expected_floor = float(meta.get("expected_floor", 0.0) or 0.0)
            suggested_floor = _parse_float(str(meta.get("suggested_floor")))
            streak = int(meta.get("overruns", 0) or 0)

            candidates = [base, expected_floor * scale]
            if max_elapsed:
                candidates.append(max_elapsed * scale + 30.0)
            if suggested_floor is not None:
                candidates.append(suggested_floor * scale)
            if streak >= _OVERRUN_STREAK_THRESHOLD and expected_floor:
                candidates.append(expected_floor * scale * 1.1)

            if streak >= _OVERRUN_STREAK_THRESHOLD:
                adaptive_floor = suggested_floor
                if adaptive_floor is None and (max_elapsed or expected_floor):
                    adaptive_floor = max(max_elapsed + 30.0, expected_floor * 1.1)
                if adaptive_floor is not None:
                    _apply_adaptive_floor(
                        component,
                        float(adaptive_floor),
                        reason="telemetry_overruns",
                        overruns=streak,
                    )
            budgets[component] = max(c for c in candidates if c is not None)

    telemetry_overruns = _summarize_component_overruns(telemetry or {})
    _apply_overruns(telemetry_overruns)
    if isinstance(host_overruns, Mapping):
        _apply_overruns(host_overruns)  # type: ignore[arg-type]

    adaptive_inputs = {
        "host": host_key,
        "load_scale": scale,
        "load_average": load_average,
        "component_complexity": component_complexity,
        "telemetry_overruns": telemetry_overruns,
        "floors": floors,
        "adaptive_floors": adaptive_floors,
    }
    _record_adaptive_context(adaptive_inputs)

    host_state_details = host_state_details if isinstance(host_state_details, dict) else {}
    if adaptive_floors:
        persisted_floors = (
            host_state_details.get("component_floors", {}) if isinstance(host_state_details, Mapping) else {}
        )
        persisted_floors = dict(persisted_floors) if isinstance(persisted_floors, Mapping) else {}
        for component, meta in adaptive_floors.items():
            persisted_floors[component] = float(
                meta.get("floor", floors.get(component, 0.0))
            )
        host_state_details["component_floors"] = persisted_floors
    host_state_details.update(
        {
            "last_component_budgets": budgets,
            "last_component_budget_inputs": adaptive_inputs,
            "updated_at": time.time(),
        }
    )
    if isinstance(host_state, dict):
        host_state[host_key] = host_state_details
    else:  # pragma: no cover - defensive against corrupted state
        host_state = {host_key: host_state_details}
    _save_timeout_state(host_state)

    if adaptive_floors:
        LOGGER.info(
            "adaptive component floors applied",
            extra={
                "event": "adaptive-component-floors",
                "floors": adaptive_floors,
                "state_path": str(_TIMEOUT_STATE_PATH),
            },
        )

    LOGGER.info(
        "prepared adaptive component budgets",
        extra={
            "event": "adaptive-component-budgets",
            "budgets": budgets,
            "component_complexity": component_complexity,
            "load_scale": scale,
            "host_load": load_average,
            "host_overruns": host_overruns,
            "telemetry_overruns": telemetry_overruns,
            "state_path": str(_TIMEOUT_STATE_PATH),
        },
    )

    return budgets


def _collect_timeout_telemetry() -> Mapping[str, object]:
    try:
        from coding_bot_interface import _PREPARE_PIPELINE_WATCHDOG

        return {
            "timeouts": int(_PREPARE_PIPELINE_WATCHDOG.get("timeouts", 0)),
            "stages": list(_PREPARE_PIPELINE_WATCHDOG.get("stages", ())),
            "shared_timeout": _PREPARE_PIPELINE_WATCHDOG.get("shared_timeout"),
            "extensions": list(_PREPARE_PIPELINE_WATCHDOG.get("extensions", ())),
            "component_windows": _PREPARE_PIPELINE_WATCHDOG.get("component_windows"),
        }
    except Exception:
        return {}


def collect_timeout_telemetry() -> Mapping[str, object]:
    """Public wrapper used by other modules to consume timeout telemetry."""

    return _collect_timeout_telemetry()


def _categorize_stage(entry: Mapping[str, object]) -> str | None:
    label = str(entry.get("label", "")).lower()
    if "vector" in label:
        return "vectorizers"
    if "retriev" in label:
        return "retrievers"
    if "db" in label or "index" in label:
        return "db_indexes"
    if "orchestrator" in label:
        return "orchestrator_state"
    return None


def _summarize_stage_telemetry(telemetry: Mapping[str, object]) -> Mapping[str, object]:
    stages = telemetry.get("stages") or []
    longest_stage = max((float(entry.get("elapsed", 0.0)) for entry in stages), default=0.0)
    vector_longest = max(
        (
            float(entry.get("elapsed", 0.0))
            for entry in stages
            if entry.get("timeout") and entry.get("vector_heavy")
        ),
        default=0.0,
    )
    vector_labels = [
        str(entry.get("label", ""))
        for entry in stages
        if entry.get("vector_heavy") and float(entry.get("elapsed", 0.0) or 0.0) > 0
    ]

    return {
        "longest_stage": longest_stage,
        "vector_longest": vector_longest,
        "vector_labels": vector_labels,
        "vector_stage_count": len(vector_labels),
    }


def _summarize_component_overruns(telemetry: Mapping[str, object]) -> dict[str, dict[str, float | int]]:
    shared = telemetry.get("shared_timeout") or {}
    timeline = shared.get("timeline") or []
    component_overruns: dict[str, dict[str, float | int]] = {}
    for entry in timeline:
        component = _categorize_stage(entry) or str(entry.get("label"))
        if not component:
            continue
        elapsed = float(entry.get("elapsed", 0.0) or 0.0)
        effective = entry.get("effective")
        if effective is None:
            continue
        try:
            effective_float = float(effective)
        except (TypeError, ValueError):
            continue
        overrun = elapsed > effective_float * _OVERRUN_TOLERANCE and elapsed > 0
        if not overrun:
            continue
        component_state = component_overruns.setdefault(
            component,
            {"overruns": 0, "max_elapsed": 0.0, "expected_floor": 0.0},
        )
        component_state["overruns"] = int(component_state.get("overruns", 0)) + 1
        component_state["max_elapsed"] = max(
            float(component_state.get("max_elapsed", 0.0) or 0.0), elapsed
        )
        component_state["expected_floor"] = max(
            float(component_state.get("expected_floor", 0.0) or 0.0), effective_float
        )

    extension_records = telemetry.get("extensions") or []
    for entry in extension_records:
        gate = _categorize_stage(entry) or str(entry.get("gate"))
        if not gate:
            continue
        extension_budget = _parse_float(str(entry.get("extension_budget")))
        if extension_budget is None:
            continue
        component_state = component_overruns.setdefault(
            gate, {"overruns": 0, "max_elapsed": 0.0, "expected_floor": 0.0}
        )
        component_state["expected_floor"] = max(
            float(component_state.get("expected_floor", 0.0) or 0.0), extension_budget
        )
        component_state["overruns"] = max(int(component_state.get("overruns", 0)), 1)

    return component_overruns


def _maybe_escalate_timeout_floors(
    minimums: MutableMapping[str, float],
    component_floors: MutableMapping[str, float],
    *,
    telemetry: Mapping[str, object],
    logger: logging.Logger,
) -> Mapping[str, object] | None:
    timeouts = int(telemetry.get("timeouts", 0) or 0)
    if timeouts <= 0:
        return None

    stages = telemetry.get("stages") or []
    stage_summary = _summarize_stage_telemetry(telemetry)
    load_scale = _host_load_scale()
    longest_stage = float(stage_summary.get("longest_stage", 0.0) or 0.0)
    vector_longest = float(stage_summary.get("vector_longest", 0.0) or 0.0)

    general_floor = minimums.get(
        "MENACE_BOOTSTRAP_WAIT_SECS", _BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_WAIT_SECS"]
    )
    vector_floor = minimums.get(
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", _BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"]
    )

    suggested_general = max(
        general_floor,
        general_floor * load_scale,
        longest_stage + 60.0 if longest_stage else general_floor + 60.0,
    )
    suggested_vector = max(
        vector_floor,
        vector_floor * load_scale,
        vector_longest + 90.0 if vector_longest else suggested_general,
    )

    component_updates: dict[str, float] = {}
    for entry in stages:
        component = _categorize_stage(entry)
        if component is None:
            continue
        elapsed = float(entry.get("elapsed", 0.0) or 0.0)
        baseline = component_floors.get(
            component, _COMPONENT_TIMEOUT_MINIMUMS.get(component, 0.0)
        )
        adjusted_elapsed = elapsed * load_scale if elapsed else 0.0
        adjusted_baseline = baseline * load_scale
        suggested = max(baseline, adjusted_baseline, adjusted_elapsed + 45.0) if elapsed else max(baseline, adjusted_baseline)
        if suggested > component_floors.get(component, baseline):
            component_floors[component] = suggested
            component_updates[component] = suggested

    escalated = False
    if suggested_general > general_floor:
        minimums["MENACE_BOOTSTRAP_WAIT_SECS"] = suggested_general
        minimums["BOOTSTRAP_STEP_TIMEOUT"] = max(
            minimums.get("BOOTSTRAP_STEP_TIMEOUT", suggested_general), suggested_general
        )
        escalated = True

    if suggested_vector > vector_floor:
        minimums["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"] = suggested_vector
        minimums["BOOTSTRAP_VECTOR_STEP_TIMEOUT"] = max(
            minimums.get("BOOTSTRAP_VECTOR_STEP_TIMEOUT", suggested_vector), suggested_vector
        )
        escalated = True

    if not escalated and not component_updates:
        return None

    details = {
        "timeouts": timeouts,
        "longest_stage": longest_stage,
        "vector_longest": vector_longest,
        "suggested_general": suggested_general,
        "suggested_vector": suggested_vector,
        "component_updates": component_updates,
        "load_scale": load_scale,
        "stage_summary": stage_summary,
    }

    logger.info(
        "adaptive prepare_pipeline timeout escalation applied",
        extra={"telemetry": details, "state_path": str(_TIMEOUT_STATE_PATH)},
    )

    return details


def _merge_consumption_overruns(
    *,
    host_state: MutableMapping[str, object],
    component_floors: MutableMapping[str, float],
    overruns: Mapping[str, Mapping[str, float | int]],
) -> bool:
    """Persist coordinator overruns and bump component floors when needed."""

    if not overruns:
        return False

    component_overruns = host_state.setdefault("component_overruns", {})
    state_changed = False

    for component, meta in overruns.items():
        existing = component_overruns.get(component, {}) if isinstance(component_overruns, dict) else {}
        streak = int(existing.get("overruns", 0) or 0) + int(meta.get("overruns", 0) or 0)
        max_elapsed = max(
            float(existing.get("max_elapsed", 0.0) or 0.0),
            float(meta.get("max_elapsed", 0.0) or 0.0),
        )
        expected_floor = max(
            float(existing.get("expected_floor", 0.0) or 0.0),
            float(meta.get("expected_floor", 0.0) or 0.0),
        )
        suggested_floor = max(max_elapsed + 30.0, expected_floor * 1.1) if max_elapsed else expected_floor
        component_overruns[component] = {
            "overruns": streak,
            "max_elapsed": max_elapsed,
            "expected_floor": expected_floor,
            "suggested_floor": suggested_floor,
            "last_seen": time.time(),
        }
        if streak >= _OVERRUN_STREAK_THRESHOLD and component in _COMPONENT_ENV_MAPPING:
            baseline = _COMPONENT_TIMEOUT_MINIMUMS.get(component, 0.0)
            previous_floor = component_floors.get(component, baseline)
            component_floors[component] = max(previous_floor, suggested_floor, baseline)
        state_changed = True

    host_state["component_overruns"] = component_overruns
    return state_changed


def _apply_success_decay(
    *,
    host_state: MutableMapping[str, object],
    component_floors: MutableMapping[str, float],
) -> bool:
    success_streak = int(host_state.get("success_streak", 0) or 0)
    if success_streak <= 0:
        return False

    state_changed = False
    for component, baseline in _COMPONENT_TIMEOUT_MINIMUMS.items():
        current = component_floors.get(component, baseline)
        decayed = max(baseline, current * (_DECAY_RATIO**min(success_streak, 3)))
        if decayed < current:
            component_floors[component] = decayed
            state_changed = True

    host_state["success_streak"] = max(success_streak - 1, 0)
    return state_changed


def _host_load_scale(load_average: float | None = None) -> float:
    try:
        load = load_average if load_average is not None else os.getloadavg()[0]
    except OSError:
        return 1.0
    cpu_count = os.cpu_count() or 1
    normalized = max(0.0, load / float(cpu_count))
    return 1.0 + min(normalized, 1.5)


def _soft_budget_from_timeout(
    timeout: float | None,
    *,
    telemetry: Mapping[str, object],
    load_scale: float,
    grace_floor: float = 30.0,
    grace_ratio: float = 0.25,
    stage_bias: float = 0.15,
) -> Dict[str, float | None]:
    stage_summary = _summarize_stage_telemetry(telemetry)
    longest_stage = float(stage_summary.get("longest_stage", 0.0) or 0.0)
    vector_longest = float(stage_summary.get("vector_longest", 0.0) or 0.0)

    if timeout is None:
        return {
            "budget": None,
            "grace": None,
            "limit": None,
            "scale": load_scale,
            "longest_stage": longest_stage,
            "vector_longest": vector_longest,
        }

    scaled_budget = timeout * load_scale
    base_grace = scaled_budget * grace_ratio
    stage_grace = longest_stage * stage_bias if longest_stage else 0.0
    vector_grace = vector_longest * (stage_bias + 0.1) if vector_longest else 0.0
    grace = max(grace_floor, base_grace, stage_grace, vector_grace)

    return {
        "budget": scaled_budget,
        "grace": grace,
        "limit": scaled_budget + grace,
        "scale": load_scale,
        "longest_stage": longest_stage,
        "vector_longest": vector_longest,
    }


def derive_phase_soft_budgets(
    phase_budgets: Mapping[str, float | None],
    *,
    telemetry: Mapping[str, object] | None = None,
    load_average: float | None = None,
) -> Dict[str, Dict[str, float | None]]:
    """Return soft budgets with grace windows and host-load scaling."""

    telemetry = telemetry or _collect_timeout_telemetry()
    scale = _host_load_scale(load_average)
    soft_budgets: Dict[str, Dict[str, float | None]] = {}
    for phase, timeout in phase_budgets.items():
        soft_budgets[phase] = _soft_budget_from_timeout(
            timeout, telemetry=telemetry, load_scale=scale
        )
    soft_budgets["meta"] = {
        "load_scale": scale,
        "telemetry": _summarize_stage_telemetry(telemetry),
    }
    return soft_budgets


def enforce_bootstrap_timeout_policy(
    *,
    logger: logging.Logger | None = None,
    prompt_override: Callable[[str, float, float], bool] | None = None,
) -> Dict[str, Dict[str, float | bool | None]]:
    """Clamp bootstrap timeouts to recommended floors when needed."""

    active_logger = logger or LOGGER
    minimums: dict[str, float] = load_escalated_timeout_floors()
    component_floors: dict[str, float] = load_component_timeout_floors()
    telemetry = _collect_timeout_telemetry()
    component_budgets = compute_prepare_pipeline_component_budgets(
        component_floors=component_floors, telemetry=telemetry
    )
    overrun_meta = _summarize_component_overruns(telemetry)
    success_run = (telemetry.get("timeouts") or 0) == 0 and not overrun_meta
    state = _load_timeout_state()
    host_key = _state_host_key()
    host_state = state.get(host_key, {}) if isinstance(state, dict) else {}
    state_changed = False

    if success_run:
        host_state["success_streak"] = int(host_state.get("success_streak", 0) or 0) + 1
    else:
        if host_state.get("success_streak"):
            state_changed = True
        host_state["success_streak"] = 0

    state_changed |= _merge_consumption_overruns(
        host_state=host_state, component_floors=component_floors, overruns=overrun_meta
    )
    state_changed |= _apply_success_decay(host_state=host_state, component_floors=component_floors)
    escalation_meta = _maybe_escalate_timeout_floors(
        minimums, component_floors, telemetry=telemetry, logger=active_logger
    )
    if escalation_meta:
        state_changed = True
        host_state.update({k: minimums.get(k, v) for k, v in _BOOTSTRAP_TIMEOUT_MINIMUMS.items()})

    if escalation_meta or state_changed:
        host_state["component_floors"] = dict(component_floors)
        host_state["updated_at"] = time.time()
        state = state if isinstance(state, dict) else {}
        state[host_key] = host_state
        _save_timeout_state(state)
        active_logger.info(
            "bootstrap timeout floors updated",
            extra={
                "timeouts": telemetry.get("timeouts"),
                "host": host_key,
                "escalation": escalation_meta,
                "floors": minimums,
                "component_floors": component_floors,
                "component_overruns": overrun_meta,
                "state_file": str(_TIMEOUT_STATE_PATH),
                "success_run": success_run,
            },
        )
    allow_unsafe = _truthy_env(os.getenv(_OVERRIDE_ENV))
    results: Dict[str, Dict[str, float | bool | None]] = {}

    for env_var, minimum in minimums.items():
        raw_value = os.getenv(env_var)
        requested_value = _parse_float(raw_value)
        effective_value = requested_value if requested_value is not None else minimum
        clamped = False
        override_granted = False

        if requested_value is None and raw_value is not None:
            active_logger.warning(
                "%s is not a valid float (%r); forcing recommended minimum %.1fs",
                env_var,
                raw_value,
                minimum,
                extra={"env_var": env_var, "raw_value": raw_value, "minimum": minimum},
            )
            clamped = True
            effective_value = minimum
            os.environ[env_var] = str(minimum)
        elif requested_value is not None and requested_value < minimum:
            if allow_unsafe:
                active_logger.warning(
                    "%s below safe floor (requested=%.1fs, minimum=%.1fs) but %s=1 allows override",
                    env_var,
                    requested_value,
                    minimum,
                    _OVERRIDE_ENV,
                )
                override_granted = True
                effective_value = requested_value
            elif prompt_override is not None:
                override_granted = prompt_override(env_var, requested_value, minimum)
                if override_granted:
                    active_logger.warning(
                        "%s below safe floor (requested=%.1fs, minimum=%.1fs); proceeding after explicit user override",
                        env_var,
                        requested_value,
                        minimum,
                    )
                    effective_value = requested_value
                else:
                    clamped = True
                    effective_value = minimum
            else:
                clamped = True
                effective_value = minimum

            if clamped:
                active_logger.warning(
                    "%s below safe floor (requested=%.1fs); clamping to %.1fs",
                    env_var,
                    requested_value,
                    minimum,
                    extra={
                        "requested_timeout": requested_value,
                        "timeout_floor": minimum,
                        "effective_timeout": effective_value,
                    },
                )
                os.environ[env_var] = str(effective_value)

        results[env_var] = {
            "requested": requested_value,
            "effective": effective_value,
            "minimum": minimum,
            "clamped": clamped,
            "override_granted": override_granted,
        }

    results[_OVERRIDE_ENV] = {"requested": float(allow_unsafe), "effective": float(allow_unsafe)}
    results["component_floors"] = component_floors
    results["component_budgets"] = component_budgets
    soft_budget_inputs = {k: v.get("effective") for k, v in results.items() if k in minimums}
    results["soft_budgets"] = derive_phase_soft_budgets(
        soft_budget_inputs, telemetry=telemetry
    )
    return results


def render_prepare_pipeline_timeout_hints(vector_heavy: bool | None = None) -> list[str]:
    """Return standard remediation hints for ``prepare_pipeline_for_bootstrap`` timeouts."""

    minimums = load_escalated_timeout_floors()
    component_floors = load_component_timeout_floors()
    adaptive_context = get_adaptive_timeout_context()
    adaptive_applied = any(
        minimums.get(key, 0.0) > _BOOTSTRAP_TIMEOUT_MINIMUMS.get(key, 0.0)
        for key in _BOOTSTRAP_TIMEOUT_MINIMUMS
    ) or any(
        component_floors.get(key, 0.0) > _COMPONENT_TIMEOUT_MINIMUMS.get(key, 0.0)
        for key in _COMPONENT_TIMEOUT_MINIMUMS
    )
    vector_wait = _parse_float(os.getenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS")) or minimums[
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"
    ]
    vector_step = _parse_float(os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT")) or minimums[
        "BOOTSTRAP_VECTOR_STEP_TIMEOUT"
    ]

    hints = [
        "Increase MENACE_BOOTSTRAP_WAIT_SECS=240 or BOOTSTRAP_STEP_TIMEOUT=240 for slower bootstrap hosts.",
        (
            "Vector-heavy pipelines: set MENACE_BOOTSTRAP_VECTOR_WAIT_SECS="
            f"{int(vector_wait)} or BOOTSTRAP_VECTOR_STEP_TIMEOUT={int(vector_step)} "
            "to bypass the legacy 30s cap and give vector services time to warm up."
        ),
        "Stagger concurrent bootstraps or shrink watched directories to reduce contention during pipeline and vector service startup.",
    ]

    recent_peers = _recent_peer_activity()
    active_peers = [peer for peer in recent_peers if peer.get("pid") not in (None, os.getpid())]
    if len(active_peers) >= 1:
        peer_hosts = sorted({str(peer.get("host")) for peer in active_peers})
        hints.append(
            (
                "Shared timeout telemetry shows recent bootstrap activity from "
                f"{', '.join(peer_hosts)}; budgets are being staggered automatically. "
                "Consider staggering new launches or honouring MENACE_BOOTSTRAP_STAGGER_SECS to avoid contention."
            )
        )
        peer_contenders = [peer for peer in active_peers if peer.get("guard_delay") or (peer.get("budget_scale") and peer.get("budget_scale") != 1.0)]
        if peer_contenders:
            hints.append(
                (
                    "Recent coordinator telemetry persisted at "
                    f"{_TIMEOUT_STATE_PATH} shows contention-driven guard delays; "
                    "timeouts have been elevated so peers inherit the scaled budgets."
                )
            )

    if adaptive_applied:
        adaptive_floors = adaptive_context.get("adaptive_floors", {}) if isinstance(adaptive_context, dict) else {}
        adaptive_reasons = "; ".join(
            f"{env} raised for {meta.get('source')} after {meta.get('overruns')} overruns"
            for env, meta in adaptive_floors.items()
        ) or "recent load and stage durations"
        hints.append(
            (
                "Adaptive timeout scaling is active for this host based on load and recent stage durations; "
                f"values are persisted at {_TIMEOUT_STATE_PATH}. Set {_OVERRIDE_ENV}=1 or remove the state file "
                "to force manual overrides."
            )
        )
        if adaptive_reasons:
            hints.append(f"Recent coordinator overruns: {adaptive_reasons}.")

    if vector_heavy:
        return list(hints)

    return hints


class SharedTimeoutCoordinator:
    """Coordinate shared timeout budgets across related bootstrap tasks.

    The coordinator exposes a lightweight API that allows modules to reserve a
    time slice from a shared budget before starting heavy work (for example
    vectorizer warm-ups, retriever hydration, database bootstrap, or
    orchestrator state loads). Reservations are serialized to prevent unrelated
    helpers from racing the same global deadline, and detailed consumption
    metadata is logged to aid debugging.
    """

    def __init__(
        self,
        total_budget: float | None,
        *,
        logger: logging.Logger | None = None,
        namespace: str = "bootstrap",
        component_floors: Mapping[str, float] | None = None,
        component_budgets: Mapping[str, float] | None = None,
        signal_hook: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.namespace = namespace
        self.logger = logger or LOGGER
        self._lock = threading.Lock()
        self._timeline: list[dict[str, float | str | None]] = []
        self.component_floors = dict(component_floors or {})
        self.component_budgets = dict(component_budgets or {})
        self.remaining_components = dict(self.component_budgets)
        self._component_windows: dict[str, dict[str, float | None]] = {}
        self._signal_hook = signal_hook

    def _reserve(
        self,
        label: str,
        requested: float | None,
        minimum: float,
        metadata: Mapping[str, object] | None,
    ) -> tuple[float | None, MutableMapping[str, object]]:
        with self._lock:
            component_floor = self.component_floors.get(label, 0.0)
            effective_floor = max(minimum, component_floor)
            effective = requested if requested is not None else effective_floor
            effective = max(effective_floor, effective)
            remaining_before = self.remaining_budget
            component_remaining = self.remaining_components.get(label)
            if component_remaining is not None:
                effective = min(effective, max(component_remaining, 0.0))
                self.remaining_components[label] = max(
                    component_remaining - effective, 0.0
                )
            if self.remaining_budget is not None:
                effective = min(effective, max(self.remaining_budget, 0.0))
                self.remaining_budget = max(self.remaining_budget - effective, 0.0)

            record: MutableMapping[str, object] = {
                "label": label,
                "requested": requested,
                "minimum": minimum,
                "component_floor": component_floor,
                "component_budget": self.component_budgets.get(label),
                "component_remaining_before": component_remaining,
                "component_remaining_after": self.remaining_components.get(label),
                "effective": effective,
                "remaining_before": remaining_before,
                "remaining_after": self.remaining_budget,
                "namespace": self.namespace,
            }
            if metadata:
                record.update({f"meta.{k}": v for k, v in metadata.items()})

        self.logger.info(
            "shared timeout budget reserved",
            extra={"shared_timeout": dict(record)},
        )
        return effective, record

    def start_component_timers(
        self, budgets: Mapping[str, float | None], *, minimum: float = 0.0
    ) -> Mapping[str, Mapping[str, float | None]]:
        """Allocate independent timers for each component gate."""

        windows: dict[str, dict[str, float | None]] = {}
        for label, requested in budgets.items():
            effective, record = self._reserve(
                label,
                requested,
                max(minimum, self.component_floors.get(label, minimum)),
                {"component_timer": True},
            )
            started = time.monotonic()
            window = {
                "budget": effective,
                "deadline": (started + effective) if effective is not None else None,
                "started": started,
                "remaining": effective,
            }
            windows[label] = window
            with self._lock:
                self._component_windows[label] = dict(window)
                self._timeline.append({**record, **window, "namespace": self.namespace})
        return windows

    @contextlib.contextmanager
    def consume(
        self,
        label: str,
        *,
        requested: float | None,
        minimum: float = 0.0,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[tuple[float | None, Mapping[str, object]]]:
        """Reserve a time slice and log consumption when complete."""

        start = time.monotonic()
        effective, record = self._reserve(label, requested, minimum, metadata)
        try:
            yield effective, record
        finally:
            elapsed = time.monotonic() - start
            record = dict(record)
            record.update({"elapsed": elapsed, "namespace": self.namespace})
            with self._lock:
                self._timeline.append(record)
            self.logger.info(
                "shared timeout budget consumed",
                extra={"shared_timeout": record},
            )

    def reserve_phase(
        self,
        label: str,
        *,
        requested: float | None,
        minimum: float = 0.0,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[float | None, Mapping[str, object]]:
        """Reserve a slice for a named phase without managing a context."""

        effective, record = self._reserve(label, requested, minimum, metadata)
        with self._lock:
            self._timeline.append(
                {
                    **record,
                    "elapsed": 0.0,
                    "namespace": self.namespace,
                    "phase": label,
                }
            )
        return effective, record

    def record_progress(
        self,
        label: str,
        *,
        elapsed: float,
        remaining: float | None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Log incremental phase consumption on the shared timeline."""

        record: MutableMapping[str, object] = {
            "label": label,
            "elapsed": elapsed,
            "remaining_budget": remaining,
            "namespace": self.namespace,
            "host_load": _host_load_average(),
            "ts": time.time(),
        }
        if metadata:
            record.update({f"meta.{k}": v for k, v in metadata.items()})
        with self._lock:
            window = self._component_windows.get(label)
            if window is not None:
                window = dict(window)
                now = time.monotonic()
                window.update({
                    "remaining": remaining,
                    "last_progress": now,
                })
                if remaining is not None:
                    window["deadline"] = now + remaining
                self._component_windows[label] = window
            self._timeline.append(dict(record))
        self.logger.info(
            "shared timeout budget progress",
            extra={"shared_timeout": record},
        )
        if self._signal_hook:
            try:
                self._signal_hook(dict(record))
            except Exception:
                self.logger.debug("progress signal hook failed", exc_info=True)

    def snapshot(self) -> Mapping[str, object]:
        """Return a shallow snapshot of coordinator state."""

        with self._lock:
            return {
                "namespace": self.namespace,
                "total_budget": self.total_budget,
                "remaining_budget": self.remaining_budget,
                "timeline": list(self._timeline),
                "component_floors": dict(self.component_floors),
                "component_budgets": dict(self.component_budgets),
                "component_remaining": dict(self.remaining_components),
                "component_windows": dict(self._component_windows),
            }
