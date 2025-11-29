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
from typing import Callable, Dict, Iterator, Mapping, MutableMapping, Any, Iterable

_SHARED_EVENT_BUS = None

LOGGER = logging.getLogger(__name__)
_ADAPTIVE_TIMEOUT_CONTEXT: dict[str, object] = {}
_BOOTSTRAP_COMPONENT_HINTS_ENV = "MENACE_BOOTSTRAP_COMPONENT_HINTS"
_BOOTSTRAP_DB_INDEX_BYTES_ENV = "MENACE_BOOTSTRAP_DB_INDEX_BYTES"
_BOOTSTRAP_COMPLEXITY_SCALE_ENV = "MENACE_BOOTSTRAP_COMPONENT_COMPLEXITY_SCALE"
BOOTSTRAP_COMPONENT_HINTS_ENV = _BOOTSTRAP_COMPONENT_HINTS_ENV
BOOTSTRAP_DB_INDEX_BYTES_ENV = _BOOTSTRAP_DB_INDEX_BYTES_ENV
BOOTSTRAP_COMPLEXITY_SCALE_ENV = _BOOTSTRAP_COMPLEXITY_SCALE_ENV

_BOOTSTRAP_TIMEOUT_MINIMUMS: dict[str, float] = {
    "MENACE_BOOTSTRAP_WAIT_SECS": 360.0,
    "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS": 540.0,
    "BOOTSTRAP_STEP_TIMEOUT": 360.0,
    "BOOTSTRAP_VECTOR_STEP_TIMEOUT": 540.0,
    "PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS": 720.0,
    "PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS": 480.0,
    "PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS": 480.0,
    "PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS": 420.0,
    "PREPARE_PIPELINE_CONFIG_BUDGET_SECS": 420.0,
}
_COMPONENT_TIMEOUT_MINIMUMS: dict[str, float] = {
    "vectorizers": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS"],
    "retrievers": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS"],
    "db_indexes": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS"],
    "orchestrator_state": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS"],
    "pipeline_config": _BOOTSTRAP_TIMEOUT_MINIMUMS["PREPARE_PIPELINE_CONFIG_BUDGET_SECS"],
}
_DEFERRED_COMPONENT_TIMEOUT_MINIMUMS: dict[str, float] = {
    "background_loops": _BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_WAIT_SECS"] * 1.5,
}
DEFERRED_COMPONENTS = frozenset(_DEFERRED_COMPONENT_TIMEOUT_MINIMUMS)
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
    "pipeline_config": "PREPARE_PIPELINE_CONFIG_BUDGET_SECS",
}
_DEFERRED_COMPONENT_ENV_MAPPING = {
    "background_loops": "MENACE_BOOTSTRAP_WAIT_SECS",
}
_ALL_COMPONENT_ENV_MAPPING = {**_COMPONENT_ENV_MAPPING, **_DEFERRED_COMPONENT_ENV_MAPPING}
_OVERRUN_TOLERANCE = 1.05
_OVERRUN_STREAK_THRESHOLD = 2
_DECAY_RATIO = 0.9
_DEFAULT_HEARTBEAT_MAX_AGE = 120.0
_DEFAULT_LOAD_THRESHOLD = 1.35
_DEFAULT_GUARD_MAX_DELAY = 90.0
_DEFAULT_GUARD_INTERVAL = 5.0
_COMPONENT_FLOOR_MAX_SCALE_ENV = "MENACE_BOOTSTRAP_COMPONENT_FLOOR_MAX_SCALE"
_COMPONENT_STALENESS_PAD_ENV = "MENACE_BOOTSTRAP_HEARTBEAT_STALENESS_PAD"
_DEFAULT_COMPONENT_FLOOR_MAX_SCALE = 3.5
_DEFAULT_STALENESS_PAD = 0.35
_BOOTSTRAP_HEARTBEAT_ENV = "MENACE_BOOTSTRAP_WATCHDOG_PATH"
_BOOTSTRAP_HEARTBEAT_MAX_AGE_ENV = "MENACE_BOOTSTRAP_HEARTBEAT_MAX_AGE"
_BOOTSTRAP_LOAD_THRESHOLD_ENV = "MENACE_BOOTSTRAP_LOAD_THRESHOLD"
_BOOTSTRAP_HEARTBEAT_PATH = Path(
    os.getenv(_BOOTSTRAP_HEARTBEAT_ENV, "/tmp/menace_bootstrap_watchdog.json")
)
_BACKGROUND_UNLIMITED_ENV = "MENACE_BOOTSTRAP_BACKGROUND_UNLIMITED"
_ESSENTIAL_PHASES = {
    "vectorizers",
    "retrievers",
    "db_indexes",
    "orchestrator_state",
    "pipeline_config",
}
_OPTIONAL_PHASES = {"background_loops"}


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


def _record_bootstrap_guard(
    delay: float,
    budget_scale: float,
    *,
    source: str,
    host_load: float | None = None,
) -> None:
    """Persist guard telemetry for downstream budget scaling."""

    global _LAST_BOOTSTRAP_GUARD
    _LAST_BOOTSTRAP_GUARD = {
        "delay": float(delay),
        "budget_scale": float(budget_scale),
        "source": source,
        "ts": time.time(),
        "host_load": host_load,
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
    queue_capacity: int | None = None,
    block_when_saturated: bool | None = None,
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
    queue_capacity = queue_capacity if queue_capacity is not None else int(
        os.getenv("MENACE_BOOTSTRAP_GUARD_QUEUE_CAPACITY", "0") or 0
    )
    block_when_saturated = (
        block_when_saturated
        if block_when_saturated is not None
        else _truthy_env(os.getenv("MENACE_BOOTSTRAP_GUARD_QUEUE_BLOCK"))
    )

    deadline = time.monotonic() + target_delay
    slept = 0.0
    budget_scale = 1.0
    queue_depth = 0

    normalized_load = _host_load_average()
    while time.monotonic() < deadline:
        heartbeat = read_bootstrap_heartbeat()
        peer_active = False
        if heartbeat:
            peer_active = int(heartbeat.get("pid", -1)) != int(ignore_pid)
            normalized_load = heartbeat.get("host_load", normalized_load)
        overloaded = normalized_load is not None and normalized_load > threshold

        recent_peers = _recent_peer_activity(max_age=target_delay * 2)
        peer_queue = []
        for peer in recent_peers:
            try:
                peer_pid = int(peer.get("pid", -1))
            except (TypeError, ValueError):
                peer_pid = -1
            if peer_pid != int(ignore_pid):
                peer_queue.append(peer)
        overload_units = 0
        if normalized_load is not None and threshold:
            overload_units = max(int(round(normalized_load / threshold)), 0)
        queue_depth = max(len(peer_queue) + (1 if peer_active else 0), overload_units)
        saturated = bool(queue_capacity and queue_depth >= queue_capacity)

        if block_when_saturated and saturated:
            raise TimeoutError("bootstrap guard queue saturated")

        if not peer_active and not overloaded and queue_depth == 0:
            break

        remaining_budget = max(deadline - time.monotonic(), 0.0)
        if remaining_budget <= 0:
            break

        sleep_for = min(poll_interval * max(queue_depth, 1), remaining_budget)
        slept += sleep_for
        if normalized_load and threshold:
            budget_scale = max(
                budget_scale,
                min(normalized_load / threshold, 2.5),
                1.0 + min(queue_depth, 3) * 0.15,
            )

        logger.info(
            "delaying bootstrap to avoid contention",
            extra={
                "event": "bootstrap-guard-delay",
                "sleep_for": round(sleep_for, 2),
                "peer_active": peer_active,
                "queue_depth": queue_depth,
                "normalized_load": normalized_load,
                "threshold": threshold,
                "budget_scale": round(budget_scale, 3),
                "saturated": saturated,
            },
        )
        time.sleep(sleep_for)

    guard_context = {
        "delay": slept,
        "budget_scale": budget_scale,
        "queue_depth": queue_depth,
        "host_load": normalized_load,
    }
    if queue_capacity:
        guard_context["queue_capacity"] = queue_capacity
    _record_bootstrap_guard(
        slept, budget_scale, source="bootstrap_guard", host_load=normalized_load
    )
    _record_adaptive_context({"guard_context": guard_context})
    return slept, budget_scale


def build_progress_signal_hook(
    *, namespace: str, run_id: str | None = None
) -> Callable[[Mapping[str, object]], None]:
    """Return a hook that broadcasts SharedTimeoutCoordinator progress."""

    supervisor = BootstrapHeartbeatSupervisor(namespace=namespace)

    def _signal(record: Mapping[str, object]) -> None:
        enriched: dict[str, Any] = {
            "namespace": namespace,
            "run_id": run_id or f"{namespace}-{os.getpid()}",
            **record,
        }
        readiness = supervisor.update(enriched)
        if readiness:
            enriched["bootstrap_supervisor"] = readiness
        emit_bootstrap_heartbeat(enriched)

    return _signal


class BootstrapHeartbeatSupervisor:
    """Track per-phase bootstrap heartbeats and mark online readiness."""

    def __init__(self, *, namespace: str) -> None:
        self.namespace = namespace
        self.components: dict[str, str] = {}
        self._last_snapshot: dict[str, object] = {}

    @staticmethod
    def _normalize_component(label: str | None) -> str | None:
        if not label:
            return None
        label = str(label)
        aliases = {
            "vectorizer": "vectorizers",
            "vector": "vectorizers",
            "retriever": "retrievers",
            "db_index": "db_indexes",
            "db_index_load": "db_indexes",
            "orchestrator": "orchestrator_state",
            "config": "pipeline_config",
            "background": "background_loops",
        }
        return aliases.get(label, label)

    def _component_state(self, heartbeat: Mapping[str, object]) -> tuple[str | None, str | None]:
        component = self._normalize_component(
            heartbeat.get("component")
            or heartbeat.get("label")
            or heartbeat.get("phase")
        )
        state = heartbeat.get("component_state") or heartbeat.get("state")
        if component is None:
            return None, None

        if state:
            try:
                return component, str(state)
            except Exception:
                return component, None

        remaining = heartbeat.get("remaining")
        progressing = _heartbeat_progressing(heartbeat)
        if progressing and remaining is None:
            state = "ready"
        elif progressing:
            state = "progressing"
        elif remaining is None:
            state = "pending"
        else:
            state = "warming"
        return component, state

    def _online(self) -> tuple[bool, set[str]]:
        lagging: set[str] = set()
        for component in _ESSENTIAL_PHASES:
            state = self.components.get(component, "pending")
            if state in {"ready", "progressing", "warming"}:
                continue
            lagging.add(component)
        return len(lagging) == 0, lagging

    def update(self, heartbeat: Mapping[str, object]) -> Mapping[str, object]:
        component, state = self._component_state(heartbeat)
        if component is None:
            return self._last_snapshot

        if state:
            self.components[component] = state

        optional = {name: self.components.get(name, "pending") for name in _OPTIONAL_PHASES}
        online, lagging = self._online()
        snapshot = {
            "namespace": self.namespace,
            "components": dict(self.components),
            "optional": optional,
            "online": online,
            "lagging": sorted(lagging),
            "ts": time.time(),
        }
        self._last_snapshot = snapshot
        return snapshot


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


def _heartbeat_progressing(heartbeat: Mapping[str, object]) -> bool:
    for key in (
        "meta.heartbeat.progressing",
        "heartbeat.progressing",
        "progressing",
    ):
        value = heartbeat.get(key)
        if isinstance(value, bool):
            if value:
                return True
        elif isinstance(value, (int, float)):
            if float(value) > 0:
                return True
    return False


def _derive_rolling_global_window(
    *,
    base_window: float | None,
    component_budget_total: float,
    host_telemetry: Mapping[str, object] | None,
) -> tuple[float | None, Mapping[str, object]]:
    """Extend the bootstrap window when live heartbeats show progress."""

    heartbeat = host_telemetry if isinstance(host_telemetry, Mapping) else {}
    window = base_window if base_window is not None else (component_budget_total or None)
    extension_meta: dict[str, object] = {}
    if not heartbeat:
        return window, extension_meta

    heartbeat_window = _parse_float(str(heartbeat.get("global_window")))
    heartbeat_remaining = _parse_float(str(heartbeat.get("remaining_budget")))
    heartbeat_ts = _parse_float(str(heartbeat.get("ts")))
    progressing = _heartbeat_progressing(heartbeat)

    if heartbeat_window is not None:
        window = max(window or 0.0, heartbeat_window)

    projected_remaining = None
    if heartbeat_remaining is not None:
        staleness = max(time.time() - heartbeat_ts, 0.0) if heartbeat_ts else 0.0
        projected_remaining = max(heartbeat_remaining - staleness, 0.0)
        guard_padding = max(projected_remaining * 0.25, 15.0 if progressing else 5.0)
        baseline = window if window is not None else component_budget_total
        candidate = max(baseline or 0.0, component_budget_total) + projected_remaining + guard_padding
        if window is None or candidate > window:
            window = candidate
            extension_meta = {
                "reason": "heartbeat_progress",
                "baseline": baseline,
                "component_budget_total": component_budget_total,
                "heartbeat_remaining": heartbeat_remaining,
                "projected_remaining": projected_remaining,
                "heartbeat_window": heartbeat_window,
                "progressing": progressing,
                "staleness": staleness,
                "guard_padding": guard_padding,
                "extended_window": window,
                "extension_seconds": window - (baseline or 0.0),
            }

    return window, extension_meta


def _derive_concurrency_global_window(
    *,
    base_window: float | None,
    component_budget_total: float,
    component_budgets: Mapping[str, float],
    scheduled_component_budgets: Mapping[str, float] | None,
    deferred_components: set[str],
    guard_context: Mapping[str, object] | None,
    host_telemetry: Mapping[str, object] | None,
) -> tuple[float | None, Mapping[str, object]]:
    """Inflate the global window when several component slices overlap."""

    if not component_budget_total and not base_window:
        return base_window, {}

    scheduled = {
        key: value
        for key, value in (scheduled_component_budgets or {}).items()
        if value is not None
    }
    budget_candidates = scheduled if scheduled else component_budgets
    active_budgets: dict[str, float] = {}
    for key, value in budget_candidates.items():
        if key in deferred_components:
            continue
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            continue
        if coerced <= 0:
            continue
        active_budgets[key] = coerced

    concurrency = len(active_budgets)
    if concurrency <= 1:
        return base_window, {}

    guard_context = guard_context or {}
    host_telemetry = host_telemetry or {}
    threshold = _parse_float(os.getenv(_BOOTSTRAP_LOAD_THRESHOLD_ENV)) or _DEFAULT_LOAD_THRESHOLD
    host_load = _parse_float(str(host_telemetry.get("host_load")))
    if host_load is None:
        host_load = _parse_float(str(guard_context.get("host_load")))
    if host_load is None:
        host_load = _host_load_average()

    load_scale = 1.0
    if host_load is not None and threshold:
        load_scale += max(host_load / threshold - 1.0, 0.0) * 0.35

    queue_depth = 0
    try:
        queue_depth = int(host_telemetry.get("queue_depth") or guard_context.get("queue_depth") or 0)
        queue_depth = max(queue_depth, 0)
    except (TypeError, ValueError):
        queue_depth = 0

    backlog_scale = 1.0 + min(queue_depth, 5) * 0.05
    concurrency_scale = 1.0 + min(concurrency - 1, 4) * 0.12

    extension_scale = max(concurrency_scale * load_scale * backlog_scale, 1.0)
    base = base_window if base_window is not None else component_budget_total
    if not base:
        return base_window, {}

    extended_window = base * extension_scale
    if extended_window <= (base_window or base):
        return base_window, {}

    extension_meta = {
        "concurrency": concurrency,
        "active_components": sorted(active_budgets),
        "base_window": base,
        "extended_window": extended_window,
        "extension_seconds": extended_window - base,
        "concurrency_scale": concurrency_scale,
        "load_scale": load_scale,
        "backlog_scale": backlog_scale,
        "queue_depth": queue_depth,
        "host_load": host_load,
    }
    return extended_window, extension_meta


def _state_host_key() -> str:
    return socket.gethostname() or "unknown-host"


def _record_adaptive_context(context: Mapping[str, object]) -> None:
    _ADAPTIVE_TIMEOUT_CONTEXT.clear()
    _ADAPTIVE_TIMEOUT_CONTEXT.update(context)


def _derive_component_floor_scale(
    *,
    guard_context: Mapping[str, object] | None = None,
    host_state: Mapping[str, object] | None = None,
    host_telemetry: Mapping[str, object] | None = None,
) -> tuple[float, Mapping[str, object]]:
    """Return a scaling factor for component floors based on guard telemetry."""

    threshold = _parse_float(os.getenv(_BOOTSTRAP_LOAD_THRESHOLD_ENV)) or _DEFAULT_LOAD_THRESHOLD
    max_scale = _parse_float(os.getenv(_COMPONENT_FLOOR_MAX_SCALE_ENV)) or _DEFAULT_COMPONENT_FLOOR_MAX_SCALE
    telemetry_load = None
    heartbeat_ts = None
    if isinstance(host_telemetry, Mapping):
        telemetry_load = _parse_float(str(host_telemetry.get("host_load")))
        heartbeat_ts = _parse_float(str(host_telemetry.get("ts")))

    guard_load = None
    if isinstance(guard_context, Mapping):
        guard_load = _parse_float(str(guard_context.get("host_load")))

    load_average = guard_load if guard_load is not None else telemetry_load
    if load_average is None:
        load_average = _host_load_average()

    load_scale = 1.0
    if load_average is not None and threshold:
        load_scale = max(1.0, min(load_average / threshold, max_scale))

    guard_scale = 1.0
    if isinstance(guard_context, Mapping):
        try:
            guard_scale = max(float(guard_context.get("budget_scale", 1.0) or 1.0), 1.0)
        except (TypeError, ValueError):
            guard_scale = 1.0

    overrun_streak = 0
    component_overruns = None
    if isinstance(host_state, Mapping):
        host_key = _state_host_key()
        host_meta = host_state.get(host_key, {}) if host_key in host_state else host_state
        if isinstance(host_meta, Mapping):
            component_overruns = host_meta.get("component_overruns")
    if isinstance(component_overruns, Mapping):
        overrun_streak = max(
            (int(meta.get("overruns", 0) or 0) for meta in component_overruns.values()), default=0
        )
    overrun_scale = 1.0 + min(overrun_streak, 6) * 0.08

    now = time.time()
    heartbeat_age = now - heartbeat_ts if heartbeat_ts is not None else None
    heartbeat_max_age = _parse_float(os.getenv(_BOOTSTRAP_HEARTBEAT_MAX_AGE_ENV)) or _DEFAULT_HEARTBEAT_MAX_AGE
    staleness_ratio = 0.0
    if heartbeat_age is None:
        staleness_ratio = 1.0
    elif heartbeat_max_age:
        staleness_ratio = max(heartbeat_age - heartbeat_max_age, 0.0) / heartbeat_max_age
    staleness_pad_cap = _parse_float(os.getenv(_COMPONENT_STALENESS_PAD_ENV)) or _DEFAULT_STALENESS_PAD
    staleness_scale = 1.0 + min(max(staleness_ratio, 0.0), 1.0) * staleness_pad_cap

    scale = max(load_scale, guard_scale, overrun_scale) * staleness_scale
    scale = min(scale, max_scale)

    context = {
        "load_scale": load_scale,
        "guard_scale": guard_scale,
        "overrun_scale": overrun_scale,
        "overrun_streak": overrun_streak,
        "staleness_scale": staleness_scale,
        "staleness_ratio": staleness_ratio,
        "heartbeat_age": heartbeat_age,
        "heartbeat_max_age": heartbeat_max_age,
        "host_load": load_average,
        "max_scale": max_scale,
    }
    return scale, context


def _derive_component_pool_scales(
    *,
    guard_context: Mapping[str, object] | None = None,
    telemetry_overruns: Mapping[str, Mapping[str, float | int]] | None = None,
    host_overruns: Mapping[str, Mapping[str, float | int]] | None = None,
    components: Iterable[str] | None = None,
) -> dict[str, float]:
    """Return per-component pool multipliers based on overruns and guard waits."""

    guard_scale = 1.0
    if isinstance(guard_context, Mapping):
        try:
            guard_scale = max(float(guard_context.get("budget_scale", 1.0) or 1.0), 1.0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            guard_scale = 1.0

    components = set(components or ())
    overruns: dict[str, Mapping[str, float | int]] = {}
    for source in (telemetry_overruns, host_overruns):
        if not isinstance(source, Mapping):
            continue
        for component, meta in source.items():
            overruns[component] = meta
            components.add(component)

    if not components:
        return {}

    pools: dict[str, float] = {component: 1.0 for component in components}
    for component in components:
        meta = overruns.get(component, {}) if isinstance(overruns, Mapping) else {}
        try:
            streak = max(int(meta.get("overruns", 0) or 0), 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            streak = 0

        overrun_scale = 1.0 + min(streak, 6) * 0.12
        guard_allocation = 1.0
        if guard_scale > 1.0:
            guard_allocation = guard_scale if streak else 1.0 + (guard_scale - 1.0) * 0.2

        pools[component] = max(overrun_scale, guard_allocation)

    return pools


def _cluster_budget_scale(
    *,
    host_state: Mapping[str, object] | None = None,
    host_telemetry: Mapping[str, object] | None = None,
    load_average: float | None = None,
) -> tuple[float, Mapping[str, object]]:
    """Return a scaling factor based on peer load and persisted overruns."""

    host_state = host_state or _load_timeout_state()
    host_telemetry = host_telemetry or read_bootstrap_heartbeat()
    threshold = _parse_float(os.getenv(_BOOTSTRAP_LOAD_THRESHOLD_ENV)) or _DEFAULT_LOAD_THRESHOLD

    telemetry_load = None
    if isinstance(host_telemetry, Mapping):
        telemetry_load = _parse_float(str(host_telemetry.get("host_load")))

    load_average = telemetry_load if telemetry_load is not None else load_average
    if load_average is None:
        load_average = _host_load_average()

    load_scale = 1.0
    if load_average is not None and threshold:
        load_scale = max(1.0, min(load_average / threshold, 2.5))

    host_key = _state_host_key()
    component_overruns: Mapping[str, object] | None = None
    if isinstance(host_state, Mapping):
        host_component_meta = host_state.get(host_key, {}) if host_key in host_state else host_state
        if isinstance(host_component_meta, Mapping):
            component_overruns = host_component_meta.get("component_overruns")

    streak = 0
    if isinstance(component_overruns, Mapping):
        streak = max(
            (int(meta.get("overruns", 0) or 0) for meta in component_overruns.values()),
            default=0,
        )

    overrun_scale = 1.0 + min(streak, 5) * 0.1
    scale = min(max(load_scale, overrun_scale), 3.0)
    context = {
        "host_load": load_average,
        "load_scale": load_scale,
        "overrun_scale": overrun_scale,
        "overrun_streak": streak,
        "threshold": threshold,
        "heartbeat": dict(host_telemetry or {}),
    }
    return scale, context


def get_adaptive_timeout_context() -> Mapping[str, object]:
    """Return the last computed adaptive timeout context."""

    return dict(_ADAPTIVE_TIMEOUT_CONTEXT)


def _parse_component_inventory(raw: Mapping[str, object] | str | None) -> Dict[str, object]:
    """Normalize component inventory hints from strings or mappings."""

    if raw is None:
        return {}

    if isinstance(raw, Mapping):
        source_items = raw.items()
    else:
        source_items = []
        for pair in str(raw).split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                source_items.append((key.strip(), value.strip()))

    inventory: Dict[str, object] = {}
    for key, value in source_items:
        normalized = str(key).strip().lower().replace("-", "_")
        if not normalized:
            continue
        try:
            inventory[normalized] = int(value) if str(value).isdigit() else float(value)
        except (TypeError, ValueError):
            inventory[normalized] = value

    return inventory


def compute_adaptive_component_minimums(
    *,
    pipeline_complexity: Mapping[str, object] | None = None,
    host_telemetry: Mapping[str, object] | None = None,
    load_average: float | None = None,
    component_hints: Mapping[str, object] | str | None = None,
) -> dict[str, float]:
    """Scale component minimums using guard, heartbeat, and hint context."""

    host_telemetry = dict(host_telemetry or read_bootstrap_heartbeat() or {})
    hints: dict[str, object] = {}
    hints.update(_parse_component_inventory(component_hints))
    env_hints = _parse_component_inventory(os.getenv(_BOOTSTRAP_COMPONENT_HINTS_ENV))
    hints.update({k: v for k, v in env_hints.items() if k not in hints})
    complexity = _parse_component_inventory(pipeline_complexity)
    complexity.update({k: v for k, v in hints.items() if k not in complexity})

    if load_average is None:
        load_average = _parse_float(str(host_telemetry.get("host_load")))
    if load_average is None:
        load_average = _host_load_average()

    guard_context = get_bootstrap_guard_context()
    try:
        guard_scale = max(float(guard_context.get("budget_scale", 1.0) or 1.0), 1.0)
    except Exception:
        guard_scale = 1.0
    load_scale = _host_load_scale(load_average)
    scale = max(load_scale, guard_scale)

    def _count(value: object) -> int:
        if isinstance(value, (list, tuple, set, frozenset)):
            return len(value)
        if isinstance(value, Mapping):
            return len(value)
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    component_complexity: dict[str, float] = {}
    component_complexity["vectorizers"] = 1.0 + max(0, _count(complexity.get("vectorizers")) - 1) * 0.35
    component_complexity["retrievers"] = 1.0 + max(0, _count(complexity.get("retrievers")) - 1) * 0.25
    component_complexity["db_indexes"] = 1.0 + max(0, _count(complexity.get("db_indexes")) - 1) * 0.2
    component_complexity["background_loops"] = 1.0 + max(0, _count(complexity.get("background_loops"))) * 0.15
    component_complexity["pipeline_config"] = 1.0 + max(
        0, _count(complexity.get("pipeline_config_sections")) - 1
    ) * 0.1
    component_complexity["orchestrator_state"] = 1.0 + max(
        0, _count(complexity.get("orchestrator_components")) - 1
    ) * 0.1

    adaptive_minimums: dict[str, float] = {}
    for component, minimum in _COMPONENT_TIMEOUT_MINIMUMS.items():
        adaptive_minimums[component] = minimum * scale * component_complexity.get(component, 1.0)

    return adaptive_minimums


def load_escalated_timeout_floors() -> dict[str, float]:
    """Return timeout floors that include host-scoped persisted escalations."""

    state = _load_timeout_state()
    host_state = state.get(_state_host_key(), {}) if isinstance(state, dict) else {}
    scale, scale_context = _cluster_budget_scale(host_state=state)

    minimums = {key: value * scale for key, value in _BOOTSTRAP_TIMEOUT_MINIMUMS.items()}

    component_overruns = (
        host_state.get("component_overruns", {}) if isinstance(host_state, dict) else {}
    )
    decay_streak = int(host_state.get("success_streak", 0) or 0)
    adaptive_notes: dict[str, object] = {
        "component_overruns": component_overruns,
        "decay_streak": decay_streak,
        "cluster_scale": scale_context | {"scale": scale},
    }
    state_updated = False

    for env_var, default_minimum in list(minimums.items()):
        try:
            host_value = float(host_state.get(env_var, default_minimum))
        except (TypeError, ValueError):
            host_value = default_minimum
        minimums[env_var] = max(default_minimum, host_value)

    for component, env_var in _ALL_COMPONENT_ENV_MAPPING.items():
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

    state = _load_timeout_state()
    guard_context = get_bootstrap_guard_context()
    host_telemetry = read_bootstrap_heartbeat()
    historical = load_last_component_budgets()
    adaptive_minimums = compute_adaptive_component_minimums(
        host_telemetry=host_telemetry
    )
    base_defaults: dict[str, float] = {}
    for component, minimum in _COMPONENT_TIMEOUT_MINIMUMS.items():
        try:
            historical_floor = float(historical.get(component, 0.0) or 0.0)
        except (TypeError, ValueError):
            historical_floor = 0.0
        adaptive_floor = adaptive_minimums.get(component, 0.0)
        base_defaults[component] = max(minimum, historical_floor, adaptive_floor)
    scale, scale_context = _derive_component_floor_scale(
        guard_context=guard_context, host_state=state, host_telemetry=host_telemetry
    )

    heartbeat_scale = 1.0
    heartbeat_meta: dict[str, object] = {}
    threshold = _parse_float(os.getenv(_BOOTSTRAP_LOAD_THRESHOLD_ENV)) or _DEFAULT_LOAD_THRESHOLD
    if isinstance(host_telemetry, Mapping):
        heartbeat_meta = dict(host_telemetry)
        heartbeat_load = _parse_float(str(host_telemetry.get("host_load")))
        if heartbeat_load is not None and threshold:
            heartbeat_scale = max(heartbeat_scale, 1.0 + max(heartbeat_load / threshold - 1.0, 0.0))
        active_clusters = 0
        for key in ("active_clusters", "clusters_bootstrapping", "cluster_backlog"):
            try:
                active_clusters = max(active_clusters, int(host_telemetry.get(key) or 0))
            except (TypeError, ValueError):
                continue
        if not active_clusters:
            peers = _recent_peer_activity(max_age=_DEFAULT_HEARTBEAT_MAX_AGE * 2)
            active_clusters = len({str(peer.get("host")) for peer in peers})
        if active_clusters > 1:
            heartbeat_scale = max(heartbeat_scale, 1.0 + min(active_clusters - 1, 4) * 0.08)

    vector_heavy = False
    vector_hint = None
    if isinstance(host_telemetry, Mapping):
        vector_heavy = bool(host_telemetry.get("vector_heavy"))
        vector_hint = _parse_float(str(host_telemetry.get("vector_longest")))
        if vector_hint is None:
            vector_hint = _parse_float(str(host_telemetry.get("vector_backlog")))
    heartbeat_scale = max(heartbeat_scale, 1.0)
    max_scale = _parse_float(os.getenv(_COMPONENT_FLOOR_MAX_SCALE_ENV)) or _DEFAULT_COMPONENT_FLOOR_MAX_SCALE
    component_floors = {
        key: min(value * max(scale, heartbeat_scale), value * max_scale)
        for key, value in base_defaults.items()
    }

    if vector_heavy or (vector_hint and vector_hint > 0):
        vector_scale = 1.0 + min(max(vector_hint or 1.0, 1.0) / 200.0, 0.5)
        component_floors["vectorizers"] = min(
            component_floors.get("vectorizers", base_defaults.get("vectorizers", 0.0))
            * vector_scale,
            base_defaults.get("vectorizers", 0.0) * max_scale,
        )
        component_floors["retrievers"] = min(
            component_floors.get("retrievers", base_defaults.get("retrievers", 0.0))
            * max(vector_scale, 1.1),
            base_defaults.get("retrievers", 0.0) * max_scale,
        )
    host_key = _state_host_key()
    host_state = state.get(host_key, {}) if isinstance(state, dict) else {}
    host_component_floors = (
        host_state.get("component_floors", {}) if isinstance(host_state, dict) else {}
    )
    host_overruns = host_state.get("component_overruns", {}) if isinstance(host_state, Mapping) else {}

    for component, default_minimum in list(component_floors.items()):
        try:
            host_value = float(host_component_floors.get(component, default_minimum))
        except (TypeError, ValueError):
            host_value = default_minimum
        component_floors[component] = max(default_minimum, host_value, component_floors[component])

        if isinstance(host_overruns, Mapping):
            overrun_meta = host_overruns.get(component, {}) if isinstance(host_overruns, Mapping) else {}
            suggested = _parse_float(str(overrun_meta.get("suggested_floor")))
            max_elapsed = _parse_float(str(overrun_meta.get("max_elapsed")))
            expected_floor = _parse_float(str(overrun_meta.get("expected_floor")))
            candidates = [component_floors[component]]
            if suggested is not None:
                candidates.append(suggested)
            if max_elapsed:
                candidates.append(float(max_elapsed) + 30.0)
            if expected_floor:
                candidates.append(float(expected_floor) * 1.1)
            component_floors[component] = max(candidates)

    if isinstance(state, Mapping):
        for peer_state in state.values():
            peer_floors = peer_state.get("component_floors", {}) if isinstance(peer_state, Mapping) else {}
            if not isinstance(peer_floors, Mapping):
                continue
            for component, value in peer_floors.items():
                try:
                    peer_floor = float(value)
                except (TypeError, ValueError):
                    continue
                base = _COMPONENT_TIMEOUT_MINIMUMS.get(component, 0.0)
                ceiling = base * max_scale if base else peer_floor
                component_floors[component] = max(
                    component_floors.get(component, base), min(peer_floor, ceiling)
                )

    floor_context = {
        "component_cluster_scale": scale_context | {"scale": scale, "max_scale": max_scale},
        "component_floor_inputs": {
            "heartbeat": heartbeat_meta,
            "vector_hint": vector_hint,
            "vector_heavy": vector_heavy,
            "max_scale": max_scale,
            "heartbeat_scale": heartbeat_scale,
        },
        "component_floors": dict(component_floors),
    }

    _record_adaptive_context(floor_context)

    if isinstance(host_state, dict):
        host_state["component_floors"] = dict(component_floors)
        host_state["component_floor_inputs"] = {
            "guard": guard_context,
            "telemetry": host_telemetry,
            "scale_context": scale_context,
            "scale": scale,
            "max_scale": max_scale,
        }
        if isinstance(state, dict):
            state[host_key] = host_state
            _save_timeout_state(state)

    return component_floors


def load_deferred_component_timeout_floors() -> dict[str, float]:
    """Return timeout floors for deferred bootstrap work such as background loops."""

    state = _load_timeout_state()
    guard_context = get_bootstrap_guard_context()
    host_telemetry = read_bootstrap_heartbeat()
    historical = load_last_component_budgets()
    base_defaults: dict[str, float] = {}
    for component, minimum in _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS.items():
        try:
            historical_floor = float(historical.get(component, 0.0) or 0.0)
        except (TypeError, ValueError):
            historical_floor = 0.0
        base_defaults[component] = max(minimum, historical_floor)
    scale, scale_context = _derive_component_floor_scale(
        guard_context=guard_context, host_state=state, host_telemetry=host_telemetry
    )
    max_scale = _parse_float(os.getenv(_COMPONENT_FLOOR_MAX_SCALE_ENV)) or _DEFAULT_COMPONENT_FLOOR_MAX_SCALE
    component_floors = {
        key: min(value * scale, value * max_scale)
        for key, value in base_defaults.items()
    }
    host_key = _state_host_key()
    host_state = state.get(host_key, {}) if isinstance(state, dict) else {}
    host_component_floors = (
        host_state.get("deferred_component_floors", {}) if isinstance(host_state, dict) else {}
    )
    if not host_component_floors:
        host_component_floors = host_state.get("component_floors", {}) if isinstance(host_state, dict) else {}
    host_component_floors = {
        key: value for key, value in (host_component_floors or {}).items() if key in DEFERRED_COMPONENTS
    }

    for component, default_minimum in list(component_floors.items()):
        try:
            host_value = float(host_component_floors.get(component, default_minimum))
        except (TypeError, ValueError):
            host_value = default_minimum
        component_floors[component] = max(default_minimum, host_value, component_floors[component])

    if isinstance(state, Mapping):
        for peer_state in state.values():
            peer_floors = peer_state.get("deferred_component_floors", {}) if isinstance(peer_state, Mapping) else {}
            if not peer_floors and isinstance(peer_state, Mapping):
                peer_floors = peer_state.get("component_floors", {})
            if not isinstance(peer_floors, Mapping):
                continue
            peer_floors = {key: value for key, value in peer_floors.items() if key in DEFERRED_COMPONENTS}
            for component, value in peer_floors.items():
                try:
                    peer_floor = float(value)
                except (TypeError, ValueError):
                    continue
                base = _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS.get(component, 0.0)
                ceiling = base * max_scale if base else peer_floor
                component_floors[component] = max(
                    component_floors.get(component, base), min(peer_floor, ceiling)
                )

    _record_adaptive_context(
        {
            "deferred_component_cluster_scale": scale_context
            | {"scale": scale, "max_scale": max_scale},
        }
    )

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


def load_last_component_budgets() -> dict[str, float]:
    """Return the last persisted component budgets when available."""

    state = _load_timeout_state()
    host_state = state.get(_state_host_key(), {}) if isinstance(state, dict) else {}
    raw_budgets = host_state.get("last_component_budgets", {}) if isinstance(host_state, Mapping) else {}
    budgets: dict[str, float] = {}

    if isinstance(raw_budgets, Mapping):
        for key, value in raw_budgets.items():
            try:
                budgets[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    total = host_state.get("last_component_budget_total") if isinstance(host_state, Mapping) else None
    try:
        budgets["__total__"] = float(total)  # type: ignore[assignment]
    except (TypeError, ValueError):
        pass

    deferred_total = (
        host_state.get("last_deferred_component_budget_total") if isinstance(host_state, Mapping) else None
    )
    try:
        budgets["__deferred_total__"] = float(deferred_total)  # type: ignore[assignment]
    except (TypeError, ValueError):
        pass
    deferred_budgets = host_state.get("last_deferred_component_budgets", {}) if isinstance(host_state, Mapping) else {}
    if isinstance(deferred_budgets, Mapping):
        for key, value in deferred_budgets.items():
            try:
                budgets[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    return budgets


def load_component_budget_pools() -> dict[str, float]:
    """Return persisted per-component budget pools when available."""

    state = _load_timeout_state()
    host_state = state.get(_state_host_key(), {}) if isinstance(state, dict) else {}
    pools = host_state.get("component_budget_pools", {}) if isinstance(host_state, Mapping) else {}
    if not pools and isinstance(host_state, Mapping):
        pools = host_state.get("last_component_budgets", {})

    resolved: dict[str, float] = {}
    if isinstance(pools, Mapping):
        for key, value in pools.items():
            try:
                resolved[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    return resolved


def load_last_global_bootstrap_window() -> tuple[float | None, Mapping[str, object]]:
    """Return the last persisted global bootstrap window and inputs."""

    state = _load_timeout_state()
    host_state = state.get(_state_host_key(), {}) if isinstance(state, dict) else {}
    window = None
    try:
        window_val = host_state.get("last_global_bootstrap_window")
        window = float(window_val) if window_val is not None else None
    except (TypeError, ValueError):
        window = None

    inputs = host_state.get("last_global_window_inputs", {}) if isinstance(host_state, Mapping) else {}
    inputs = dict(inputs) if isinstance(inputs, Mapping) else {}
    return window, inputs


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
        if "component_budget_total" in metadata:
            window_state["component_total"] = metadata.get("component_budget_total")
        if metadata.get("component_budgets"):
            window_state["component_budgets"] = dict(metadata.get("component_budgets", {}))

    windows[key] = window_state
    host_state["bootstrap_wait_windows"] = windows
    if metadata:
        total = metadata.get("component_budget_total")
        try:
            host_state["last_component_budget_total"] = float(total) if total is not None else host_state.get(
                "last_component_budget_total"
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive casting
            pass
    state = state if isinstance(state, dict) else {}
    state[host_key] = host_state
    _save_timeout_state(state)



def compute_prepare_pipeline_component_budgets(
    *,
    component_floors: Mapping[str, float] | None = None,
    deferred_component_floors: Mapping[str, float] | None = None,
    telemetry: Mapping[str, object] | None = None,
    load_average: float | None = None,
    pipeline_complexity: Mapping[str, object] | None = None,
    host_telemetry: Mapping[str, object] | None = None,
    scheduled_component_budgets: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return proactive per-component budgets for prepare_pipeline gates."""

    telemetry = telemetry or {}
    complexity = pipeline_complexity or {}
    guard_context = get_bootstrap_guard_context()
    try:
        guard_scale = max(float(guard_context.get("budget_scale", 1.0) or 1.0), 1.0)
    except Exception:
        guard_scale = 1.0
    host_state = _load_timeout_state()
    host_key = _state_host_key()
    host_state_details = host_state.get(host_key, {}) if isinstance(host_state, dict) else {}
    adaptive_minimums = compute_adaptive_component_minimums(
        pipeline_complexity=pipeline_complexity,
        host_telemetry=host_telemetry,
        load_average=load_average,
    )
    floors = dict(component_floors or load_component_timeout_floors())
    for component, floor in adaptive_minimums.items():
        floors[component] = max(floor, floors.get(component, 0.0))
    deferred_floors = dict(deferred_component_floors or load_deferred_component_timeout_floors())
    deferred_components = set(deferred_floors)
    floors.update(deferred_floors)
    host_telemetry = dict(host_telemetry or read_bootstrap_heartbeat() or {})
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
    if load_average is None:
        load_average = _host_load_average()
    cluster_scale, cluster_context = _cluster_budget_scale(
        host_state=host_state, host_telemetry=host_telemetry, load_average=load_average
    )
    host_scale = _host_load_scale(load_average)
    scale = max(host_scale, cluster_scale)

    def _coerce_int(value: object) -> int:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    backlog_queue_depth = _coerce_int(
        host_telemetry.get("queue_depth")
        or guard_context.get("queue_depth")
        or host_state_details.get("last_component_budget_inputs", {}).get("backlog", {}).get("queue_depth")
    )
    pending_background_loops = _coerce_int(
        host_telemetry.get("pending_background_loops")
        or host_telemetry.get("background_pending")
        or host_state_details.get("last_component_budget_inputs", {}).get("backlog", {}).get("pending_background_loops")
    )
    active_bootstraps = _coerce_int(
        host_telemetry.get("active_bootstraps")
        or host_state_details.get("last_component_budget_inputs", {}).get("backlog", {}).get("active_bootstraps")
    )
    if not active_bootstraps:
        active_bootstraps = len(_recent_peer_activity())
        if host_telemetry:
            active_bootstraps = max(active_bootstraps, 1)

    backlog_scale = 1.0 + min(backlog_queue_depth, 5) * 0.05
    backlog_scale += min(active_bootstraps, 5) * 0.07
    backlog_scale += min(pending_background_loops, 5) * 0.04
    backlog_scale = min(backlog_scale, 2.0)

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

    telemetry_overruns = _summarize_component_overruns(telemetry or {})

    component_pool_scales = _derive_component_pool_scales(
        guard_context=guard_context,
        telemetry_overruns=telemetry_overruns,
        host_overruns=host_overruns,
        components=floors.keys(),
    )

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
        key: value
        * scale
        * backlog_scale
        * component_complexity.get(key, 1.0)
        * component_pool_scales.get(key, 1.0)
        for key, value in floors.items()
    }
    component_budget_total = sum(
        value for key, value in budgets.items() if key not in deferred_components
    ) if budgets else 0.0
    deferred_component_budget_total = sum(
        value for key, value in budgets.items() if key in deferred_components
    ) if budgets else 0.0
    global_window = component_budget_total or None
    global_window_extension: Mapping[str, object] | None = None

    global_window, concurrency_meta = _derive_concurrency_global_window(
        base_window=global_window,
        component_budget_total=component_budget_total,
        component_budgets=budgets,
        scheduled_component_budgets=scheduled_component_budgets,
        deferred_components=deferred_components,
        guard_context=guard_context,
        host_telemetry=host_telemetry,
    )
    if concurrency_meta:
        global_window_extension = {"concurrency": concurrency_meta}

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

    _apply_overruns(telemetry_overruns)
    if isinstance(host_overruns, Mapping):
        _apply_overruns(host_overruns)  # type: ignore[arg-type]

    deferred_component_budgets = {
        key: value for key, value in budgets.items() if key in deferred_components
    }

    component_work_units = {
        "vectorizers": max(int(complexity.get("vectorizers", 0) or 1), 1),
        "retrievers": max(int(complexity.get("retrievers", 0) or 1), 1),
        "db_indexes": max(int(complexity.get("db_indexes", 0) or 1), 1),
        "orchestrator_state": max(int(complexity.get("db_index_bytes", 0) or 1), 1),
        "pipeline_config": max(int(complexity.get("pipeline_config_sections", 0) or 1), 1),
        "background_loops": max(int(complexity.get("background_loops", 0) or 1), 1),
    }

    global_window, extension_meta = _derive_rolling_global_window(
        base_window=global_window,
        component_budget_total=component_budget_total,
        host_telemetry=host_telemetry,
    )
    if extension_meta:
        global_window_extension = {**(global_window_extension or {}), **extension_meta}

    adaptive_inputs = {
        "host": host_key,
        "load_scale": scale,
        "host_scale": host_scale,
        "cluster_scale": cluster_context | {"scale": cluster_scale},
        "load_average": load_average,
        "component_complexity": component_complexity,
        "component_pool_scales": component_pool_scales,
        "component_work_units": component_work_units,
        "telemetry_overruns": telemetry_overruns,
        "component_budget_pools": {key: budgets.get(key, floors.get(key, 0.0)) for key in floors},
        "floors": floors,
        "adaptive_floors": adaptive_floors,
        "component_budget_total": component_budget_total,
        "deferred_component_budget_total": deferred_component_budget_total,
        "deferred_component_budgets": deferred_component_budgets,
        "global_window": global_window,
        "global_window_extension": global_window_extension,
        "concurrency_extension": concurrency_meta,
        "backlog": {
            "queue_depth": backlog_queue_depth,
            "active_bootstraps": active_bootstraps,
            "pending_background_loops": pending_background_loops,
            "scale": backlog_scale,
        },
    }
    _record_adaptive_context(adaptive_inputs)

    host_state_details = host_state_details if isinstance(host_state_details, dict) else {}
    persisted_floors = (
        host_state_details.get("component_floors", {}) if isinstance(host_state_details, Mapping) else {}
    )
    persisted_floors = dict(persisted_floors) if isinstance(persisted_floors, Mapping) else {}
    for component, floor in floors.items():
        try:
            persisted_floors[component] = max(float(floor), float(persisted_floors.get(component, 0.0) or 0.0))
        except (TypeError, ValueError):
            persisted_floors[component] = float(floor)
    for component, meta in adaptive_floors.items():
        persisted_floors[component] = float(meta.get("floor", floors.get(component, 0.0)))
    if persisted_floors:
        host_state_details["component_floors"] = persisted_floors
    deferred_persisted_floors = {
        key: persisted_floors.get(key, deferred_floors.get(key, 0.0))
        for key in deferred_components
        if key in persisted_floors or key in deferred_floors
    }
    if deferred_persisted_floors:
        host_state_details["deferred_component_floors"] = deferred_persisted_floors
    host_state_details.update(
        {
            "last_component_budgets": budgets,
            "last_deferred_component_budgets": deferred_component_budgets,
            "last_component_budget_inputs": adaptive_inputs,
            "last_component_budget_total": component_budget_total,
            "last_deferred_component_budget_total": deferred_component_budget_total,
            "last_global_bootstrap_window": global_window,
            "last_global_window_inputs": {
                "component_total": component_budget_total,
                "host_scale": host_scale,
                "cluster_scale": cluster_scale,
                "load_average": load_average,
                "component_complexity": component_complexity,
                "guard_scale": guard_scale,
                "backlog": adaptive_inputs.get("backlog"),
                "global_window_extension": global_window_extension,
            },
            "last_global_window_extension": global_window_extension,
            "component_work_units": component_work_units,
            "component_budget_pools": adaptive_inputs.get("component_budget_pools", {}),
            "component_pool_scales": component_pool_scales,
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
            "component_budget_pools": adaptive_inputs.get("component_budget_pools"),
            "component_pool_scales": component_pool_scales,
            "component_complexity": component_complexity,
            "load_scale": scale,
            "host_load": load_average,
            "host_overruns": host_overruns,
            "telemetry_overruns": telemetry_overruns,
            "state_path": str(_TIMEOUT_STATE_PATH),
        },
    )

    broadcast_timeout_floors(
        source="compute_prepare_pipeline_component_budgets",
        component_floors=persisted_floors or floors,
        timeout_floors=load_escalated_timeout_floors(),
        guard_context=guard_context,
    )

    return budgets


def derive_bootstrap_timeout_env(
    *,
    minimum: float = 240.0,
    telemetry: Mapping[str, object] | None = None,
    pipeline_complexity: Mapping[str, object] | None = None,
    host_telemetry: Mapping[str, object] | None = None,
    scheduled_component_budgets: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """Return adaptive bootstrap timeout env values derived from telemetry."""

    telemetry = telemetry or collect_timeout_telemetry()
    host_telemetry = host_telemetry or read_bootstrap_heartbeat()
    complexity: Dict[str, object] = {}
    complexity.update(_parse_component_inventory(pipeline_complexity))
    env_hints = _parse_component_inventory(os.getenv(_BOOTSTRAP_COMPONENT_HINTS_ENV))
    complexity.update({k: v for k, v in env_hints.items() if k not in complexity})
    db_index_bytes = os.getenv(_BOOTSTRAP_DB_INDEX_BYTES_ENV)
    try:
        if db_index_bytes:
            complexity.setdefault("db_index_bytes", float(db_index_bytes))
    except (TypeError, ValueError):
        pass
    complexity_scale_env = os.getenv(_BOOTSTRAP_COMPLEXITY_SCALE_ENV)
    try:
        complexity_scale = max(float(complexity_scale_env), 1.0) if complexity_scale_env else 1.0
    except (TypeError, ValueError):
        complexity_scale = 1.0
    if complexity_scale > 1.0:
        for key, value in list(complexity.items()):
            try:
                complexity[key] = max(int(round(float(value) * complexity_scale)), int(value))
            except (TypeError, ValueError):
                complexity[key] = value
        complexity["complexity_scale"] = complexity_scale

    floors = load_escalated_timeout_floors()
    component_budgets = compute_prepare_pipeline_component_budgets(
        telemetry=telemetry,
        pipeline_complexity=complexity,
        host_telemetry=host_telemetry,
        scheduled_component_budgets=scheduled_component_budgets,
    )
    adaptive_context = get_adaptive_timeout_context()
    resolved: Dict[str, float] = {}

    for env_var, floor in floors.items():
        try:
            resolved[env_var] = max(float(floor), minimum)
        except (TypeError, ValueError):
            resolved[env_var] = minimum

    for component, budget in component_budgets.items():
        env_var = _ALL_COMPONENT_ENV_MAPPING.get(component)
        if not env_var:
            continue
        try:
            resolved[env_var] = max(float(budget), resolved.get(env_var, minimum))
        except (TypeError, ValueError):
            resolved[env_var] = resolved.get(env_var, minimum)

    global_window = adaptive_context.get("global_window")
    try:
        window_value = float(global_window) if global_window is not None else None
    except (TypeError, ValueError):
        window_value = None

    if window_value:
        for env_var in ("MENACE_BOOTSTRAP_WAIT_SECS", "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"):
            resolved[env_var] = max(resolved.get(env_var, minimum), window_value)

    for env_var, floor in _BOOTSTRAP_TIMEOUT_MINIMUMS.items():
        resolved[env_var] = max(resolved.get(env_var, minimum), floor if floor is not None else minimum)

    return resolved


def _collect_timeout_telemetry() -> Mapping[str, object]:
    try:
        from coding_bot_interface import _PREPARE_PIPELINE_WATCHDOG

        return {
            "timeouts": int(_PREPARE_PIPELINE_WATCHDOG.get("timeouts", 0)),
            "stages": list(_PREPARE_PIPELINE_WATCHDOG.get("stages", ())),
            "shared_timeout": _PREPARE_PIPELINE_WATCHDOG.get("shared_timeout"),
            "extensions": list(_PREPARE_PIPELINE_WATCHDOG.get("extensions", ())),
            "component_windows": _PREPARE_PIPELINE_WATCHDOG.get("component_windows"),
            "global_window": _PREPARE_PIPELINE_WATCHDOG.get("global_bootstrap_window"),
            "global_window_extension": _PREPARE_PIPELINE_WATCHDOG.get(
                "global_bootstrap_extension"
            ),
            "component_complexity": _PREPARE_PIPELINE_WATCHDOG.get("component_complexity"),
        }
    except Exception:
        return {}


def collect_timeout_telemetry() -> Mapping[str, object]:
    """Public wrapper used by other modules to consume timeout telemetry."""

    return _collect_timeout_telemetry()


def _categorize_stage(entry: Mapping[str, object]) -> str | None:
    components = entry.get("components") or entry.get("component")
    if isinstance(components, (list, tuple, set)):
        for component in components:
            normalized = str(component).strip().lower()
            if normalized in _COMPONENT_TIMEOUT_MINIMUMS:
                return normalized
    elif isinstance(components, str) and components in _COMPONENT_TIMEOUT_MINIMUMS:
        return components

    label = str(entry.get("label", "")).lower()
    if "vector" in label:
        return "vectorizers"
    if "retriev" in label:
        return "retrievers"
    if "db" in label or "index" in label:
        return "db_indexes"
    if "orchestrator" in label:
        return "orchestrator_state"
    if any(token in label for token in ("config", "context", "pipeline")):
        return "pipeline_config"
    if any(token in label for token in ("background", "loop", "scheduler")):
        return "background_loops"
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
        if streak >= _OVERRUN_STREAK_THRESHOLD and component in _ALL_COMPONENT_ENV_MAPPING:
            baseline = _COMPONENT_TIMEOUT_MINIMUMS.get(
                component, _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS.get(component, 0.0)
            )
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


def render_prepare_pipeline_timeout_hints(
    vector_heavy: bool | None = None,
    *,
    components: Mapping[str, Mapping[str, object]] | Iterable[str] | None = None,
    overruns: Mapping[str, Mapping[str, object]] | None = None,
) -> list[str]:
    """Return remediation hints for ``prepare_pipeline_for_bootstrap`` timeouts.

    ``components`` allows callers to highlight the specific gates that overran so
    the suggested knobs include the relevant environment variables instead of a
    single global deadline.
    """

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

    component_map: Mapping[str, Mapping[str, object]] = {}
    component_labels: Iterable[str] | None = None
    if isinstance(components, Mapping):
        component_map = components
        component_labels = components.keys()
    else:
        component_labels = components

    dominant_component: str | None = None
    dominant_spent: float | None = None
    dominant_budget: float | None = None
    dominant_remaining: float | None = None

    if component_labels:
        for component in sorted(set(component_labels)):
            env_var = _ALL_COMPONENT_ENV_MAPPING.get(component)
            if env_var:
                floor = component_floors.get(component) or _COMPONENT_TIMEOUT_MINIMUMS.get(
                    component, _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS.get(component, 0.0)
                )
                budget = component_map.get(component, {}) if component_map else {}
                budget_clause = ""
                if isinstance(budget, Mapping):
                    budget_value = _parse_float(
                        str(
                            budget.get("budget")
                            or budget.get("effective")
                            or budget.get("component_budget")
                        )
                    )
                    remaining = _parse_float(
                        str(
                            budget.get("remaining")
                            or budget.get("component_remaining_after")
                            or budget.get("component_remaining")
                        )
                    )
                    if budget_value is not None and remaining is not None:
                        spent = budget_value - remaining
                        if dominant_spent is None or spent > dominant_spent:
                            dominant_spent = spent
                            dominant_budget = budget_value
                            dominant_remaining = remaining
                            dominant_component = component
                        budget_clause = (
                            f" (spent {spent:.1f}s of {budget_value:.1f}s, remaining {remaining:.1f}s)"
                        )
                    elif budget_value is not None:
                        budget_clause = f" (budget {budget_value:.1f}s)"
                        dominant_spent = dominant_spent or 0.0
                        dominant_budget = dominant_budget or budget_value
                        dominant_component = dominant_component or component
                        if dominant_remaining is None:
                            dominant_remaining = budget_value

                if overruns and component in overruns:
                    overrun_meta = overruns.get(component) or {}
                    suggested_floor = _parse_float(
                        str(
                            (overrun_meta.get("suggested_floor") if isinstance(overrun_meta, Mapping) else None)
                            or (overrun_meta.get("expected_floor") if isinstance(overrun_meta, Mapping) else None)
                            or (overrun_meta.get("floor") if isinstance(overrun_meta, Mapping) else None)
                        )
                    )
                    if suggested_floor and suggested_floor > floor:
                        floor = suggested_floor
                hints.append(
                    (
                        f"Repeated {component} overruns detected; increase {env_var}"
                        f" to at least {int(floor)}s to grant that gate more room without"
                        f" stretching the global deadline.{budget_clause}"
                    )
                )

    if dominant_component:
        env_var = _ALL_COMPONENT_ENV_MAPPING.get(dominant_component)
        target_budget = dominant_budget or component_floors.get(dominant_component) or _COMPONENT_TIMEOUT_MINIMUMS.get(
            dominant_component, _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS.get(dominant_component, 0.0)
        )
        baseline_env_hint = (
            f" {env_var}" if env_var else " MENACE_BOOTSTRAP_WAIT_SECS"
        )
        hints.insert(
            0,
            (
                f"{dominant_component} is consuming the bootstrap window"
                f" (spent {dominant_spent:.1f}s, remaining {dominant_remaining:.1f}s);"
                f" raise{baseline_env_hint} to >= {int(target_budget)}s or set a component override"
                f" to avoid tripping the aggregate deadline."
            ),
        )

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

        component_floor_inputs = {}
        if isinstance(adaptive_context, Mapping):
            component_floor_inputs = adaptive_context.get("component_floor_inputs", {})
        if component_floor_inputs:
            heartbeat_meta = component_floor_inputs.get("heartbeat", {}) if isinstance(component_floor_inputs, Mapping) else {}
            heartbeat_load = heartbeat_meta.get("host_load") if isinstance(heartbeat_meta, Mapping) else None
            hints.append(
                (
                    "Component floors are being elevated using persisted heartbeat telemetry"
                    f" (host_load={heartbeat_load}, vector_hint={component_floor_inputs.get('vector_hint')});"
                    f" override with {_OVERRIDE_ENV}=1 to bypass clamping."
                )
            )

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
        global_window: float | None = None,
        complexity_inputs: Mapping[str, object] | None = None,
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
        self._deadline_extensions: list[dict[str, object]] = []
        self._expanded_global_window: float | None = global_window or total_budget
        self._signal_hook = signal_hook
        self.global_window = global_window
        self.complexity_inputs = dict(complexity_inputs or {})
        self._historical_budgets = load_last_component_budgets()
        self.component_states: dict[str, str] = {}

    def _component_baseline(self, label: str) -> float:
        candidates = [
            self.component_budgets.get(label, 0.0),
            self.component_floors.get(label, 0.0),
            self._historical_budgets.get(label, 0.0),
            _COMPONENT_TIMEOUT_MINIMUMS.get(label, 0.0),
            _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS.get(label, 0.0),
        ]
        try:
            return max(float(value) for value in candidates)
        except (TypeError, ValueError):
            return _COMPONENT_TIMEOUT_MINIMUMS.get(label, 0.0)

    def _register_component_window(self, label: str, budget: float | None) -> None:
        """Track component windows and expand the aggregate deadline when needed."""

        budget = budget if budget is not None else 0.0
        with self._lock:
            self._component_windows.setdefault(label, {"budget": budget, "remaining": budget})
            self._component_windows[label].update({"budget": budget, "remaining": budget})

            load_scale = 1.0
            host_load = _host_load_average()
            if host_load is not None:
                load_scale += max(host_load - 1.0, 0.0) * 0.35

            def _window_budget(window_label: str, window: Mapping[str, float | None]) -> float:
                baseline = float(window.get("budget") or 0.0)
                return max(baseline, self._component_baseline(window_label))

            window_total = sum(
                _window_budget(name, meta) for name, meta in self._component_windows.items()
            )
            scaled_total = window_total * load_scale if window_total else None

            expanded = scaled_total
            if self.total_budget is not None:
                expanded = max(self.total_budget, scaled_total or 0.0)

            if expanded is not None and expanded > (self._expanded_global_window or 0.0):
                previous = self._expanded_global_window
                self._expanded_global_window = expanded
                extension = {
                    "event": "component-deadline-extended",
                    "gate": label,
                    "expanded_window": expanded,
                    "previous_window": previous,
                    "load_scale": load_scale,
                    "component_count": len(self._component_windows),
                }
                self._deadline_extensions.append(extension)
                self.global_window = max(self.global_window or 0.0, expanded)
                self.remaining_budget = expanded

    def _reserve(
        self,
        label: str,
        requested: float | None,
        minimum: float,
        metadata: Mapping[str, object] | None,
        *,
        component_timer: bool = False,
        ) -> tuple[float | None, MutableMapping[str, object]]:
        with self._lock:
            component_floor = max(self.component_floors.get(label, 0.0), self._component_baseline(label))
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
            if self.remaining_budget is not None and not component_timer:
                effective = min(effective, max(self.remaining_budget, 0.0))
                self.remaining_budget = max(self.remaining_budget - effective, 0.0)
            elif component_timer:
                self._register_component_window(label, effective)

            record: MutableMapping[str, object] = {
                "label": label,
                "component": label,
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
            floor = max(minimum, self.component_floors.get(label, minimum))
            effective, record = self._reserve(
                label,
                requested,
                floor,
                {"component_timer": True},
                component_timer=True,
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
            "global_window": self.global_window,
            "progressing": True,
        }
        state = self.component_states.get(label)
        if state:
            record["component_state"] = state
        if self.complexity_inputs:
            record["component_complexity"] = dict(self.complexity_inputs)
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

    def mark_component_state(self, component: str, state: str) -> None:
        """Track component readiness without halting background work."""

        with self._lock:
            self.component_states[component] = state
        payload = {
            "component": component,
            "state": state,
            "namespace": self.namespace,
            "ts": time.time(),
            "progressing": state not in {"failed", "blocked"},
        }
        if self._signal_hook:
            try:
                self._signal_hook(dict(payload))
            except Exception:
                self.logger.debug("state signal hook failed", exc_info=True)

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
                "deadline_extensions": list(self._deadline_extensions),
                "expanded_global_window": self._expanded_global_window,
                "global_window": self.global_window,
                "complexity_inputs": dict(self.complexity_inputs),
                "component_states": dict(self.component_states),
            }
