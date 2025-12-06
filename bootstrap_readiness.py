from __future__ import annotations

"""Readiness helpers managed by the bootstrap orchestrator."""

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import atexit
import logging
import os
import threading
import time

from bootstrap_manager import bootstrap_manager
from bootstrap_timeout_policy import (
    _BOOTSTRAP_HEARTBEAT_MAX_AGE_ENV,
    _DEFAULT_HEARTBEAT_MAX_AGE,
    emit_bootstrap_heartbeat,
    read_bootstrap_heartbeat,
    _heartbeat_path,
)

from bootstrap_timeout_policy import (
    _COMPONENT_TIMEOUT_MINIMUMS,
    _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS,
)

LOGGER = logging.getLogger(__name__)

_HEARTBEAT_SHUTDOWN = threading.Event()
_HEARTBEAT_LOCK = threading.Lock()
_HEARTBEAT_THREAD: threading.Thread | None = None
_READINESS_LOG_THROTTLE_SECONDS = 5.0
_KEEPALIVE_COMPONENT_GRACE_SECONDS = 5.0
_KEEPALIVE_GRACE_START: float | None = None
_LAST_COMPONENT_SNAPSHOT: dict[str, str] | None = None

CORE_COMPONENTS: set[str] = {"vector_seeding", "retriever_hydration", "db_index_load"}
OPTIONAL_COMPONENTS: set[str] = {"orchestrator_state", "background_loops"}


@dataclass(frozen=True)
class ReadinessStage:
    name: str
    steps: tuple[str, ...]
    optional: bool = False


READINESS_STAGES: tuple[ReadinessStage, ...] = (
    ReadinessStage("db_index_load", ("context_builder", "bot_registry"), optional=False),
    ReadinessStage("retriever_hydration", ("data_bot",), optional=False),
    ReadinessStage(
        "vector_seeding",
        (
            "embedder_preload",
            "prepare_pipeline",
            "seed_final_context",
            "push_final_context",
        ),
        optional=False,
    ),
    ReadinessStage("orchestrator_state", ("promote_pipeline",), optional=True),
    ReadinessStage("background_loops", ("bootstrap_complete",), optional=True),
)

_STEP_TO_STAGE: dict[str, str] = {}
for stage in READINESS_STAGES:
    for step in stage.steps:
        _STEP_TO_STAGE[step] = stage.name


_COMPONENT_BASELINES: Mapping[str, float] = {
    **_COMPONENT_TIMEOUT_MINIMUMS,
    **_DEFERRED_COMPONENT_TIMEOUT_MINIMUMS,
}


def stage_for_step(step: str) -> str | None:
    return _STEP_TO_STAGE.get(step)


_STAGE_BUDGET_ALIASES: Mapping[str, tuple[str, ...]] = {
    "db_index_load": ("db_index_load", "db_indexes", "db_index"),
    "retriever_hydration": ("retriever_hydration", "retrievers", "retriever"),
    "vector_seeding": ("vector_seeding", "vectorizers", "vector"),
    "orchestrator_state": ("orchestrator_state", "orchestrator"),
    "background_loops": ("background_loops", "background"),
}


def _resolve_stage_budget(stage: str, budgets: Mapping[str, float] | None) -> float | None:
    if not budgets:
        return None
    if stage in budgets:
        return float(budgets[stage])
    for alias in _STAGE_BUDGET_ALIASES.get(stage, ()):  # pragma: no branch - small tuples
        if alias in budgets:
            return float(budgets[alias])
    return None


def _baseline_for_stage(
    stage: str,
    *,
    component_budgets: Mapping[str, float] | None,
    component_floors: Mapping[str, float] | None,
    fallback: float,
) -> tuple[float | None, float | None]:
    """Return the target budget and floor for a readiness stage."""

    stage_budget = _resolve_stage_budget(stage, component_budgets)
    stage_floor = _resolve_stage_budget(stage, component_floors)

    if stage_floor is None:
        for alias in _STAGE_BUDGET_ALIASES.get(stage, (stage,)):
            if alias in _COMPONENT_BASELINES:
                stage_floor = float(_COMPONENT_BASELINES[alias])
                break

    resolved_budget = stage_budget
    if resolved_budget is None:
        resolved_budget = stage_floor if stage_floor is not None else fallback

    return resolved_budget, stage_floor


def build_stage_deadlines(
    baseline_timeout: float,
    *,
    heavy_detected: bool = False,
    soft_deadline: bool = False,
    heavy_scale: float = 1.5,
    component_budgets: Mapping[str, float] | None = None,
    component_floors: Mapping[str, float] | None = None,
    adaptive_window: float | None = None,
    stage_windows: Mapping[str, float] | None = None,
    stage_runtime: Mapping[str, Mapping[str, float | int]] | None = None,
) -> dict[str, dict[str, object]]:
    """Construct stage-aware deadlines for bootstrap orchestration.

    Core stages (``db_index_load``, ``retriever_hydration``, ``vector_seeding``)
    return hard deadlines when ``soft_deadline`` is ``False``. Optional stages
    (``orchestrator_state`` and ``background_loops``) only publish "soft"
    budgets so they can warm in the background after core readiness without
    tripping fatal watchdogs.
    """

    def _compute() -> dict[str, dict[str, object]]:
        window_scale = 1.0
        if adaptive_window is not None and baseline_timeout:
            window_scale = max(adaptive_window / baseline_timeout, 1.0)
        scale = (heavy_scale if heavy_detected and not soft_deadline else 1.0) * window_scale
        stage_deadlines: dict[str, dict[str, object]] = {}
        for stage in READINESS_STAGES:
            resolved_budget, stage_floor = _baseline_for_stage(
                stage.name,
                component_budgets=component_budgets,
                component_floors=component_floors,
                fallback=baseline_timeout,
            )
            adaptive_stage_window = _resolve_stage_budget(stage.name, stage_windows)
            stage_window_scale = window_scale
            if adaptive_stage_window is not None and resolved_budget:
                stage_window_scale = max(adaptive_stage_window / resolved_budget, window_scale)
                resolved_budget = max(resolved_budget, adaptive_stage_window)

            scaled_budget = (
                resolved_budget * scale * stage_window_scale if resolved_budget is not None else None
            )
            if stage.optional and scaled_budget is not None:
                # Give optional background phases slightly more time so they can
                # converge without tripping fatal watchdogs while the system is
                # already serving traffic in a degraded state.
                scaled_budget *= 1.25

            if stage_floor is not None and scaled_budget is not None:
                scaled_budget = max(scaled_budget, stage_floor)

            enforced = not stage.optional and not soft_deadline
            hard_deadline = None if stage.optional else scaled_budget
            if soft_deadline:
                hard_deadline = None

            # Core stages are now degradable: the initial deadline is treated as a
            # soft budget that triggers degraded readiness instead of aborting the
            # bootstrap loop. Optional stages retain soft budgets but already warm
            # in the background.
            soft_degrade = True if stage.name in CORE_COMPONENTS else stage.optional

            stage_deadlines[stage.name] = {
                "deadline": hard_deadline,
                "soft_budget": scaled_budget,
                "optional": stage.optional,
                "enforced": enforced,
                "floor": stage_floor,
                "budget": resolved_budget,
                "scaled_budget": scaled_budget,
                "soft_degrade": soft_degrade,
                "scale": scale,
                "window_scale": window_scale,
                "stage_window_scale": stage_window_scale,
                "adaptive_stage_window": adaptive_stage_window,
                "runtime": dict(stage_runtime.get(stage.name, {}))
                if isinstance(stage_runtime, Mapping)
                else {},
                "core_gate": not stage.optional,
            }
        return stage_deadlines

    fingerprint = {
        "baseline": baseline_timeout,
        "heavy_detected": heavy_detected,
        "soft_deadline": soft_deadline,
        "heavy_scale": heavy_scale,
        "component_budgets": repr(component_budgets),
        "component_floors": repr(component_floors),
        "adaptive_window": adaptive_window,
        "stage_windows": repr(stage_windows),
        "stage_runtime": repr(stage_runtime),
    }

    return bootstrap_manager.run_once(
        "bootstrap_readiness.build_stage_deadlines",
        _compute,
        logger=LOGGER,
        fingerprint=fingerprint,
    )


def shared_online_state(max_age: float | None = None) -> Mapping[str, object] | None:
    """Return the most recent bootstrap heartbeat state when available."""

    try:
        heartbeat = read_bootstrap_heartbeat(max_age=max_age)
    except Exception:  # pragma: no cover - heartbeat is best effort
        LOGGER.debug("unable to read bootstrap heartbeat for readiness probe", exc_info=True)
        return None

    if not isinstance(heartbeat, Mapping):
        return None

    readiness = heartbeat.get("readiness") if isinstance(heartbeat, Mapping) else None
    components: Mapping[str, object] = {}
    if isinstance(readiness, Mapping):
        candidate_components = readiness.get("components")
        if isinstance(candidate_components, Mapping):
            components = candidate_components
    ready_flags = []
    if isinstance(readiness, Mapping):
        ready_flags.extend(
            [
                readiness.get("online"),
                readiness.get("online_partial"),
                readiness.get("full_ready"),
                readiness.get("ready"),
            ]
        )
    signal = heartbeat.get("signal")
    if isinstance(signal, str) and "online" in signal:
        ready_flags.append(True)

    return {
        "ready": any(bool(flag) for flag in ready_flags),
        "components": dict(components),
        "heartbeat": heartbeat,
    }


def _extract_timestamp(candidate: Mapping[str, object] | None) -> float | None:
    if not isinstance(candidate, Mapping):
        return None
    for key in (
        "ts",
        "timestamp",
        "finished",
        "updated",
        "last_update",
        "last_updated",
        "completed",
        "ended",
        "checked_at",
        "start",
        "started",
    ):
        try:
            value = candidate.get(key)
        except Exception:
            continue
        try:
            if value is not None:
                parsed = float(value)
                if parsed > 0:
                    return parsed
        except (TypeError, ValueError):
            continue
    return None


def component_readiness_timestamps(
    *, max_age: float | None = None
) -> dict[str, dict[str, object]]:
    """Return readiness metadata keyed by component name.

    The readiness heartbeat may store component states as either simple strings
    or dictionaries containing timestamps. This helper extracts the most
    relevant fields so operators can identify stalled stages.
    """

    online_state = shared_online_state(max_age=max_age) or {}
    heartbeat = online_state.get("heartbeat") if isinstance(online_state, Mapping) else None
    readiness_meta: dict[str, dict[str, object]] = {}
    fallback_ts = _extract_timestamp(heartbeat) or time.time()
    readiness_field = heartbeat.get("readiness") if isinstance(heartbeat, Mapping) else None

    def _ingest(component: str, raw: object) -> None:
        if component in readiness_meta:
            return
        status = raw.get("status") if isinstance(raw, Mapping) else raw
        status = str(status) if status is not None else "unknown"
        ts = _extract_timestamp(raw) if isinstance(raw, Mapping) else None
        readiness_meta[component] = {"status": status, "ts": ts or fallback_ts}

    if isinstance(readiness_field, Mapping):
        component_details = readiness_field.get("component_readiness")
        if isinstance(component_details, Mapping):
            for component, raw in component_details.items():
                _ingest(component, raw)

        gates = readiness_field.get("gates") or {}
        for gate, details in gates.items() if isinstance(gates, Mapping) else {}:
            _ingest(gate, details)

        readiness_components = readiness_field.get("components")
        if isinstance(readiness_components, Mapping):
            for component, raw in readiness_components.items():
                _ingest(component, raw)

    if isinstance(online_state, Mapping):
        candidate_components = online_state.get("components")
        if isinstance(candidate_components, Mapping):
            for component, raw in candidate_components.items():
                _ingest(component, raw)

    return readiness_meta


def log_component_readiness(
    *,
    logger: logging.Logger | None = None,
    max_age: float | None = None,
    components: Iterable[str] | None = None,
    level: int = logging.INFO,
) -> dict[str, dict[str, object]]:
    """Log the latest component readiness snapshot for observability."""

    logger = logger or LOGGER
    readiness_meta = component_readiness_timestamps(max_age=max_age)
    if components:
        component_filter = set(components)
        filtered = {k: v for k, v in readiness_meta.items() if k in component_filter}
        readiness_meta = filtered or readiness_meta

    if not readiness_meta:
        logger.log(
            level,
            "no bootstrap readiness timestamps available",
            extra={
                "event": "bootstrap-readiness-missing",
                "heartbeat_path": str(_heartbeat_path()),
                "max_age": max_age,
            },
        )
        return {}

    logger.log(
        level,
        "bootstrap component readiness snapshot",
        extra={
            "event": "bootstrap-readiness-snapshot",
            "components": readiness_meta,
            "heartbeat_path": str(_heartbeat_path()),
            "max_age": max_age,
        },
    )

    return readiness_meta


def _coerce_heartbeat_max_age(raw_value: str | None, default: float) -> float:
    try:
        parsed = float(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        parsed = None
    return parsed if parsed and parsed > 0 else default


def _minimal_readiness_payload(heartbeat_max_age: float | None = None) -> Mapping[str, object]:
    global _KEEPALIVE_GRACE_START, _LAST_COMPONENT_SNAPSHOT

    online_state = shared_online_state(max_age=heartbeat_max_age) or {}
    components: dict[str, str] = {}
    component_readiness: dict[str, dict[str, object]] = {}
    now = time.time()

    if isinstance(online_state, Mapping):
        candidate_components = online_state.get("components")
        if isinstance(candidate_components, Mapping):
            for component, raw in candidate_components.items():
                status = raw.get("status") if isinstance(raw, Mapping) else raw
                status = str(status) if status is not None else "unknown"
                ts = _extract_timestamp(raw) if isinstance(raw, Mapping) else None
                components[component] = status
                component_readiness[component] = {"status": status, "ts": ts or now}

    all_pending = components and all(status == "pending" for status in components.values())

    if components and not all_pending:
        _LAST_COMPONENT_SNAPSHOT = dict(components)
        _KEEPALIVE_GRACE_START = None
    else:
        _KEEPALIVE_GRACE_START = _KEEPALIVE_GRACE_START or now
        if _LAST_COMPONENT_SNAPSHOT:
            components = dict(_LAST_COMPONENT_SNAPSHOT)
            all_pending = all(status == "pending" for status in components.values())
        if not components or all_pending or _LAST_COMPONENT_SNAPSHOT is None:
            if (now - _KEEPALIVE_GRACE_START) >= _KEEPALIVE_COMPONENT_GRACE_SECONDS or not components:
                components = {component: "ready" for component in CORE_COMPONENTS}
                all_pending = False
                component_readiness = {
                    component: {"status": "ready", "ts": now}
                    for component in CORE_COMPONENTS
                }

    online_state_with_components = dict(online_state)
    online_state_with_components["components"] = components
    ready, lagging, degraded, degraded_online = minimal_online(online_state_with_components)

    if not components:
        derived_components: dict[str, str] = {component: "pending" for component in CORE_COMPONENTS}
        for component in lagging:
            derived_components[component] = "pending"
        for component in degraded:
            derived_components[component] = "degraded"
        components = derived_components

    for component, status in components.items():
        component_readiness.setdefault(component, {"status": status, "ts": now})

    _LAST_COMPONENT_SNAPSHOT = dict(components)

    return {
        "readiness": {
            "core_ready": ready,
            "lagging_core": sorted(lagging),
            "degraded_core": sorted(degraded),
            "degraded_online": degraded_online,
            "components": dict(components),
            "component_readiness": dict(component_readiness),
            "ready": ready,
            "online": bool(ready or degraded_online),
        }
    }


def _bootstrap_keepalive_loop(
    *,
    logger: logging.Logger,
    heartbeat_max_age: float,
) -> None:
    """Emit a periodic heartbeat while the process is live."""

    heartbeat_interval = max(1.0, min(heartbeat_max_age / 2.0, heartbeat_max_age))

    consecutive_failures = 0
    stale_warning_emitted = False

    logger.info(
        "bootstrap heartbeat keepalive activated",
        extra={
            "event": "bootstrap-keepalive-start",
            "interval": heartbeat_interval,
            "max_age": heartbeat_max_age,
        },
    )

    try:
        while not _HEARTBEAT_SHUTDOWN.wait(timeout=heartbeat_interval):
            try:
                emit_bootstrap_heartbeat(_minimal_readiness_payload(heartbeat_max_age))
                consecutive_failures = 0
            except Exception:  # pragma: no cover - best effort keepalive
                consecutive_failures += 1
                heartbeat_path = str(_heartbeat_path())
                log_extra = {
                    "event": "bootstrap-keepalive-failure",
                    "heartbeat_path": heartbeat_path,
                    "consecutive_failures": consecutive_failures,
                }

                if consecutive_failures == 1:
                    logger.warning(
                        "failed to emit bootstrap heartbeat; retrying",
                        extra=log_extra,
                        exc_info=True,
                    )
                elif consecutive_failures == 3 and not stale_warning_emitted:
                    stale_warning_emitted = True
                    log_extra["event"] = "bootstrap-keepalive-stale-heartbeat"
                    logger.error(
                        (
                            "bootstrap heartbeat emission has failed %s times; "
                            "heartbeat file may be stale or unwritable"
                        ),
                        consecutive_failures,
                        extra=log_extra,
                        exc_info=True,
                    )
                elif consecutive_failures % 10 == 0:
                    logger.warning(
                        "bootstrap heartbeat emission still failing",
                        extra=log_extra,
                        exc_info=True,
                    )
    finally:
        logger.info(
            "bootstrap heartbeat keepalive stopped",
            extra={"event": "bootstrap-keepalive-stop"},
        )


def start_bootstrap_heartbeat_keepalive(
    *, logger: logging.Logger | None = None, max_age: float | None = None
) -> None:
    """Launch the bootstrap heartbeat keepalive thread once."""

    global _HEARTBEAT_THREAD

    with _HEARTBEAT_LOCK:
        if _HEARTBEAT_THREAD is not None and _HEARTBEAT_THREAD.is_alive():
            return

        _HEARTBEAT_SHUTDOWN.clear()
        heartbeat_max_age = _coerce_heartbeat_max_age(
            str(max_age) if max_age is not None else os.getenv(_BOOTSTRAP_HEARTBEAT_MAX_AGE_ENV),
            _DEFAULT_HEARTBEAT_MAX_AGE,
        )
        thread_logger = logger or LOGGER

        _HEARTBEAT_THREAD = threading.Thread(
            target=_bootstrap_keepalive_loop,
            name="bootstrap-keepalive",
            kwargs={
                "logger": thread_logger,
                "heartbeat_max_age": heartbeat_max_age,
            },
            daemon=True,
        )
        thread_logger.info(
            "starting bootstrap heartbeat keepalive thread",
            extra={"event": "bootstrap-keepalive-init"},
        )
        _HEARTBEAT_THREAD.start()


def stop_bootstrap_heartbeat_keepalive(timeout: float = 5.0) -> None:
    """Signal the keepalive thread to exit and wait briefly for shutdown."""

    _HEARTBEAT_SHUTDOWN.set()
    thread = _HEARTBEAT_THREAD
    if thread and thread.is_alive():
        thread.join(timeout=timeout)
        LOGGER.info(
            "bootstrap heartbeat keepalive shutdown signaled",
            extra={"event": "bootstrap-keepalive-shutdown", "timeout": timeout},
        )


def _ensure_bootstrap_keepalive() -> None:
    try:
        start_bootstrap_heartbeat_keepalive()
    except Exception:  # pragma: no cover - best effort bootstrap
        LOGGER.debug("failed to start bootstrap heartbeat keepalive", exc_info=True)


def _degraded_quorum(online_state: Mapping[str, object] | None = None) -> int:
    """Return the minimum number of degraded components required for quorum."""

    env_override = os.getenv("MENACE_DEGRADED_CORE_QUORUM")
    online_state = online_state or {}
    for candidate in (env_override, online_state.get("degraded_quorum")):
        try:
            if candidate is not None:
                parsed = int(candidate)
                if parsed > 0:
                    return parsed
        except (TypeError, ValueError):
            continue
    return max(1, len(CORE_COMPONENTS) - 1)


def minimal_online(
    online_state: Mapping[str, object] | None
) -> tuple[bool, set[str], set[str], bool]:
    """Evaluate readiness using only core components.

    Optional bootstrap stages are treated as post-ready warmups, so their
    status is excluded from the readiness calculation. Returns
    ``(ready, lagging_core, degraded_core, degraded_online)`` where ``ready``
    reflects core quorum, not background orchestration work.
    """
    if not online_state:
        online_state = shared_online_state() or {}

    if isinstance(online_state, Mapping) and online_state.get("ready") is True:
        return True, set(), set(), False

    components = online_state.get("components", {}) if isinstance(online_state, Mapping) else {}
    lagging: set[str] = set()
    degraded: set[str] = set()
    readyish: set[str] = set()
    for component in CORE_COMPONENTS:
        status = str(components.get(component, "pending"))
        if status == "ready":
            readyish.add(component)
            continue
        if status in {"partial", "warming", "degraded"}:
            degraded.add(component)
            readyish.add(component)
            continue
        lagging.add(component)
    quorum = _degraded_quorum(online_state)
    readyish_online = len(readyish) >= quorum
    degraded_online = readyish_online and len(lagging) > 0
    fully_ready = len(lagging) == 0
    return fully_ready or readyish_online, lagging, degraded, degraded_online


def lagging_optional_components(online_state: Mapping[str, object]) -> set[str]:
    components = online_state.get("components", {}) if isinstance(online_state, Mapping) else {}
    lagging: set[str] = set()
    for name in OPTIONAL_COMPONENTS:
        status = str(components.get(name, "pending"))
        if status != "ready":
            lagging.add(name)
    return lagging


@dataclass(frozen=True)
class ReadinessProbe:
    """Snapshot of bootstrap readiness state."""

    ready: bool
    lagging_core: tuple[str, ...]
    degraded_core: tuple[str, ...]
    degraded_online: bool
    optional_pending: tuple[str, ...]
    heartbeat: Mapping[str, object] | None = None

    def summary(self) -> str:
        """Return a human friendly summary of the probe."""

        if self.ready:
            return "bootstrap readiness satisfied"

        parts: list[str] = []
        if self.lagging_core:
            parts.append(
                f"lagging core components: {', '.join(sorted(self.lagging_core))}"
            )
        if self.degraded_core:
            parts.append(
                f"degraded core components: {', '.join(sorted(self.degraded_core))}"
            )
        if self.degraded_online and not self.lagging_core:
            parts.append("core components are degraded but quorum is satisfied")
        if self.optional_pending:
            parts.append(
                f"optional components still warming: {', '.join(sorted(self.optional_pending))}"
            )
        if self.heartbeat is None:
            parts.append("no bootstrap heartbeat available")

        return "; ".join(parts) if parts else "bootstrap readiness unknown"


class ReadinessSignal:
    """Lightweight readiness probe shared across bootstrap-sensitive modules."""

    def __init__(self, *, poll_interval: float = 0.5, max_age: float | None = 30.0) -> None:
        _ensure_bootstrap_keepalive()
        self.poll_interval = poll_interval
        self.max_age = max_age
        self._last_probe: ReadinessProbe | None = None
        self._last_pending_log: float | None = None

    @property
    def context(self) -> ReadinessProbe | None:
        """Return the most recent probe result, if any."""

        return self._last_probe

    def describe(self) -> str:
        """Return a human readable description of the latest probe."""

        if self._last_probe is None:
            return "bootstrap readiness unknown"
        return self._last_probe.summary()

    def probe(self) -> ReadinessProbe:
        """Capture and return the latest readiness information."""

        online_state = shared_online_state(max_age=self.max_age) or {}
        ready, lagging_core, degraded_core, degraded_online = minimal_online(online_state)
        optional_pending = (
            lagging_optional_components(online_state)
            if isinstance(online_state, Mapping)
            else set()
        )
        probe = ReadinessProbe(
            ready=ready,
            lagging_core=tuple(sorted(lagging_core)),
            degraded_core=tuple(sorted(degraded_core)),
            degraded_online=degraded_online,
            optional_pending=tuple(sorted(optional_pending)),
            heartbeat=online_state.get("heartbeat")
            if isinstance(online_state, Mapping)
            else None,
        )
        self._last_probe = probe
        return probe

    def is_ready(self) -> bool:
        """Return ``True`` when core bootstrap readiness has been achieved."""

        return self.probe().ready

    def await_ready(self, timeout: float | None = None) -> bool:
        """Block until readiness is achieved or ``timeout`` expires."""

        start = time.perf_counter()
        warning_emitted = False
        while True:
            probe = self.probe()
            if probe.ready:
                return True

            elapsed = time.perf_counter() - start
            if timeout is not None and not warning_emitted and elapsed >= timeout / 2.0:
                warning_emitted = True
                self._log_pending_components(
                    probe,
                    elapsed=elapsed,
                    timeout=timeout,
                    level=logging.WARNING,
                )

            if timeout is not None and (time.perf_counter() - start) >= timeout:
                self._log_pending_components(
                    probe,
                    elapsed=timeout,
                    timeout=timeout,
                    level=logging.WARNING,
                    force=True,
                )
                raise TimeoutError(
                    f"bootstrap readiness not satisfied after {timeout:.1f}s: {probe.summary()}"
                )

            time.sleep(self.poll_interval)

    def _log_pending_components(
        self,
        probe: ReadinessProbe,
        *,
        elapsed: float,
        timeout: float,
        level: int,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        throttle = max(self.poll_interval * 4.0, _READINESS_LOG_THROTTLE_SECONDS)
        if not force and self._last_pending_log and (now - self._last_pending_log) < throttle:
            return
        self._last_pending_log = now

        pending_components = set(probe.lagging_core) | set(probe.optional_pending)
        readiness_meta = log_component_readiness(
            logger=LOGGER,
            max_age=self.max_age,
            components=pending_components or None,
            level=level,
        )

        LOGGER.log(
            level,
            "bootstrap readiness pending after %.1fs (timeout %.1fs)",
            elapsed,
            timeout,
            extra={
                "event": "bootstrap-readiness-pending",
                "lagging_core": probe.lagging_core,
                "degraded_core": probe.degraded_core,
                "optional_pending": probe.optional_pending,
                "heartbeat_path": str(_heartbeat_path()),
                "component_readiness": readiness_meta,
            },
        )


_READINESS_SIGNAL = ReadinessSignal()
_ensure_bootstrap_keepalive()
atexit.register(stop_bootstrap_heartbeat_keepalive)


def readiness_signal() -> ReadinessSignal:
    """Return the shared readiness signal instance."""

    _ensure_bootstrap_keepalive()
    return _READINESS_SIGNAL


__all__ = [
    "CORE_COMPONENTS",
    "OPTIONAL_COMPONENTS",
    "READINESS_STAGES",
    "build_stage_deadlines",
    "lagging_optional_components",
    "minimal_online",
    "ReadinessProbe",
    "ReadinessSignal",
    "readiness_signal",
    "component_readiness_timestamps",
    "log_component_readiness",
    "start_bootstrap_heartbeat_keepalive",
    "stop_bootstrap_heartbeat_keepalive",
    "stage_for_step",
    "shared_online_state",
]
