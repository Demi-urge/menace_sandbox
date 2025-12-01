"""Automated environment bootstrap utilities.

The bootstrapper verifies system requirements, installs optional
dependencies and now exports secrets listed in ``BOOTSTRAP_SECRET_NAMES``
via :class:`SecretsManager`.
"""

from __future__ import annotations

import inspect
import logging
import os
import shlex
import subprocess
import shutil
import importlib.util
from pathlib import Path
from typing import Callable, Iterable, Mapping, TYPE_CHECKING
import threading
import json
import sys
import time
import urllib.error
import urllib.request

import bootstrap_timeout_policy
import bootstrap_metrics

from .bootstrap_readiness import READINESS_STAGES, build_stage_deadlines

from .config_discovery import ensure_config, ConfigDiscovery
from .bootstrap_policy import DependencyPolicy, PolicyLoader
from .logging_utils import log_record

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .cluster_supervisor import ClusterServiceSupervisor
from .infrastructure_bootstrap import InfrastructureBootstrapper
from .retry_utils import retry
from .system_provisioner import SystemProvisioner
from .secrets_manager import SecretsManager
from .vault_secret_provider import VaultSecretProvider
from .external_dependency_provisioner import ExternalDependencyProvisioner
from . import startup_checks
from .vector_service.embedding_scheduler import start_scheduler_from_env

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore


class _PhaseBudgetContext:
    def __init__(
        self,
        *,
        phase: str,
        budget: dict[str, float | None],
        clock: Callable[[], float],
        logger: logging.Logger,
        mark_online: Callable[[float | None, str, bool], None],
        enforce_deadline: bool = True,
        allow_partial_online: bool = False,
        soft_overrun_handler: Callable[[str, float, float | None], None] | None = None,
    ) -> None:
        self.phase = phase
        self.budget = budget
        self._clock = clock
        self._start = clock()
        self._logger = logger
        self._mark_online = mark_online
        self._warned_over_budget = False
        self._soft_overrun_emitted = False
        self.online_ready = False
        self.online_reason: str | None = None
        self._enforce_deadline = enforce_deadline
        self._soft_overrun_handler = soft_overrun_handler
        self._allow_partial_online = allow_partial_online

    def check(self) -> None:
        limit = self.budget.get("limit")
        soft_budget = self.budget.get("budget")
        elapsed = self.elapsed
        if soft_budget is not None and elapsed > soft_budget and not self._warned_over_budget:
            remaining = (limit - elapsed) if limit is not None else None
            self._warned_over_budget = True
            self._logger.warning(
                "phase %s exceeded soft budget (%.1fs > %.1fs); grace window %s",
                self.phase,
                elapsed,
                soft_budget,
                f"{remaining:.1f}s" if remaining is not None else "unbounded",
            )
        if (
            not self._enforce_deadline
            and self._soft_overrun_handler
            and not self._soft_overrun_emitted
            and (soft_budget is not None and elapsed > soft_budget)
        ):
            self._soft_overrun_emitted = True
            self._soft_overrun_handler(self.phase, elapsed, limit or soft_budget)
        if limit is not None and elapsed > limit:
            if not self._enforce_deadline:
                self._warned_over_budget = True
                if self._soft_overrun_handler and not self._soft_overrun_emitted:
                    self._soft_overrun_emitted = True
                    self._soft_overrun_handler(self.phase, elapsed, limit)
                return
            raise TimeoutError(
                f"{self.phase} phase exhausted soft budget ({elapsed:.1f}s > {limit:.1f}s)"
            )

    def mark_online_ready(self, *, reason: str = "subset-ready") -> None:
        if self.phase != "critical" and self.phase != "provisioning":
            return
        self.online_ready = True
        self.online_reason = reason
        self._mark_online(self.elapsed, reason, True if self._allow_partial_online else False)

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    def snapshot(self) -> dict[str, float | str | None]:
        return {
            "phase": self.phase,
            "budget": self.budget.get("budget"),
            "grace": self.budget.get("grace"),
            "limit": self.budget.get("limit"),
            "scale": self.budget.get("scale"),
            "elapsed": self.elapsed,
            "online_ready": self.online_ready,
            "online_reason": self.online_reason,
        }


class _BootstrapScheduler:
    def __init__(
        self,
        *,
        ready_events: Mapping[str, threading.Event],
        online_event: threading.Event,
        semaphore: threading.Semaphore,
        logger: logging.Logger,
        readiness_tracker: Callable[[str, str], None],
    ) -> None:
        self._ready_events = dict(ready_events)
        self._online_event = online_event
        self._semaphore = semaphore
        self._logger = logger
        self._readiness_tracker = readiness_tracker

    def mark_ready(self, phase: str) -> None:
        if phase in self._ready_events:
            self._ready_events[phase].set()
        if phase == "online":
            self._online_event.set()

    def schedule(
        self,
        name: str,
        work: Callable[[], threading.Thread | None],
        *,
        delay_until_ready: bool = True,
        join_inner: bool = True,
        ready_gate: str | None = "critical",
    ) -> threading.Thread:
        self._readiness_tracker("scheduled", name)

        def _runner() -> None:
            if delay_until_ready and ready_gate:
                event = self._ready_events.get(ready_gate)
                if event is None and ready_gate == "online":
                    event = self._online_event
                if event is None:
                    event = next(iter(self._ready_events.values())) if self._ready_events else None
                if event is not None:
                    event.wait()
            with self._semaphore:
                self._readiness_tracker("running", name)
                try:
                    inner = work()
                    if join_inner and isinstance(inner, threading.Thread):
                        inner.join()
                except Exception as exc:  # pragma: no cover - log only
                    self._logger.error("background task %s failed: %s", name, exc)
                finally:
                    self._readiness_tracker("finished", name)

        t = threading.Thread(target=_runner, daemon=True, name=f"bootstrap-{name}")
        t.start()
        return t


class _StageAwareScheduler:
    def __init__(
        self,
        *,
        stage_policy: Mapping[str, Mapping[str, object]] | None,
        component_budgets: Mapping[str, float] | None,
        scheduler: _BootstrapScheduler,
        online_gate: threading.Event,
        host_load_probe: Callable[[], float | None],
        load_threshold: float,
        logger: logging.Logger,
    ) -> None:
        self._stage_policy = stage_policy or {}
        self._component_budgets = component_budgets or {}
        self._scheduler = scheduler
        self._online_gate = online_gate
        self._host_load_probe = host_load_probe
        self._load_threshold = max(load_threshold, 0.0)
        self._logger = logger
        self._stage_order: tuple[str, ...] = tuple(stage.name for stage in READINESS_STAGES)
        self._optional: dict[str, bool] = {stage.name: stage.optional for stage in READINESS_STAGES}
        self._scheduled: set[str] = set()
        self._stage_events: dict[str, threading.Event] = {
            name: threading.Event() for name in self._stage_order
        }

    def _budget_hint(self, stage: str, budget_hint: float | None) -> float | None:
        if budget_hint is not None:
            return budget_hint
        meta = self._stage_policy.get(stage, {})
        for key in ("scaled_budget", "budget", "soft_budget", "deadline"):
            try:
                value = meta.get(key) if isinstance(meta, Mapping) else None
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        try:
            return float(self._component_budgets.get(stage))
        except Exception:
            return None

    def _previous_event(self, stage: str) -> threading.Event | None:
        if stage not in self._stage_order:
            return None
        idx = self._stage_order.index(stage)
        for prior in reversed(self._stage_order[:idx]):
            if prior in self._scheduled:
                return self._stage_events.get(prior)
        return None

    def _throttle_delay(self, stage: str, budget_hint: float | None) -> float:
        delay = 0.0
        load = self._host_load_probe()
        if load is not None and self._load_threshold:
            over = max(load / self._load_threshold, 0.0)
            if over > 1.0:
                delay = max(delay, min(over * 1.5, 30.0))
        if budget_hint is not None and budget_hint > 0:
            delay = max(delay, min(budget_hint * 0.05, 15.0))
        return delay

    def schedule_stage(
        self,
        stage: str,
        name: str,
        func: Callable[[], threading.Thread | None],
        *,
        delay_until_ready: bool,
        join_inner: bool,
        ready_gate: str | None,
        budget_hint: float | None = None,
    ) -> threading.Thread:
        if stage not in self._stage_order:
            return self._scheduler.schedule(
                name,
                func,
                delay_until_ready=delay_until_ready,
                join_inner=join_inner,
                ready_gate=ready_gate,
            )
        self._scheduled.add(stage)
        event = self._stage_events[stage]
        prior = self._previous_event(stage)

        def _runner() -> None:
            if prior:
                prior.wait()
            if self._optional.get(stage) and not self._online_gate.is_set():
                self._logger.info(
                    "delaying optional stage %s until core readiness stabilizes", stage
                )
                self._online_gate.wait()
            hint = self._budget_hint(stage, budget_hint)
            delay = self._throttle_delay(stage, hint)
            if delay > 0:
                self._logger.info(
                    "throttling stage %s for %.1fs (load/budget guard)", stage, delay
                )
                time.sleep(delay)
            try:
                inner = self._scheduler.schedule(
                    name,
                    func,
                    delay_until_ready=delay_until_ready,
                    join_inner=join_inner,
                    ready_gate=ready_gate,
                )
                if isinstance(inner, threading.Thread):
                    inner.join()
            finally:
                event.set()

        wrapper = threading.Thread(target=_runner, daemon=True, name=f"stage-{stage}-{name}")
        wrapper.start()
        return wrapper


class EnvironmentBootstrapper:
    """Bootstrap dependencies and infrastructure on startup."""

    PHASES: tuple[str, ...] = ("critical", "provisioning", "optional")

    def __init__(
        self,
        *,
        tf_dir: str | None = None,
        vault: VaultSecretProvider | None = None,
        cluster_supervisor: "ClusterServiceSupervisor" | None = None,
        policy: DependencyPolicy | None = None,
        shared_timeout_coordinator: bootstrap_timeout_policy.SharedTimeoutCoordinator
        | None = None,
    ) -> None:
        self._clock = time.monotonic
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_discovery = ConfigDiscovery()
        self.config_discovery.discover()
        self._background_threads: list[threading.Thread] = []
        self._stop_events: list[threading.Event] = []
        self._watchers_limited = os.getenv("MENACE_MINIMISE_BOOTSTRAP_WATCHERS") == "1"
        self.readiness_mode = os.getenv("MENACE_BOOTSTRAP_READINESS_MODE", "staged")
        self._phase_readiness: dict[str, bool | dict[str, object]] = {
            phase: False for phase in self.PHASES
        }
        self._phase_readiness["online_partial"] = False
        self._phase_readiness["online_degraded"] = False
        self._phase_readiness["online"] = False
        self._phase_readiness["full_ready"] = False
        self._phase_gates: dict[str, dict[str, object]] = {
            phase: {"status": "pending", "started": None, "finished": None}
            for phase in self.PHASES
        }
        self._background_state: dict[str, object] = {
            "scheduled": 0,
            "running": 0,
            "finished": 0,
            "active": set(),
        }
        self._stage_scheduler: _StageAwareScheduler | None = None
        self._phase_readiness["background_tasks"] = {
            "scheduled": 0,
            "running": 0,
            "finished": 0,
            "active": [],
        }
        semaphore_size = max(
            1,
            int(
                os.getenv(
                    "MENACE_BOOTSTRAP_BACKGROUND_PARALLELISM",
                    "1" if self._watchers_limited else "3",
                )
            ),
        )
        self._phase_events = {phase: threading.Event() for phase in self.PHASES}
        self._online_event = threading.Event()
        self._full_ready_event = threading.Event()
        self._phase_events["online"] = self._online_event
        self._phase_events["full_ready"] = self._full_ready_event
        self._background_semaphore = threading.Semaphore(semaphore_size)
        self._scheduler = _BootstrapScheduler(
            ready_events=self._phase_events,
            online_event=self._online_event,
            semaphore=self._background_semaphore,
            logger=self.logger,
            readiness_tracker=self._track_background_task,
        )
        threshold = os.getenv("MENACE_BOOTSTRAP_LOAD_THRESHOLD")
        try:
            self._load_threshold = float(threshold) if threshold is not None else None
        except ValueError:
            self._load_threshold = None
        self.shared_timeout_coordinator = shared_timeout_coordinator
        self._component_timeouts: dict[str, float] | None = None
        self._budget_snapshot: dict[str, object] = {}
        self._persisted_phase_durations: set[str] = set()
        interval = os.getenv("CONFIG_DISCOVERY_INTERVAL")
        if interval:
            try:
                sec = float(interval)
            except ValueError:
                sec = 0.0
            if sec > 0:
                self._queue_background_task(
                    "config-discovery",
                    lambda: self._start_config_discovery_watcher(sec),
                )
                if self._watchers_limited:
                    self.logger.info(
                        "config discovery watcher queued with semaphore throttling",
                    )
        elif self._watchers_limited:
            self.logger.info(
                "config discovery watcher disabled via MENACE_MINIMISE_BOOTSTRAP_WATCHERS",
            )
        self.tf_dir = tf_dir or os.getenv("TERRAFORM_DIR")
        self.bootstrapper = InfrastructureBootstrapper(self.tf_dir)
        self.policy = policy or PolicyLoader().resolve()
        self.required_commands = list(self.policy.required_commands)
        self.remote_endpoints = os.getenv("MENACE_REMOTE_ENDPOINTS", "").split(",")
        self.required_os_packages = os.getenv("MENACE_OS_PACKAGES", "").split(",")
        self.secrets = SecretsManager()
        self.vault = vault or (VaultSecretProvider() if os.getenv("SECRET_VAULT_URL") else None)
        self.secret_names = os.getenv("BOOTSTRAP_SECRET_NAMES", "").split(",")
        self.min_driver_version = os.getenv("MIN_NVIDIA_DRIVER_VERSION")
        self.strict_driver_check = os.getenv("STRICT_NVIDIA_DRIVER_CHECK") == "1"
        self.cluster_sup = cluster_supervisor
        if self.cluster_sup:
            hosts = [h.strip() for h in os.getenv("CLUSTER_HOSTS", "").split(",") if h.strip()]
            if hosts:
                self._queue_background_task(
                    "cluster-supervisor-hosts",
                    lambda: self._start_cluster_hosts(hosts),
                )

    def _run(self, cmd: list[str], **kwargs) -> None:
        """Run a subprocess command with retries and duration logging."""

        display = " ".join(shlex.quote(part) for part in cmd)

        @retry(Exception, attempts=3)
        def _execute() -> subprocess.CompletedProcess:
            return subprocess.run(cmd, check=True, **kwargs)

        start = self._clock()
        self.logger.info("executing command: %s", display)
        try:
            _execute()
        except Exception as exc:
            duration = self._clock() - start
            self.logger.error(
                "command failed after %.1fs (retried): %s", duration, display
            )
            raise
        duration = self._clock() - start
        level = logging.WARNING if duration >= 60 else logging.INFO
        self.logger.log(level, "command finished in %.1fs: %s", duration, display)

    def _check_phase_timeout(self, phase: str, start: float, timeout: float | None) -> None:
        if timeout is None:
            return
        elapsed = self._clock() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"environment bootstrap exceeded {phase} timeout ({timeout:.1f}s > {elapsed:.1f}s)"
            )

    def _emit_phase_event(self, phase: str, status: str, **fields: object) -> None:
        payload = log_record(
            event="bootstrap-readiness",
            phase=phase,
            status=status,
            mode=self.readiness_mode,
            readiness_tokens=dict(self._phase_readiness),
            readiness_gates=self._readiness_snapshot(),
            budgets=self._budget_snapshot,
            **fields,
        )
        self.logger.info("bootstrap phase %s %s", phase, status, extra=payload)
        heartbeat_background = {
            "scheduled": self._background_state.get("scheduled", 0),
            "running": self._background_state.get("running", 0),
            "finished": self._background_state.get("finished", 0),
            "active": sorted(self._background_state.get("active", ()) or ()),
        }
        bootstrap_timeout_policy.emit_bootstrap_heartbeat(
            {
                "event": "bootstrap-phase",
                "phase": phase,
                "status": status,
                "mode": self.readiness_mode,
                "readiness": dict(self._phase_readiness),
                "background": heartbeat_background,
                **fields,
            }
        )

    def _mark_online_ready(self, elapsed: float | None, reason: str, partial: bool) -> None:
        core_ready = self._core_gates_ready()
        critical_ready = self._critical_gate_ready()
        lagging = self._lagging_core_phases()
        degraded = bool(lagging)

        if partial and not self._phase_readiness.get("online_partial") and critical_ready:
            self._phase_readiness["online_partial"] = True
            self._emit_phase_event(
                "online_partial", "ready", elapsed=elapsed, reason=reason
            )

        if self._phase_readiness.get("online"):
            self._phase_readiness["online_degraded"] = degraded
            if not degraded:
                self._phase_readiness["online_partial"] = True
                self._emit_phase_event(
                    "online", "ready", elapsed=elapsed, reason=reason
                )
            return

        if not core_ready:
            self._phase_readiness["online_degraded"] = degraded
            return

        self._phase_readiness["online_partial"] = True
        self._phase_readiness["online_degraded"] = degraded
        self._phase_readiness["online"] = True
        self._scheduler.mark_ready("online")
        event_status = "degraded" if degraded else "ready"
        self._emit_phase_event(
            "online",
            event_status,
            elapsed=elapsed,
            reason=reason,
            lagging_core=sorted(lagging),
        )

    def _mark_full_ready(self, *, reason: str | None = None) -> None:
        if self._phase_readiness.get("full_ready"):
            return
        if not all(self._phase_readiness.get(phase) for phase in self.PHASES):
            return
        if int(self._background_state.get("running", 0)) > 0:
            return
        self._phase_readiness["full_ready"] = True
        self._scheduler.mark_ready("full_ready")
        self._emit_phase_event("full_ready", "ready", reason=reason)

    def _readiness_snapshot(self) -> dict[str, dict[str, object]]:
        return {phase: dict(details) for phase, details in self._phase_gates.items()}

    def _persist_phase_duration(self, phase: str, elapsed: float | None) -> None:
        if elapsed is None or elapsed <= 0 or phase in self._persisted_phase_durations:
            return

        store = bootstrap_metrics.record_durations(
            durations={phase: float(elapsed)},
            category="bootstrap_phases",
            logger=self.logger,
        )
        stats = bootstrap_metrics.compute_stats(store.get("bootstrap_phases", {}))
        self._budget_snapshot.setdefault("phase_stats", stats)
        self.logger.info(
            "persisted bootstrap phase duration",
            extra=log_record(
                event="bootstrap-phase-duration-recorded",
                phase=phase,
                elapsed=round(elapsed, 2),
                stats={
                    key: {k: round(v, 2) for k, v in value.items()}
                    for key, value in stats.items()
                },
                store=str(bootstrap_metrics.BOOTSTRAP_DURATION_STORE),
            ),
        )
        self._persisted_phase_durations.add(phase)

    def _persist_phase_durations(self) -> None:
        for phase, gate in self._phase_gates.items():
            try:
                started = float(gate.get("started", 0.0) or 0.0)
                finished = float(gate.get("finished", 0.0) or 0.0)
            except Exception:
                continue
            elapsed = finished - started
            self._persist_phase_duration(phase, elapsed)

    def _update_gate(self, phase: str, status: str, **fields: object) -> None:
        gate = self._phase_gates.setdefault(
            phase, {"status": "pending", "started": None, "finished": None}
        )
        now = self._clock()
        if status == "start":
            gate["started"] = now
        if status in {"ready", "failed", "degraded"}:
            gate["finished"] = now
        gate.update({"status": status, **fields})

    def _critical_gate_ready(self) -> bool:
        return self._phase_gates.get("critical", {}).get("status") in {
            "ready",
            "degraded",
        }

    def _core_gates_ready(self) -> bool:
        return all(
            self._phase_gates.get(name, {}).get("status") in {"ready", "degraded"}
            for name in ("critical", "provisioning")
        )

    def _lagging_core_phases(self) -> set[str]:
        lagging: set[str] = set()
        for name in ("critical", "provisioning"):
            if self._phase_gates.get(name, {}).get("status") not in {"ready", "degraded"}:
                lagging.add(name)
        return lagging

    def _phase_elapsed(self, phase: str) -> float | None:
        try:
            gate = self._phase_gates.get(phase, {})
            started = float(gate.get("started") or 0.0)
            finished = float(gate.get("finished") or 0.0)
            if started <= 0.0:
                return None
            if finished <= 0.0:
                finished = self._clock()
            return max(finished - started, 0.0)
        except Exception:
            return None

    def _maybe_mark_online(self, *, reason: str, partial: bool = False) -> None:
        critical_elapsed = self._phase_elapsed("critical")
        if partial:
            self._mark_online_ready(critical_elapsed, reason, True)
        if self._phase_readiness.get("online"):
            return
        if not self._core_gates_ready():
            return
        core_elapsed = None
        try:
            core_elapsed = max(
                (self._phase_gates.get(name, {}).get("finished") or 0.0)
                - (self._phase_gates.get(name, {}).get("started") or 0.0)
                for name in ("critical", "provisioning")
            )
        except Exception:
            core_elapsed = None
        self._mark_online_ready(core_elapsed, reason, True)

    def _run_phase(
        self,
        phase: str,
        budget: float | None | dict[str, float | None],
        work: Callable[..., None],
        *,
        strict: bool = True,
    ) -> bool:
        start = self._clock()
        normalized_budget: dict[str, float | None]
        if isinstance(budget, dict):
            normalized_budget = dict(budget)
        else:
            if budget is None:
                normalized_budget = {"budget": None, "grace": None, "limit": None, "scale": 1.0}
            else:
                grace = max(30.0, budget * 0.25)
                normalized_budget = {
                    "budget": budget,
                    "grace": grace,
                    "limit": budget + grace,
                    "scale": 1.0,
                }

        def _parse_float(value: object) -> float | None:
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        guard_context = (
            self._budget_snapshot.get("bootstrap_guard", {})
            if isinstance(self._budget_snapshot, dict)
            else {}
        )
        host_context = (
            self._budget_snapshot.get("cluster_load", {})
            if isinstance(self._budget_snapshot, dict)
            else {}
        )
        host_load = None
        if isinstance(host_context, Mapping):
            host_load = _parse_float(
                host_context.get("host_load")
                or host_context.get("load")
                or host_context.get("load_avg")
            )
        guard_budget_scale = _parse_float(
            guard_context.get("budget_scale") or guard_context.get("scale") or 1.0
        ) or 1.0
        guard_delay = _parse_float(
            guard_context.get("delay") or guard_context.get("guard_delay")
        )
        load_threshold = _parse_float(os.getenv("MENACE_BOOTSTRAP_LOAD_THRESHOLD"))
        if load_threshold is None:
            load_threshold = getattr(bootstrap_timeout_policy, "_DEFAULT_LOAD_THRESHOLD", 1.35)
        load_scale = 1.0
        if host_load is not None and host_load > 0 and load_threshold:
            load_scale = max(1.0, host_load / load_threshold)
        contention_scale = max(guard_budget_scale, load_scale)
        contention_mode = contention_scale > 1.0 or (guard_delay or 0.0) > 0.0

        def _scaled(value: float | None) -> float | None:
            if value is None:
                return None
            return value * contention_scale

        if contention_mode:
            normalized_budget["budget"] = _scaled(normalized_budget.get("budget"))
            normalized_budget["grace"] = _scaled(normalized_budget.get("grace"))
            limit = normalized_budget.get("limit")
            if limit is None:
                budget_value = normalized_budget.get("budget")
                grace_value = normalized_budget.get("grace")
                if budget_value is not None or grace_value is not None:
                    limit = (budget_value or 0.0) + (grace_value or 0.0)
            normalized_budget["limit"] = _scaled(limit)
            normalized_budget["scale"] = max(
                _parse_float(normalized_budget.get("scale")) or 1.0, contention_scale
            )
            normalized_budget["contention_mode"] = True
            normalized_budget["contention_scale"] = contention_scale
            normalized_budget["contention_load"] = host_load
            normalized_budget["contention_threshold"] = load_threshold
            if guard_delay is not None:
                normalized_budget["guard_delay"] = guard_delay

        soft_overrun = False

        def _mark_soft_overrun(name: str, elapsed: float, limit: float | None) -> None:
            nonlocal soft_overrun
            soft_overrun = True
            self.logger.warning(
                "phase %s exceeded enforced deadline but continuing in degraded mode", name
            )
            self._emit_phase_event(
                name,
                "degraded",
                elapsed=elapsed,
                limit=limit,
                overrun=True,
            )
            self._update_gate(name, "degraded", elapsed=elapsed, limit=limit, overrun=True)

        enforce_deadline = phase == "critical" and not contention_mode
        phase_budget = _PhaseBudgetContext(
            phase=phase,
            budget=normalized_budget,
            clock=self._clock,
            logger=self.logger,
            mark_online=self._mark_online_ready,
            enforce_deadline=enforce_deadline,
            allow_partial_online=phase in {"critical", "provisioning"},
            soft_overrun_handler=None if enforce_deadline else _mark_soft_overrun,
        )

        self._emit_phase_event(
            phase,
            "start",
            budget=normalized_budget.get("budget"),
            grace=normalized_budget.get("grace"),
            scale=normalized_budget.get("scale"),
            limit=normalized_budget.get("limit"),
            guard_delay=guard_delay,
            contention_scale=contention_scale if contention_mode else None,
            contention_load=host_load,
        )
        self._update_gate(
            phase,
            "start",
            budget=normalized_budget.get("budget"),
            grace=normalized_budget.get("grace"),
            scale=normalized_budget.get("scale"),
            limit=normalized_budget.get("limit"),
            guard_delay=guard_delay,
            contention_scale=contention_scale if contention_mode else None,
            contention_load=host_load,
        )

        def guard() -> None:
            try:
                phase_budget.check()
            finally:
                if enforce_deadline:
                    self._check_phase_timeout(phase, start, normalized_budget.get("limit"))

        try:
            try:
                work(guard, phase_budget)
            except TypeError as exc:
                sig = inspect.signature(work)
                if len(sig.parameters) > 1:
                    raise
                work(guard)
        except Exception as exc:
            self._emit_phase_event(phase, "failed", error=str(exc))
            self._update_gate(phase, "failed", error=str(exc))
            if strict:
                raise
            return False
        elapsed = phase_budget.elapsed
        self._phase_readiness[phase] = True
        gate_status = "ready"
        if soft_overrun and phase != "critical":
            gate_status = "degraded"
        self._update_gate(
            phase,
            gate_status,
            elapsed=elapsed,
            overrun=soft_overrun,
            online_ready=phase_budget.online_ready,
            online_reason=phase_budget.online_reason,
            limit=normalized_budget.get("limit"),
            guard_delay=guard_delay,
            contention_scale=contention_scale if contention_mode else None,
            contention_load=host_load,
        )
        if phase in self.PHASES:
            self._scheduler.mark_ready(phase)
        self._emit_phase_event(
            phase,
            gate_status,
            elapsed=elapsed,
            limit=normalized_budget.get("limit"),
            guard_delay=guard_delay,
            contention_scale=contention_scale if contention_mode else None,
            contention_load=host_load,
        )
        self._maybe_mark_online(
            reason=phase_budget.online_reason or "core-phases-complete",
            partial=True,
        )
        self._persist_phase_duration(phase, elapsed)
        self._mark_full_ready(reason=f"{phase}-complete")
        return True

    def _track_background_task(self, action: str, name: str) -> None:
        scheduled = int(self._background_state.get("scheduled", 0))
        running = int(self._background_state.get("running", 0))
        finished = int(self._background_state.get("finished", 0))
        active: set[str] = set(self._background_state.get("active", set()))
        if action == "scheduled":
            scheduled += 1
        elif action == "running":
            running += 1
            active.add(name)
        elif action == "finished":
            running = max(running - 1, 0)
            finished += 1
            active.discard(name)
        self._background_state.update(
            {
                "scheduled": scheduled,
                "running": running,
                "finished": finished,
                "active": active,
            }
        )
        self._phase_readiness["background_tasks"] = {
            "scheduled": scheduled,
            "running": running,
            "finished": finished,
            "active": sorted(active),
        }
        bootstrap_timeout_policy.emit_bootstrap_heartbeat(
            {
                "event": "bootstrap-background",
                "active": sorted(active),
                "running": running,
                "finished": finished,
                "scheduled": scheduled,
                "online": bool(self._phase_readiness.get("online")),
            }
        )
        self._mark_full_ready(reason="background-complete")

    def _queue_background_task(
        self,
        name: str,
        func: Callable[[], threading.Thread | None],
        *,
        delay_until_ready: bool = True,
        join_inner: bool = True,
        ready_gate: str | None = "critical",
        stage: str | None = None,
        budget_hint: float | None = None,
    ) -> threading.Thread:
        if stage and self._stage_scheduler:
            t = self._stage_scheduler.schedule_stage(
                stage,
                name,
                func,
                delay_until_ready=delay_until_ready,
                join_inner=join_inner,
                ready_gate=ready_gate,
                budget_hint=budget_hint,
            )
        else:
            t = self._scheduler.schedule(
                name,
                func,
                delay_until_ready=delay_until_ready,
                join_inner=join_inner,
                ready_gate=ready_gate,
            )
        self._background_threads.append(t)
        return t

    def _start_config_discovery_watcher(self, interval: float) -> threading.Thread:
        stop_event = threading.Event()
        thread = self.config_discovery.run_continuous(
            interval=interval, stop_event=stop_event
        )
        self._stop_events.append(stop_event)
        self._background_threads.append(thread)
        return thread

    def _start_cluster_hosts(self, hosts: list[str]) -> None:
        try:
            self.cluster_sup.add_hosts(hosts)
            self.cluster_sup.start_all()
        except Exception as exc:  # pragma: no cover - remote may fail
            self.logger.error("cluster supervisor failed: %s", exc)

    def _resolve_phase_budgets(
        self, timeout: float | None
    ) -> dict[str, dict[str, float | None]]:
        duration_store = bootstrap_metrics.load_duration_store()
        phase_stats = bootstrap_metrics.compute_stats(duration_store.get("bootstrap_phases", {}))
        component_stats = bootstrap_metrics.compute_stats(
            duration_store.get("bootstrap_components", {})
        )
        guard_env = os.getenv("MENACE_BOOTSTRAP_GUARD_OPTOUT")
        guard_enabled = str(guard_env or "").lower() not in ("1", "true", "yes")
        guard_delay = 0.0
        guard_scale = 1.0
        if guard_enabled:
            guard_delay, guard_scale = bootstrap_timeout_policy.wait_for_bootstrap_quiet_period(
                self.logger
            )
        else:
            bootstrap_timeout_policy._record_bootstrap_guard(  # type: ignore[attr-defined]
                guard_delay, guard_scale, source="bootstrap_guard_optout"
            )
        enforcement = bootstrap_timeout_policy.enforce_bootstrap_timeout_policy(
            logger=self.logger
        )
        floors = bootstrap_timeout_policy.load_escalated_timeout_floors()
        base_wait_floor = float(floors.get("MENACE_BOOTSTRAP_WAIT_SECS", 0.0) or 0.0)
        persisted_wait = bootstrap_timeout_policy.load_persisted_bootstrap_wait()
        if persisted_wait is not None:
            try:
                base_wait_floor = max(base_wait_floor, float(persisted_wait))
            except (TypeError, ValueError):
                pass
        phase_floors = {
            phase: max(base_wait_floor, 0.0) * guard_scale for phase in self.PHASES
        }
        component_floors = {
            key: value * guard_scale
            for key, value in enforcement.get("component_floors", {}).items()
        }
        calibrated_component_floors, component_calibration = bootstrap_metrics.calibrate_step_budgets(
            base_budgets=component_floors,
            stats=component_stats,
            floors=component_floors,
        )
        component_floors = calibrated_component_floors

        budgets = {phase: None for phase in self.PHASES}
        budgets = self.policy.resolved_phase_timeouts(budgets)

        def _scale_budget(value: float | None) -> float | None:
            if value is None or guard_scale == 1.0:
                return value
            return value * guard_scale

        budgets = {phase: _scale_budget(value) for phase, value in budgets.items()}

        base_phase_budgets: dict[str, float] = {}
        for phase in self.PHASES:
            provided = budgets.get(phase)
            floor = phase_floors.get(phase, base_wait_floor)
            base = provided if provided is not None else floor
            base_phase_budgets[phase] = max(base, floor)

        calibrated_phase_budgets, phase_calibration = bootstrap_metrics.calibrate_step_budgets(
            base_budgets=base_phase_budgets,
            stats=phase_stats,
            floors=phase_floors,
        )
        budgets.update(calibrated_phase_budgets)

        host_telemetry = bootstrap_timeout_policy.read_bootstrap_heartbeat()
        telemetry = bootstrap_timeout_policy.collect_timeout_telemetry()
        component_timeouts = bootstrap_timeout_policy.compute_prepare_pipeline_component_budgets(
            component_floors=component_floors,
            telemetry=telemetry,
            host_telemetry=host_telemetry,
        )
        component_budget_total = float(sum(component_timeouts.values())) if component_timeouts else 0.0
        active_component_count = len(
            [gate for gate, budget in component_timeouts.items() if gate not in bootstrap_timeout_policy.DEFERRED_COMPONENTS]
        )
        persisted_window, persisted_window_inputs = (
            bootstrap_timeout_policy.load_last_global_bootstrap_window()
        )

        def _component_remaining_budget() -> float:
            windows = telemetry.get("component_windows", {}) if isinstance(telemetry, dict) else {}
            remaining = 0.0
            for meta in windows.values() if isinstance(windows, dict) else ():
                try:
                    remaining += max(float(meta.get("remaining", 0.0) or 0.0), 0.0)
                except Exception:
                    continue
            return remaining

        component_windows = (
            telemetry.get("component_windows", {}) if isinstance(telemetry, dict) else {}
        )
        component_remaining = _component_remaining_budget()
        progress_signal = (
            telemetry.get("progress_signal", {}) if isinstance(telemetry, dict) else {}
        )
        progressing_components = {
            key
            for key, meta in (component_windows or {}).items()
            if isinstance(meta, dict)
            and (meta.get("remaining", 0.0) or 0.0) > 0
            and (progress_signal.get("progressing") or progress_signal.get("recent_heartbeats"))
        }
        heartbeat_hint = dict(host_telemetry or {})
        if component_remaining:
            heartbeat_hint["remaining_budget"] = component_remaining
        if persisted_window is not None and heartbeat_hint.get("global_window") is None:
            heartbeat_hint["global_window"] = persisted_window
        if progress_signal:
            heartbeat_hint.setdefault("progressing", bool(progress_signal.get("progressing")))

        rolling_base = max(
            [
                component_budget_total,
                max(base_phase_budgets.values() or [0.0]),
                base_wait_floor,
                persisted_window or 0.0,
            ]
        )
        rolling_window, rolling_meta = bootstrap_timeout_policy._derive_rolling_global_window(  # type: ignore[attr-defined]
            base_window=rolling_base,
            component_budget_total=component_budget_total,
            host_telemetry=heartbeat_hint,
        )
        if rolling_window is not None:
            for phase in self.PHASES:
                floor = phase_floors.get(phase, base_wait_floor)
                phase_floors[phase] = max(floor, rolling_window)
                current = budgets.get(phase)
                if current is None or current < rolling_window:
                    budgets[phase] = rolling_window
        rolling_meta = {
            **(rolling_meta or {}),
            "component_budget_total": component_budget_total,
            "component_remaining": component_remaining,
            "progress_signal": progress_signal,
            "progressing_components": sorted(progressing_components),
            "persisted_window": persisted_window,
            "persisted_window_inputs": persisted_window_inputs,
            "heartbeat_hint": heartbeat_hint,
        }

        def parse_env(name: str) -> float | None:
            raw = os.getenv(name)
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                self.logger.warning("invalid timeout override for %s: %r", name, raw)
                return None

        overrides = {
            "critical": parse_env("MENACE_BOOTSTRAP_CRITICAL_BUDGET"),
            "provisioning": parse_env("MENACE_BOOTSTRAP_PROVISIONING_BUDGET"),
            "optional": parse_env("MENACE_BOOTSTRAP_OPTIONAL_BUDGET"),
        }
        for phase, override in overrides.items():
            if override is None:
                continue
            scaled_override = _scale_budget(override)
            floor = phase_floors.get(phase, base_wait_floor)
            effective_override = max(
                floor, scaled_override if scaled_override is not None else override
            )
            if effective_override != override:
                self.logger.warning(
                    "%s override below persisted floor; clamping", phase,
                    extra={
                        "requested_budget": override,
                        "effective_budget": effective_override,
                        "floor": floor,
                    },
                )
            os.environ[
                f"MENACE_BOOTSTRAP_{phase.upper()}_BUDGET"
            ] = str(effective_override)
            budgets[phase] = effective_override

        shared_snapshot: dict[str, object] | None = None
        coordinator = self.shared_timeout_coordinator
        if coordinator is not None:
            shared_snapshot = coordinator.snapshot()
        if timeout is not None and coordinator is None:
            coordinator = bootstrap_timeout_policy.SharedTimeoutCoordinator(
                timeout,
                logger=self.logger,
                namespace="bootstrap_phases",
                component_floors=component_floors,
                component_budgets=component_timeouts,
            )
            phase_minimums = {
                "critical": phase_floors.get("critical", base_wait_floor),
                "provisioning": phase_floors.get("provisioning", base_wait_floor),
                "optional": max(60.0 * guard_scale, phase_floors.get("optional", base_wait_floor)),
            }
            for phase in self.PHASES:
                requested = budgets.get(phase)
                minimum = phase_minimums.get(phase, base_wait_floor)
                effective, _ = coordinator._reserve(  # type: ignore[attr-defined]
                    phase,
                    requested,
                    minimum,
                    {"source": "global_timeout"},
                )
                budgets[phase] = effective
            shared_snapshot = coordinator.snapshot()

        soft_budgets = bootstrap_timeout_policy.derive_phase_soft_budgets(
            budgets, telemetry=telemetry
        )

        if coordinator is not None and component_timeouts:
            coordinator.component_budgets.update(component_timeouts)
            shared_snapshot = coordinator.snapshot()

        def _max_budget_value(values: Mapping[str, float | None]) -> float | None:
            finite = [v for v in values.values() if isinstance(v, (int, float))]
            return max(finite) if finite else None

        adaptive_window = rolling_window
        if adaptive_window is None:
            adaptive_window = _max_budget_value(budgets)

        progress_detected = bool(
            progress_signal.get("progressing") or progress_signal.get("recent_heartbeats")
        )
        progress_extension = component_remaining if progress_detected else 0.0
        if progress_extension > 0:
            baseline_window = adaptive_window if adaptive_window is not None else _max_budget_value(budgets)
            if baseline_window is None:
                baseline_window = base_wait_floor
            extended_window = (baseline_window or 0.0) + progress_extension
            previous_window = adaptive_window
            adaptive_window = max(adaptive_window or 0.0, extended_window)

            for phase in self.PHASES:
                current = budgets.get(phase)
                budgets[phase] = max(current or adaptive_window, adaptive_window)

            soft_budgets = bootstrap_timeout_policy.derive_phase_soft_budgets(
                budgets, telemetry=telemetry
            )

            if coordinator is not None:
                for label, meta in (component_windows or {}).items():
                    if not isinstance(meta, Mapping):
                        continue
                    try:
                        budget_hint = float(meta.get("budget") or meta.get("remaining") or 0.0)
                    except (TypeError, ValueError):
                        budget_hint = 0.0
                    if budget_hint > 0:
                        coordinator._register_component_window(  # type: ignore[attr-defined]
                            label, budget_hint
                        )
                with coordinator._lock:  # type: ignore[attr-defined]
                    coordinator._expanded_global_window = max(  # type: ignore[attr-defined]
                        coordinator._expanded_global_window or 0.0, adaptive_window
                    )
                    coordinator.global_window = max(
                        coordinator.global_window or 0.0, adaptive_window
                    )
                shared_snapshot = coordinator.snapshot()

            self.logger.info(
                "bootstrap window extended from progress telemetry",
                extra=log_record(
                    event="bootstrap-window-extended",
                    baseline_window=baseline_window,
                    previous_window=previous_window,
                    extended_window=adaptive_window,
                    progress_extension=progress_extension,
                    component_remaining=component_remaining,
                    progress_signal=progress_signal,
                ),
            )

        elastic_window, elastic_meta = bootstrap_timeout_policy.derive_elastic_global_window(
            base_window=adaptive_window,
            component_budgets=component_timeouts,
            backlog_queue_depth=int(heartbeat_hint.get("queue_depth") or 0),
            active_components=active_component_count,
            component_stats=component_stats,
        )
        if elastic_window is not None:
            adaptive_window = max(adaptive_window or 0.0, elastic_window)
            rolling_meta = {**(rolling_meta or {}), "elastic_window": elastic_meta}

        stage_windows = bootstrap_timeout_policy.load_adaptive_stage_windows(
            component_budgets=component_timeouts
        )
        stage_deadlines = build_stage_deadlines(
            max(base_phase_budgets.values() or [base_wait_floor, 0.0]),
            heavy_detected=component_budget_total > 0,
            soft_deadline=bool(enforcement.get("soft_deadline")),
            component_budgets=component_timeouts,
            component_floors=component_floors,
            adaptive_window=adaptive_window,
            stage_windows=stage_windows,
            stage_runtime=bootstrap_timeout_policy.load_component_runtime_samples(),
        )

        bootstrap_timeout_policy.persist_bootstrap_wait_window(
            adaptive_window,
            vector_heavy=False,
            source="environment_bootstrap",
            metadata={
                "component_budget_total": component_budget_total,
                "component_budgets": component_timeouts,
                "phase_budgets": budgets,
                "rolling_meta": rolling_meta,
                "shared_timeout": shared_snapshot,
            },
        )

        self.logger.info(
            "adaptive bootstrap window resolved",
            extra=log_record(
                event="bootstrap-window",
                global_window=adaptive_window,
                component_budget_total=component_budget_total,
                component_budgets=component_timeouts,
                progress_signal=progress_signal,
                component_windows=component_windows,
                rolling_meta=rolling_meta,
            ),
        )

        guard_context = bootstrap_timeout_policy.get_bootstrap_guard_context()

        self._budget_snapshot = {
            "phase_budgets": budgets,
            "soft_budgets": soft_budgets,
            "timeout_enforcement": enforcement,
            "shared_timeout": shared_snapshot,
            "component_timeouts": component_timeouts,
            "cluster_load": host_telemetry,
            "global_window": adaptive_window,
            "global_window_meta": rolling_meta,
            "stage_deadlines": stage_deadlines,
            "bootstrap_guard": guard_context
            | {"enabled": guard_enabled, "delay": guard_delay, "scale": guard_scale},
            "phase_stats": phase_stats,
            "phase_calibration": phase_calibration,
            "component_stats": component_stats,
            "component_calibration": component_calibration,
        }

        self.logger.info(
            "resolved bootstrap phase budgets",
            extra=dict(self._budget_snapshot),
        )

        self._configure_stage_scheduler(stage_deadlines, component_timeouts)

        self.shared_timeout_coordinator = coordinator
        self._component_timeouts = component_timeouts

        return soft_budgets

    def readiness_state(self) -> dict[str, object]:
        """Return a snapshot of bootstrap readiness gates."""

        snapshot = dict(self._phase_readiness)
        snapshot["gates"] = self._readiness_snapshot()
        snapshot["background_tasks"] = dict(self._phase_readiness.get("background_tasks", {}))
        return snapshot

    def _current_host_load(self) -> float | None:
        ctx = self._budget_snapshot.get("cluster_load") if isinstance(self._budget_snapshot, dict) else None
        if isinstance(ctx, Mapping):
            for key in ("host_load", "load", "load_avg"):
                try:
                    value = ctx.get(key)
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError, AttributeError):
                    continue
        try:
            load_avg = os.getloadavg()
            return float(load_avg[0]) if load_avg else None
        except (OSError, AttributeError):
            return None

    def _configure_stage_scheduler(
        self,
        stage_policy: Mapping[str, Mapping[str, object]] | None,
        component_budgets: Mapping[str, float] | None,
    ) -> None:
        load_threshold = self._load_threshold
        if load_threshold is None:
            load_threshold = getattr(bootstrap_timeout_policy, "_DEFAULT_LOAD_THRESHOLD", 1.35)

        def _probe() -> float | None:
            return self._current_host_load()

        self._stage_scheduler = _StageAwareScheduler(
            stage_policy=stage_policy,
            component_budgets=component_budgets,
            scheduler=self._scheduler,
            online_gate=self._online_event,
            host_load_probe=_probe,
            load_threshold=load_threshold or 1.35,
            logger=self.logger,
        )

    def _stage_budget_hint(self, stage: str) -> float | None:
        stage_policy = self._budget_snapshot.get("stage_deadlines")
        if isinstance(stage_policy, Mapping):
            entry = stage_policy.get(stage, {})
            for key in ("scaled_budget", "soft_budget", "budget", "deadline"):
                try:
                    value = entry.get(key) if isinstance(entry, Mapping) else None
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    continue
        if self._component_timeouts:
            try:
                return float(self._component_timeouts.get(stage))
            except Exception:
                return None
        return None

    def shutdown(self, *, timeout: float | None = 5.0) -> None:
        """Stop background bootstrap threads started by the instance."""

        for event in self._stop_events:
            event.set()
        for thread in self._background_threads:
            if thread.is_alive():
                thread.join(timeout)

    # ------------------------------------------------------------------
    def check_commands(self, cmds: Iterable[str]) -> None:
        """Verify that required commands are available."""
        for cmd in cmds:
            if shutil.which(cmd) is None:
                self.logger.warning("required command missing: %s", cmd)

    # ------------------------------------------------------------------
    def check_nvidia_driver(self) -> None:
        """Ensure the installed NVIDIA driver meets the minimum version."""
        if not self.min_driver_version:
            return
        if shutil.which("nvidia-smi") is None:
            self.logger.warning("nvidia-smi not found; skipping driver check")
            return
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            ver_str = out.splitlines()[0].strip()
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("failed to query nvidia driver: %s", exc)
            return

        try:
            from packaging import version as packaging_version

            current = packaging_version.parse(ver_str)
            minimum = packaging_version.parse(self.min_driver_version)
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("driver version parse failed: %s", exc)
            return

        if current < minimum:
            msg = f"NVIDIA driver {current} < required {minimum}"
            if self.strict_driver_check:
                raise RuntimeError(msg)
            self.logger.warning(msg)

    # ------------------------------------------------------------------
    def check_remote_dependencies(self, urls: Iterable[str]) -> None:
        """Ensure remote services are reachable."""
        if not self.policy.enforce_remote_checks:
            self.logger.debug("remote dependency checks disabled by policy")
            return
        missing = False
        for url in urls:
            u = url.strip()
            if not u:
                continue
            if shutil.which("curl"):
                try:
                    subprocess.run(
                        ["curl", "-I", "--max-time", "5", u],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                    continue
                except Exception as exc:
                    self.logger.debug(
                        "curl probe failed for %s: %s; falling back to urllib", u, exc
                    )
            try:
                request = urllib.request.Request(u, method="HEAD")
                with urllib.request.urlopen(request, timeout=5):
                    pass
            except urllib.error.HTTPError as exc:
                status = getattr(exc, "code", 0)
                message = f"remote dependency {u} responded with HTTP {status}"
                if status >= 500:
                    self.logger.error(message)
                    missing = True
                else:
                    self.logger.warning(message)
            except Exception as exc:
                self.logger.error(
                    "remote dependency unreachable: %s - %s", u, exc
                )
                missing = True
        if missing:
            ExternalDependencyProvisioner().provision()

    # ------------------------------------------------------------------
    def check_os_packages(self, packages: Iterable[str]) -> None:
        """Verify that required system packages are installed."""
        if not packages or not self.policy.enforce_os_package_checks:
            if packages and not self.policy.enforce_os_package_checks:
                self.logger.debug(
                    "policy '%s' disabled OS package checks", self.policy.name
                )
            return
        probe = self._resolve_package_probe()
        if probe is None:
            return

        missing: list[str] = []
        for pkg in packages:
            p = pkg.strip()
            if not p:
                continue
            try:
                if probe[0] in {"winget", "choco"}:
                    if not self._check_windows_package(probe, p):
                        raise RuntimeError("missing")
                else:
                    subprocess.run(
                        list(probe) + [p],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
            except Exception:
                self.logger.error("required package missing: %s", p)
                missing.append(p)
        if missing:
            raise RuntimeError("missing OS packages: " + ", ".join(missing))

    # ------------------------------------------------------------------
    def _resolve_package_probe(self) -> tuple[str, ...] | None:
        """Return command prefix used to check OS packages."""

        if os.name == "nt":
            for candidate in self.policy.windows_package_managers:
                if shutil.which(candidate):
                    if candidate == "winget":
                        return ("winget", "list", "--exact", "--id")
                    if candidate == "choco":
                        return ("choco", "list", "--local-only")
            self.logger.info(
                "no supported Windows package manager available; skipping OS package verification"
            )
            return None

        for candidate in self.policy.linux_package_managers:
            if candidate == "dpkg" and shutil.which("dpkg"):
                return ("dpkg", "-s")
            if candidate == "rpm" and shutil.which("rpm"):
                return ("rpm", "-q")
        self.logger.info(
            "no supported package manager found to verify packages"
        )
        return None

    # ------------------------------------------------------------------
    def _check_windows_package(self, probe: tuple[str, ...], package: str) -> bool:
        """Return ``True`` if ``package`` is installed on Windows hosts."""

        if probe[0] == "winget":
            cmd = list(probe) + [package]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return False
            output = (result.stdout or "") + (result.stderr or "")
            return package.lower() in output.lower()
        if probe[0] == "choco":
            cmd = list(probe) + [package]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return False
            output = (result.stdout or "") + (result.stderr or "")
            return package.lower() in output.lower()
        raise ValueError(f"Unsupported Windows package probe: {probe}")

    # ------------------------------------------------------------------
    def export_secrets(self) -> None:
        """Expose configured secrets as environment variables."""
        for name in self.secret_names:
            n = name.strip()
            if not n:
                continue
            try:
                if self.vault:
                    self.vault.export_env(n)
                else:
                    self.secrets.export_env(n)
            except Exception as exc:  # pragma: no cover - log only
                self.logger.error("export secret %s failed: %s", n, exc)

    # ------------------------------------------------------------------
    def deploy_across_hosts(self, hosts: Iterable[str]) -> None:
        """Run the bootstrap script on remote hosts via SSH."""
        for host in hosts:
            h = host.strip()
            if not h:
                continue
            try:
                self.logger.info("remote bootstrap on %s", h)
                subprocess.run(
                    ["ssh", h, "python3", "-m", "menace.environment_bootstrap"],
                    check=True,
                )
            except Exception as exc:
                self.logger.error("remote bootstrap failed for %s: %s", h, exc)

    # ------------------------------------------------------------------
    def bootstrap_vector_assets(self) -> None:
        """Download model and seed default ranking weights."""
        if importlib.util.find_spec("huggingface_hub") is None:
            self.logger.info(
                "Skipping embedding model download; install huggingface-hub to enable automatic provisioning"
            )
        else:
            try:
                from .vector_service import download_model as _dm

                dest = resolve_path(
                    "vector_service/minilm/tiny-distilroberta-base.tar.xz"
                )
                if not dest.exists():
                    _dm.bundle(dest)
            except Exception as exc:  # pragma: no cover - log only
                self.logger.warning("embedding model download failed: %s", exc)

        try:
            reg_path = resolve_path(
                "vector_service/embedding_registry.json"
            )
            with open(reg_path, "r", encoding="utf-8") as fh:
                names = list(json.load(fh).keys())
        except Exception:
            names = []

        if names:
            try:
                from .vector_metrics_db import VectorMetricsDB

                vdb = VectorMetricsDB("vector_metrics.db")
                if not vdb.get_db_weights():
                    vdb.set_db_weights({n: 1.0 for n in names})
                vdb.conn.close()
            except Exception as exc:  # pragma: no cover - log only
                self.logger.warning("VectorMetricsDB bootstrap failed: %s", exc)

            try:
                data_root = resolve_path("sandbox_data")
            except FileNotFoundError:
                data_root = resolve_path(".") / "sandbox_data"
                try:
                    data_root.mkdir(parents=True, exist_ok=True)
                except Exception as exc:  # pragma: no cover - log only
                    self.logger.warning(
                        "failed creating sandbox_data directory: %s", exc
                    )
                    return
            hist = data_root / "roi_history.json"
            if not hist.exists():
                try:
                    hist.parent.mkdir(parents=True, exist_ok=True)
                    with open(hist, "w", encoding="utf-8") as fh:
                        json.dump({"origin_db_deltas": {n: [0.0] for n in names}}, fh)
                except Exception as exc:  # pragma: no cover - log only
                    self.logger.warning("ROITracker bootstrap failed: %s", exc)

    # ------------------------------------------------------------------
    def install_dependencies(self, requirements: Iterable[str]) -> None:
        pkgs = [req for req in requirements if req]
        if not pkgs:
            return
        if not startup_checks.auto_install_enabled():
            joined = ", ".join(pkgs)
            self.logger.info(
                "Automatic installation disabled; install missing dependencies manually: %s",
                joined,
            )
            self.logger.info(
                "Set %s=1 to re-enable automatic installation during bootstrap.",
                startup_checks.AUTO_INSTALL_ENV,
            )
            return
        for req in pkgs:
            try:
                self.logger.info("installing python dependency: %s", req)
                self._run([sys.executable, "-m", "pip", "install", req])
            except Exception as exc:  # pragma: no cover - log only
                self.logger.error("failed installing %s: %s", req, exc)

    # ------------------------------------------------------------------
    def run_migrations(self) -> None:
        if not Path("alembic.ini").exists():
            return
        if shutil.which("alembic") is None:
            self.logger.info("alembic command not available; skipping migrations")
            return
        try:
            self._run(["alembic", "upgrade", "head"])
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("migrations failed: %s", exc)

    # ------------------------------------------------------------------
    def _critical_prerequisites(
        self, check_budget: Callable[[], None], phase_budget: _PhaseBudgetContext | None = None
    ) -> None:
        ensure_config()
        check_budget()
        self.export_secrets()
        self.check_commands(self.required_commands)
        check_budget()
        self.check_nvidia_driver()
        pkgs = [p.strip() for p in self.required_os_packages if p.strip()]
        if pkgs:
            try:
                self.check_os_packages(pkgs)
            except RuntimeError as exc:
                msg = str(exc)
                if "missing OS packages:" in msg:
                    missing = [p.strip() for p in msg.split(":", 1)[1].split(",")]
                    SystemProvisioner(packages=missing).ensure_packages()
                    self.check_os_packages(pkgs)
                else:
                    raise
        check_budget()
        if self.policy.enforce_remote_checks:
            self.check_remote_dependencies(self.remote_endpoints)
        self.logger.info("verifying project dependencies declared in pyproject.toml")
        missing = startup_checks.verify_project_dependencies(policy=self.policy)
        if missing:
            self.logger.warning(
                "%d python packages are missing: %s",
                len(missing),
                ", ".join(sorted(missing)),
            )
            self.install_dependencies(missing)
        else:
            self.logger.info("all declared project dependencies import successfully")
        if self.policy.ensure_apscheduler and importlib.util.find_spec("apscheduler") is None:
            self.install_dependencies(["apscheduler"])
        check_budget()
        if phase_budget and not phase_budget.online_ready:
            phase_budget.mark_online_ready(reason="critical-subset-complete")

    # ------------------------------------------------------------------
    def _provisioning_phase(
        self,
        check_budget: Callable[[], None],
        phase_budget: _PhaseBudgetContext | None = None,
        *,
        skip_db_init: bool,
    ) -> None:
        if self.policy.enforce_systemd and shutil.which("systemctl"):
            result = subprocess.run(
                ["systemctl", "enable", "--now", "sandbox_autopurge.timer"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                details = stderr or stdout or f"exit code {result.returncode}"
                known_systemd_issue = any(
                    marker in details.lower()
                    for marker in (
                        "system has not been booted with systemd",
                        "failed to connect to bus",
                        "sandbox_autopurge.timer does not exist",
                    )
                )
                if known_systemd_issue:
                    self.logger.info(
                        "systemd unavailable; skipping sandbox_autopurge timer activation (%s)",
                        details,
                    )
                else:
                    self.logger.warning(
                        "failed enabling sandbox_autopurge.timer: %s", details
                    )
        elif self.policy.enforce_systemd:
            self.logger.info(
                "systemctl not available; skipping sandbox_autopurge timer activation"
            )
        deps = os.getenv("MENACE_BOOTSTRAP_DEPS", "").split(",")
        deps = [d.strip() for d in deps if d.strip()]
        if deps:
            self.install_dependencies(deps)
        if self.policy.additional_python_dependencies:
            self.install_dependencies(self.policy.additional_python_dependencies)
        if skip_db_init:
            self.logger.info(
                "database migrations skipped via MENACE_BOOTSTRAP_SKIP_DB_INIT"
            )
        elif self.policy.run_database_migrations:
            self.run_migrations()
        else:
            self.logger.debug(
                "policy '%s' disabled database migrations", self.policy.name
            )
        check_budget()
        self.bootstrapper.bootstrap()
        if phase_budget and not phase_budget.online_ready:
            phase_budget.mark_online_ready(reason="provisioning-core-ready")
        check_budget()
        interval = os.getenv("AUTO_PROVISION_INTERVAL")
        if interval:
            try:
                sec = float(interval)
            except ValueError:
                sec = 0.0
            if sec > 0:
                self._queue_background_task(
                    "auto-provision",
                    lambda: self._start_auto_provisioner(sec),
                    delay_until_ready=False,
                )
        check_budget()
        hosts = os.getenv("REMOTE_HOSTS", "").split(",")
        hosts = [h.strip() for h in hosts if h.strip()]
        if hosts:
            self.deploy_across_hosts(hosts)
        check_budget()
        self._queue_background_task(
            "vector-service-scheduler",
            start_scheduler_from_env,
            delay_until_ready=True,
            ready_gate="provisioning",
            stage="vector_seeding",
            budget_hint=self._stage_budget_hint("vector_seeding"),
            join_inner=False,
        )

    # ------------------------------------------------------------------
    def _optional_tail(self, check_budget: Callable[[], None]) -> None:
        if self.policy.provision_vector_assets:
            self.bootstrap_vector_assets()
            check_budget()

    # ------------------------------------------------------------------
    def bootstrap(
        self,
        *,
        timeout: float | None = None,
        halt_background: bool | None = None,
        skip_db_init: bool | None = None,
    ) -> None:
        with bootstrap_metrics.bootstrap_attempt(logger=self.logger):
            timeout = (
                timeout
                if timeout is not None
                else float(os.getenv("MENACE_BOOTSTRAP_TIMEOUT", "0") or 0) or None
            )
            halt_background = (
                halt_background
                if halt_background is not None
                else os.getenv("MENACE_BOOTSTRAP_HALT_THREADS") == "1"
            )
            skip_db_init = (
                skip_db_init
                if skip_db_init is not None
                else os.getenv("MENACE_BOOTSTRAP_SKIP_DB_INIT") == "1"
            )
            budgets = self._resolve_phase_budgets(timeout)
            try:
                self._run_phase(
                    "critical",
                    budgets.get("critical"),
                    self._critical_prerequisites,
                )
                self._run_phase(
                    "provisioning",
                    budgets.get("provisioning"),
                    lambda guard, state=None: self._provisioning_phase(
                        guard, state, skip_db_init=skip_db_init
                    ),
                )

                def optional_work() -> None:
                    self._run_phase(
                        "optional",
                        budgets.get("optional"),
                        lambda guard, state=None: self._optional_tail(guard),
                        strict=False,
                    )

                has_optional = self.policy.provision_vector_assets or budgets.get(
                    "optional"
                )
                if has_optional:
                    if halt_background:
                        optional_work()
                    else:
                        self._queue_background_task(
                            "optional-phase",
                            lambda: self._run_optional(optional_work),
                            delay_until_ready=False,
                            join_inner=False,
                            ready_gate="online",
                            stage="background_loops",
                            budget_hint=self._stage_budget_hint("background_loops"),
                        )
            finally:
                if halt_background:
                    self.shutdown()
                self._persist_phase_durations()

    def _run_optional(self, work: Callable[[], None]) -> None:
        work()

    def _start_auto_provisioner(self, interval: float) -> threading.Thread:
        stop_event = threading.Event()
        t = self.bootstrapper.run_continuous(interval=interval, stop_event=stop_event)
        self._stop_events.append(stop_event)
        self._background_threads.append(t)
        return t


__all__ = ["EnvironmentBootstrapper"]
