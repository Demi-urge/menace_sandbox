from __future__ import annotations

"""Generic autoscaling infrastructure.

This module contains an extensible :class:`Autoscaler` with provider
implementations for local processes, Docker Swarm and Kubernetes.  Several
defaults were historically hard coded which made the component tricky to tune
at runtime.  The implementation now exposes configuration parameters for the
cooldown timers, history window and executable paths so deployments can adjust
behaviour without modifying source code.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
import threading
from abc import ABC, abstractmethod
import subprocess
from typing import Dict, Deque
from collections import deque
import math
from dynamic_path_router import resolve_path

from .env_config import (
    BUDGET_MAX_INSTANCES,
    AUTOSCALE_TOLERANCE,
    SCALE_UP_THRESHOLD,
    SCALE_DOWN_THRESHOLD,
    LOCAL_PROVIDER_MAX_RESTARTS,
    LOCAL_PROVIDER_LOG_MAX_BYTES,
    LOCAL_PROVIDER_LOG_BACKUP_COUNT,
)

from logging_utils import get_logger

if not logging.getLogger().hasHandlers():  # pragma: no cover - config once
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

logger = get_logger(__name__)


def _resolve_binary(name: str, fallback_paths: list[str] | None = None) -> str:
    """Return the path to ``name`` if found or the first existing fallback."""
    from shutil import which

    path = which(name)
    if path:
        return path
    fallback_paths = fallback_paths or []
    for candidate in fallback_paths:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return name

if os.getenv("MENACE_LIGHT_IMPORTS"):
    from .light_bootstrap import EnvironmentBootstrapper  # pragma: no cover
else:  # pragma: no cover - real import when available
    try:
        from .environment_bootstrap import EnvironmentBootstrapper
    except Exception as exc:
        logger.debug("primary bootstrap import failed: %s", exc)
        try:
            from .environment_bootstrap import EnvironmentBootstrapper  # type: ignore
        except Exception as exc2:
            logger.debug("fallback bootstrap import failed: %s", exc2)
            try:
                from .light_bootstrap import EnvironmentBootstrapper  # pragma: no cover
            except Exception as exc3:
                logger.error("light bootstrap import failed: %s", exc3)
                raise ImportError("EnvironmentBootstrapper unavailable") from exc3


class Provider(ABC):
    """Base class for autoscaling providers."""

    @abstractmethod
    def scale_up(self, amount: int = 1) -> bool:
        """Increase the number of running instances.

        Returns ``True`` on success and ``False`` when scaling fails.
        """

    @abstractmethod
    def scale_down(self, amount: int = 1) -> bool:
        """Decrease the number of running instances.

        Returns ``True`` on success and ``False`` when scaling fails.
        """


class MetricFetcher(ABC):
    """Interface for retrieving metrics for autoscaling."""

    @abstractmethod
    def fetch(self) -> Dict[str, float]:
        """Return a mapping of metric names to values."""


class LocalProvider(Provider):
    """Simple subprocess based scaler for local processes.

    The provider starts an executable locally for each instance.  Historically
    ``menace_master.py`` was hard coded which made testing alternative entry
    points difficult.  ``executable`` now allows overriding this value either via
    parameter or the ``MENACE_EXECUTABLE_PATH`` environment variable.  ``log_path``
    can also be customised.
    """

    def __init__(self, executable: str | None = None, *, log_path: str | None = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.processes: list[subprocess.Popen] = []
        self.restart_counts: Dict[int, int] = {}
        self.max_restarts = LOCAL_PROVIDER_MAX_RESTARTS
        self._lock = threading.Lock()
        self.exec_path = executable or os.getenv("MENACE_EXECUTABLE_PATH")
        if self.exec_path:
            if not os.path.isabs(self.exec_path):
                try:
                    self.exec_path = str(resolve_path(self.exec_path))
                except FileNotFoundError:
                    pass
        else:
            self.exec_path = str(resolve_path("menace_master.py"))
        if not os.path.isfile(self.exec_path):
            self.logger.error("executable %s not found", self.exec_path)
            raise FileNotFoundError(f"Executable {self.exec_path} not found")
        log_file = log_path or os.getenv("LOCAL_PROVIDER_LOG_PATH", "local_provider.log")
        max_bytes = LOCAL_PROVIDER_LOG_MAX_BYTES
        backup_count = LOCAL_PROVIDER_LOG_BACKUP_COUNT
        handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.log_handle = open(log_file, "ab")

    def __del__(self) -> None:
        try:
            self.log_handle.close()
        except Exception as exc:
            self.logger.error("failed to close log file: %s", exc)

    def scale_up(self, amount: int = 1) -> bool:  # pragma: no cover - subprocess
        success = True
        for _ in range(amount):
            try:
                proc = subprocess.Popen(
                    [sys.executable, self.exec_path],
                    stdout=self.log_handle,
                    stderr=subprocess.STDOUT,
                )
                with self._lock:
                    self.processes.append(proc)
                    self.restart_counts[proc.pid] = 0
            except Exception as exc:
                self.logger.error("scale up failed: %s", exc)
                success = False
        return success

    def scale_down(self, amount: int = 1) -> bool:  # pragma: no cover - subprocess
        success = True
        for _ in range(amount):
            with self._lock:
                if not self.processes:
                    break
                proc = self.processes.pop()
                self.restart_counts.pop(proc.pid, None)
            try:
                proc.terminate()
                proc.wait(timeout=5)
                if proc.poll() is None:
                    proc.kill()
            except Exception as exc:
                self.logger.error("scale down failed: %s", exc)
                success = False
        return success

    def check_process_health(self) -> None:  # pragma: no cover - subprocess
        with self._lock:
            procs = list(self.processes)
        for i, proc in enumerate(procs):
            if proc.poll() is not None:
                count = self.restart_counts.get(proc.pid, 0) + 1
                if count > self.max_restarts:
                    self.logger.error("process %s exceeded restart limit", i)
                    with self._lock:
                        if proc in self.processes:
                            self.processes.remove(proc)
                            self.restart_counts.pop(proc.pid, None)
                    continue
                self.logger.warning("process %s exited; restarting", i)
                with self._lock:
                    if proc in self.processes:
                        self.processes.remove(proc)
                        self.restart_counts.pop(proc.pid, None)
                try:
                    time.sleep(2 ** min(count - 1, 5))
                    new_proc = subprocess.Popen(
                        [sys.executable, self.exec_path],
                        stdout=self.log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    with self._lock:
                        self.processes.append(new_proc)
                        self.restart_counts[new_proc.pid] = count
                except Exception as exc:
                    self.logger.error("failed to restart process: %s", exc)


class KubernetesProvider(Provider):
    """Scale deployments using ``kubectl``.

    The deployment name defaults to ``"menace"`` and can be overridden with the
    ``K8S_DEPLOYMENT`` environment variable.
    """

    def __init__(self, deployment: str | None = None) -> None:
        self.deployment = deployment or os.getenv("K8S_DEPLOYMENT", "menace")
        self.logger = get_logger(self.__class__.__name__)
        self._kubectl = _resolve_binary(
            "kubectl", ["/usr/local/bin/kubectl", "/usr/bin/kubectl"]
        )

    def _current(self) -> int:
        try:
            out = subprocess.check_output(
                [
                    self._kubectl,
                    "get",
                    "deployment",
                    self.deployment,
                    "-o",
                    "jsonpath={.spec.replicas}",
                ]
            )
            return int(out.decode().strip() or 0)
        except Exception as exc:  # pragma: no cover - optional
            self.logger.error("failed to fetch replicas: %s", exc)
            return 0

    def _set(self, replicas: int) -> bool:
        for attempt in range(3):
            try:
                subprocess.check_call(
                    [
                        self._kubectl,
                        "scale",
                        "deployment",
                        self.deployment,
                        f"--replicas={replicas}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception as exc:  # pragma: no cover - kubectl may be missing
                self.logger.error("scale failed: %s", exc)
                time.sleep(2 ** attempt)
        return False

    def scale_up(self, amount: int = 1) -> bool:
        return self._set(self._current() + amount)

    def scale_down(self, amount: int = 1) -> bool:
        return self._set(max(0, self._current() - amount))


class DockerSwarmProvider(Provider):
    """Scale Docker Swarm services.

    The service name defaults to ``"menace"`` and can be overridden with the
    ``SWARM_SERVICE`` environment variable.
    """

    def __init__(self, service: str | None = None) -> None:
        self.service = service or os.getenv("SWARM_SERVICE", "menace")
        self.logger = get_logger(self.__class__.__name__)
        self._docker = _resolve_binary(
            "docker", ["/usr/local/bin/docker", "/usr/bin/docker"]
        )

    def _current(self) -> int:
        try:
            out = subprocess.check_output(
                [
                    self._docker,
                    "service",
                    "inspect",
                    self.service,
                    "-f",
                    "{{.Spec.Mode.Replicated.Replicas}}",
                ]
            )
            return int(out.decode().strip() or 0)
        except Exception as exc:  # pragma: no cover - docker may be missing
            self.logger.error("failed to fetch replicas: %s", exc)
            return 0

    def _set(self, replicas: int) -> bool:
        for attempt in range(3):
            try:
                subprocess.check_call(
                    [self._docker, "service", "scale", f"{self.service}={replicas}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception as exc:  # pragma: no cover
                self.logger.error("scale failed: %s", exc)
                time.sleep(2 ** attempt)
        return False

    def scale_up(self, amount: int = 1) -> bool:
        return self._set(self._current() + amount)

    def scale_down(self, amount: int = 1) -> bool:
        return self._set(max(0, self._current() - amount))


class ScalingPolicy:
    """Compute desired replicas from historical metrics using moving averages."""

    def __init__(
        self,
        window: int = 5,
        *,
        max_instances: int | None = None,
        tolerance: float = 0.1,
        trend_threshold: float = 0.05,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
    ) -> None:
        self.window = window
        self.max_instances = max_instances
        self.tolerance = tolerance
        self.trend_threshold = trend_threshold
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.history: Deque[Dict[str, float]] = deque(maxlen=window)

    def record(self, metrics: Dict[str, float]) -> None:
        self.history.append(metrics)

    def desired_replicas(self, current: int) -> int:
        if not self.history:
            return current

        def _ema(key: str, alpha: float = 0.3) -> float:
            values = [m.get(key, 0.0) for m in self.history]
            if not values:
                return 0.0
            ema = values[0]
            for val in values[1:]:
                ema = alpha * val + (1 - alpha) * ema
            return ema

        def _trend(key: str) -> float:
            if len(self.history) < 2:
                return 0.0
            diffs = [
                b.get(key, 0.0) - a.get(key, 0.0)
                for a, b in zip(list(self.history)[:-1], list(self.history)[1:])
            ]
            return sum(diffs) / len(diffs)

        avg_cpu = _ema("cpu")
        avg_mem = _ema("memory")
        pred_cpu = max(0.0, min(1.0, avg_cpu + _trend("cpu")))
        pred_mem = max(0.0, min(1.0, avg_mem + _trend("memory")))

        desired = current
        high_cpu = avg_cpu * (1 + self.tolerance)
        low_cpu = avg_cpu * (1 - self.tolerance)
        high_mem = avg_mem * (1 + self.tolerance)
        low_mem = avg_mem * (1 - self.tolerance)

        trend_cpu = _trend("cpu")
        trend_mem = _trend("memory")

        load = max(pred_cpu, pred_mem)
        if (
            pred_cpu > self.scale_up_threshold
            or pred_mem > self.scale_up_threshold
            or pred_cpu > high_cpu
            or pred_mem > high_mem
        ):
            delta = max(1, int(math.ceil((load - self.scale_up_threshold) / 0.1)))
            desired += delta
        elif (
            (pred_cpu < self.scale_down_threshold and pred_mem < self.scale_down_threshold)
            or (pred_cpu < low_cpu and pred_mem < low_mem)
        ):
            delta = max(1, int(math.ceil((self.scale_down_threshold - load) / 0.1)))
            desired = max(0, desired - delta)

        if self.max_instances is not None:
            desired = min(desired, self.max_instances)
        return desired

class Autoscaler:
    """Manage resource scaling with pluggable providers."""

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        max_instances: int | None = None,
        policy: ScalingPolicy | None = None,
        metric_fetcher: MetricFetcher | None = None,
        cooldown: float | None = None,
        scale_up_cooldown: float | None = None,
        scale_down_cooldown: float | None = None,
        history_window: int = 5,
    ) -> None:
        if provider is None:
            name = os.getenv("AUTOSCALER_PROVIDER", "local").lower()
            if name == "kubernetes":
                provider = KubernetesProvider()
            elif name in {"swarm", "docker", "docker-swarm"}:
                provider = DockerSwarmProvider()
            else:
                provider = LocalProvider()
        self.provider = provider
        # Global limit set via environment or configuration
        self.max_instances = max_instances if max_instances is not None else BUDGET_MAX_INSTANCES
        self.policy = policy
        self.metric_fetcher = metric_fetcher
        if self.policy and self.policy.max_instances is None:
            self.policy.max_instances = self.max_instances
        self.instances = 0
        self.cooldown = cooldown if cooldown is not None else float(os.getenv("AUTOSCALE_COOLDOWN", "30"))
        self.scale_up_cooldown = scale_up_cooldown if scale_up_cooldown is not None else float(os.getenv("SCALE_UP_COOLDOWN", str(self.cooldown)))
        self.scale_down_cooldown = scale_down_cooldown if scale_down_cooldown is not None else float(os.getenv("SCALE_DOWN_COOLDOWN", str(self.cooldown)))
        self.last_scaled_at = 0.0
        self.last_scaled_up_at = 0.0
        self.last_scaled_down_at = 0.0
        self.logger = get_logger(self.__class__.__name__)
        self.history: Deque[Dict[str, float]] = deque(maxlen=history_window)
        self.tolerance = AUTOSCALE_TOLERANCE
        self.scale_up_threshold = SCALE_UP_THRESHOLD
        self.scale_down_threshold = SCALE_DOWN_THRESHOLD
        self._handled_hosts: set[str] = set()
        self._lock = threading.Lock()

    def set_max_instances(self, value: int) -> None:
        """Update the maximum allowed instances at runtime."""
        with self._lock:
            self.max_instances = value
            if self.policy:
                self.policy.max_instances = value

    def scale_up(self, amount: int = 1, *, hosts: list[str] | None = None) -> None:
        with self._lock:
            if time.time() - self.last_scaled_up_at < self.scale_up_cooldown:
                self.logger.info("scale_up ignored; cooldown active")
                return
            if self.max_instances and self.instances >= self.max_instances:
                self.logger.info("scale_up ignored; max instances reached")
                return
            allowed = amount
            if self.max_instances:
                allowed = min(amount, self.max_instances - self.instances)
                if allowed <= 0:
                    self.logger.info("scale_up ignored; max instances reached")
                    return
            try:
                ok = self.provider.scale_up(allowed)
            except Exception as exc:
                self.logger.error("provider scale_up failed: %s", exc)
                ok = False
            if ok:
                self.instances += allowed
                now = time.time()
                self.last_scaled_up_at = now
                self.last_scaled_at = now
        if hosts is None:
            hosts_env = os.getenv("NEW_HOSTS", "")
            hosts = []
            if hosts_env:
                try:
                    parsed = json.loads(hosts_env)
                    if isinstance(parsed, list):
                        hosts = [str(h).strip() for h in parsed if str(h).strip()]
                    else:
                        hosts = [str(parsed).strip()]
                except Exception:
                    hosts = [h.strip() for h in hosts_env.split(",") if h.strip()]
        new_hosts = [h for h in (hosts or []) if h not in self._handled_hosts]
        if new_hosts:
            self._deploy_hosts(new_hosts)
            self._handled_hosts.update(new_hosts)

    def _deploy_hosts(self, hosts: list[str]) -> None:
        """Safely bootstrap *hosts* using :class:`EnvironmentBootstrapper`."""
        try:
            EnvironmentBootstrapper().deploy_across_hosts(hosts)
        except Exception as exc:
            self.logger.error("host bootstrap failed: %s", exc)

    def scale_down(self, amount: int = 1) -> None:
        with self._lock:
            if time.time() - self.last_scaled_down_at < self.scale_down_cooldown:
                self.logger.info("scale_down ignored; cooldown active")
                return
            try:
                ok = self.provider.scale_down(amount)
            except Exception as exc:
                self.logger.error("provider scale_down failed: %s", exc)
                ok = False
            if ok:
                self.instances = max(0, self.instances - amount)
                now = time.time()
                self.last_scaled_down_at = now
                self.last_scaled_at = now

    def scale(self, metrics: Dict[str, float] | None = None) -> None:
        if metrics is None and self.metric_fetcher is not None:
            metrics = self.metric_fetcher.fetch()
        if metrics is None:
            self.logger.warning("No metrics provided for scaling")
            return
        if self.policy:
            self.policy.record(metrics)
            desired = self.policy.desired_replicas(self.instances)
            diff = desired - self.instances
            if diff > 0:
                self.logger.info("Scaling up")
                self.scale_up(diff)
            elif diff < 0:
                self.logger.info("Scaling down")
            self.scale_down(-diff)
            return

        self.history.append(metrics)
        if hasattr(self.provider, "check_process_health"):
            try:
                self.provider.check_process_health()
            except Exception as exc:  # pragma: no cover - optional
                self.logger.error("health check failed: %s", exc)
        cpu = float(metrics.get("cpu", 0.0))
        mem = float(metrics.get("memory", 0.0))
        avg_cpu = sum(m.get("cpu", 0.0) for m in self.history) / len(self.history)
        avg_mem = sum(m.get("memory", 0.0) for m in self.history) / len(self.history)

        def _trend(key: str) -> float:
            if len(self.history) < 2:
                return 0.0
            diffs = [
                self.history[i + 1].get(key, 0.0) - self.history[i].get(key, 0.0)
                for i in range(len(self.history) - 1)
            ]
            return sum(diffs) / len(diffs)

        pred_cpu = max(0.0, min(1.0, avg_cpu + _trend("cpu")))
        pred_mem = max(0.0, min(1.0, avg_mem + _trend("memory")))

        high_cpu = avg_cpu * (1 + self.tolerance)
        low_cpu = avg_cpu * (1 - self.tolerance)
        high_mem = avg_mem * (1 + self.tolerance)
        low_mem = avg_mem * (1 - self.tolerance)

        load = max(pred_cpu, pred_mem)
        if (
            pred_cpu > self.scale_up_threshold
            or pred_mem > self.scale_up_threshold
            or pred_cpu > high_cpu
            or pred_mem > high_mem
        ):
            if self.max_instances and self.instances >= self.max_instances:
                self.logger.info("scale_up ignored; max instances reached")
                return
            delta = max(1, int(math.ceil((load - self.scale_up_threshold) / 0.1)))
            self.logger.info("Scaling up")
            self.scale_up(delta)
        elif (
            (pred_cpu < self.scale_down_threshold and pred_mem < self.scale_down_threshold)
            or (pred_cpu < low_cpu and pred_mem < low_mem)
        ):
            delta = max(1, int(math.ceil((self.scale_down_threshold - load) / 0.1)))
            self.logger.info("Scaling down")
            self.scale_down(delta)


__all__ = [
    "Provider",
    "LocalProvider",
    "KubernetesProvider",
    "DockerSwarmProvider",
    "MetricFetcher",
    "ScalingPolicy",
    "Autoscaler",
]
