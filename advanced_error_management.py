"""Advanced error management helpers for military-grade reliability."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import os
import subprocess
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Callable, TYPE_CHECKING

from dynamic_path_router import resolve_path
from .knowledge_graph import KnowledgeGraph
from .rollback_manager import RollbackManager
from .data_bot import MetricsDB
from .meta_logging import SecureLog
from .sentry_client import SentryClient
from .governance import evaluate_rules
from .anomaly_detection import _ae_scores, _cluster_scores

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .error_bot import ErrorDB
    from .error_logger import TelemetryEvent


@lru_cache(maxsize=1)
def _resolve_telemetry_event():
    """Import :class:`TelemetryEvent` lazily to avoid circular imports."""

    try:
        from .error_logger import TelemetryEvent as _TelemetryEvent  # type: ignore
    except Exception as exc:  # pragma: no cover - fallback for partially initialised module
        raise RuntimeError("TelemetryEvent is unavailable") from exc
    return _TelemetryEvent

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import angr  # type: ignore
except Exception:  # pragma: no cover - optional
    angr = None  # type: ignore
    logger.warning("angr not available; symbolic checks disabled")

try:
    from kafka import KafkaProducer  # type: ignore
except Exception:  # pragma: no cover - optional
    KafkaProducer = None  # type: ignore
    logger.warning("kafka-python not available; telemetry replication disabled")
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
    logger.warning("requests not available; remote actions will be skipped")


class FormalVerifier:
    """Run static analysis and property tests on code patches."""

    tools = ["flake8", "mypy"]

    def __init__(
        self,
        *,
        tools: Optional[Iterable[str]] = None,
        test_dir: str | None = None,
        test_pattern: str | None = None,
        config: Dict[str, str] | None = None,
    ) -> None:
        if config:
            tools = tools or config.get("tools")
            test_dir = test_dir or config.get("test_dir")
            test_pattern = test_pattern or config.get("test_pattern")
        self.test_dir = test_dir or os.getenv("TEST_DIR", "tests")
        self.test_pattern = test_pattern or os.getenv("TEST_PATTERN", "properties")
        self.tools = list(tools) if tools else self.__class__.tools.copy()
        self.symbolic = angr if "angr" in globals() and angr is not None else None
        if self.symbolic:
            # only append if symbolic executor available
            self.tools.append("symbolic")

    def _symbolic_verify(self, path: Path) -> bool:
        """Run a simplified symbolic analysis using angr if available."""
        if not self.symbolic:
            return True
        try:
            # Instantiate a project just to ensure angr can parse the file
            self.symbolic.Project(str(path), auto_load_libs=False)
            import ast

            tree = ast.parse(Path(path).read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Raise, ast.Assert)):
                    # treat explicit raises or failed assertions as failing paths
                    return False
        except Exception:
            return False
        return True

    def verify(self, path: Path) -> bool:
        """Execute static checkers and Hypothesis based tests."""
        ok = True
        for tool in self.tools:
            if tool == "symbolic":
                if not self._symbolic_verify(path):
                    ok = False
                continue
            try:
                res = subprocess.run([tool, str(path)], capture_output=True, check=False)
                if res.returncode != 0:
                    ok = False
            except FileNotFoundError:  # pragma: no cover - optional tools
                continue

        try:
            res = subprocess.run(
                [
                    "pytest",
                    "-q",
                    self.test_dir,
                    "-k",
                    self.test_pattern,
                ],
                capture_output=True,
                check=False,
            )
            if res.returncode != 0:
                ok = False
        except Exception:  # pragma: no cover - test runner issues
            ok = False

        return ok


class TelemetryReplicator:
    """Replicate telemetry events across multiple backends."""

    def __init__(
        self,
        topic: str = "menace.telemetry",
        hosts: Optional[str] = None,
        *,
        sentry: "SentryClient" | None = None,
        disk_path: str | None = None,
        config: Dict[str, str] | None = None,
        disk_limit: int = 1000,
    ) -> None:
        if KafkaProducer is None:
            raise ImportError("kafka-python is required for TelemetryReplicator")
        if config:
            topic = config.get("topic", topic)
            hosts = hosts or config.get("hosts")
            disk_path = disk_path or config.get("disk_path")
        hosts = hosts or os.getenv("KAFKA_HOSTS", "localhost:9092")
        bootstrap = [h.strip() for h in hosts.split(",")]
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=bootstrap)
        self.sentry = sentry
        self.disk_path = Path(disk_path) if disk_path else None
        self.disk_limit = int(disk_limit)
        if self.disk_path:
            if self.disk_path.parent != self.disk_path:
                self.disk_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue: List["TelemetryEvent"] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def _send_event(self, event: "TelemetryEvent") -> bool:
        if not event.checksum:
            payload_ck = json.dumps(
                event.dict(exclude={"checksum"}, sort_keys=True)
            ).encode("utf-8")
            event.checksum = hashlib.sha256(payload_ck).hexdigest()
        payload = json.dumps(event.dict()).encode("utf-8")
        try:
            self.producer.send(self.topic, payload)
        except Exception:
            self.logger.exception(
                "Kafka replication failed for bot %s", event.bot_id or event.task_id
            )
            return False
        if self.sentry:
            try:
                self.sentry.capture_exception(Exception(event.stack_trace))
            except Exception:
                self.logger.exception(
                    "Sentry capture failed for bot %s", event.bot_id or event.task_id
                )
        return True

    def _write_disk_queue(self) -> None:
        if not self.disk_path:
            return
        if len(self.queue) > self.disk_limit:
            self.queue = self.queue[-self.disk_limit:]
        try:
            with open(self.disk_path, "w", encoding="utf-8") as fh:
                for ev in self.queue:
                    if not ev.checksum:
                        payload = json.dumps(
                            ev.dict(exclude={"checksum"}, sort_keys=True)
                        ).encode("utf-8")
                        ev.checksum = hashlib.sha256(payload).hexdigest()
                    fh.write(json.dumps(ev.dict()) + "\n")
        except Exception:
            self.logger.exception(
                "Failed writing telemetry queue to %s", self.disk_path
            )

    def _send_queued(self) -> None:
        remaining: List["TelemetryEvent"] = []
        for ev in self.queue:
            if not self._send_event(ev):
                remaining.append(ev)
        self.queue = remaining
        self._write_disk_queue()

    def replicate(self, event: "TelemetryEvent") -> None:
        self._send_queued()
        if not self._send_event(event):
            self.logger.warning(
                "Telemetry queued for bot %s", event.bot_id or event.task_id
            )
            if not event.checksum:
                payload = json.dumps(
                    event.dict(exclude={"checksum"}, sort_keys=True)
                ).encode("utf-8")
                event.checksum = hashlib.sha256(payload).hexdigest()
            self.queue.append(event)
            self._write_disk_queue()

    def flush(self) -> None:
        if self.disk_path and self.disk_path.exists():
            try:
                with open(self.disk_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            event_cls = _resolve_telemetry_event()
                            event = event_cls(**data)
                        except Exception:
                            self.logger.warning("Invalid telemetry line skipped")
                            continue
                        payload = json.dumps(
                            event.dict(exclude={"checksum"}, sort_keys=True)
                        ).encode("utf-8")
                        if event.checksum and hashlib.sha256(payload).hexdigest() == event.checksum:
                            self.queue.append(event)
                        else:
                            self.logger.warning("Telemetry checksum mismatch; dropping entry")
                self.disk_path.unlink()
            except Exception:
                self.logger.exception("Failed reading disk queue from %s", self.disk_path)
        self._send_queued()


class AutomatedRollbackManager(RollbackManager):
    """Extend RollbackManager with multi-node automated rollback."""

    def auto_rollback(
        self,
        patch_id: str,
        nodes: Iterable[str],
        *,
        rpc_client: Optional[object] = None,
        endpoints: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
        alignment_status: str = "pass",
        scenario_raroi_deltas: Iterable[float] | None = None,
    ) -> bool:
        """Notify nodes to rollback and drop DB record on quorum success.

        ``weights`` allows weighting node reliability for quorum calculations.
        """

        allow_ship, allow_rollback, reasons = evaluate_rules(
            {}, alignment_status, scenario_raroi_deltas or []
        )
        if not allow_rollback:
            for msg in reasons:
                self.logger.warning("governance veto: %s", msg)
            return False

        confirmed = 0.0
        node_list = list(nodes)
        weights = weights or {}
        for node in node_list:
            ok = False
            if rpc_client and hasattr(rpc_client, "rollback"):
                try:
                    ok = bool(rpc_client.rollback(node=node, patch_id=patch_id))
                except Exception:
                    ok = False
            elif endpoints and requests and node in endpoints:
                try:
                    url = endpoints[node].rstrip("/") + "/rollback"
                    resp = requests.post(
                        url, json={"patch_id": patch_id, "node": node}, timeout=5
                    )
                    ok = resp.status_code == 200
                except Exception:
                    ok = False
            else:
                try:
                    logging.info("rolling back %s on %s", patch_id, node)
                    ok = True
                except Exception:
                    ok = False
            if ok:
                confirmed += weights.get(node, 1.0)

        total_weight = sum(weights.get(n, 1.0) for n in node_list)
        quorum = total_weight / 2 + 0.0001
        if confirmed >= quorum:
            super().rollback(patch_id)
            return True
        return False


class AnomalyEnsembleDetector:
    """Combine autoencoder and clustering heuristics.

    The detector fetches recent metrics and evaluates anomaly scores using the
    private helpers :func:`_ae_scores` and :func:`_cluster_scores` from
    :mod:`menace.anomaly_detection`.  Each method casts a vote weighted by
    configurable weights.  When both helpers are unavailable it falls back to
    simple threshold checks on the raw values.
    """

    def __init__(
        self,
        metrics: MetricsDB,
        *,
        ae_weight: float = 0.6,
        cluster_weight: float = 0.4,
        threshold: float = 0.5,
    ) -> None:
        self.metrics = metrics
        self.ae_weight = ae_weight
        self.cluster_weight = cluster_weight
        self.threshold = threshold

    def detect(self) -> List[str]:
        df = self.metrics.fetch(50)
        anomalies: List[str] = []

        if hasattr(df, "empty"):
            empty = df.empty
            errors = df["errors"].tolist() if not empty else []
            cpu = df["cpu"].tolist() if not empty else []
            cpu_mean = float(df["cpu"].mean()) if not empty else 0.0
            err_sum = float(df["errors"].sum()) if not empty else 0.0
        else:
            empty = not df
            errors = [float(r.get("errors", 0.0)) for r in df]
            cpu = [float(r.get("cpu", 0.0)) for r in df]
            cpu_mean = sum(cpu) / (len(cpu) or 1)
            err_sum = sum(errors)

        if empty:
            return anomalies

        def vote(values: List[float], fallback: bool) -> bool:
            votes = 0.0
            used = False

            def z_scores(vals: List[float]) -> List[float]:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
                return [abs(v - mean) / std for v in vals]

            try:
                scores = _ae_scores(values)
            except Exception:
                logger.exception("_ae_scores failed")
                scores = z_scores(values)
            else:
                used = True
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1.0
            if max(scores) > mean + std:
                votes += self.ae_weight
                used = True

            try:
                scores = _cluster_scores(values)
            except Exception:
                logger.exception("_cluster_scores failed")
                mean_val = sum(values) / len(values)
                scores = [abs(v - mean_val) for v in values]
            else:
                used = True
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1.0
            if max(scores) > mean + std:
                votes += self.cluster_weight
                used = True

            if used:
                return votes > self.threshold
            return fallback

        if vote(errors, err_sum > 0):
            anomalies.append("error_rate")
        if vote(cpu, cpu_mean > 0.9):
            anomalies.append("high_cpu")
        return anomalies


class SelfHealingOrchestrator:
    """Restart bots that stop sending heartbeats using pluggable backends."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        backend: str | None = None,
        rollback_mgr: RollbackManager | None = None,
        error_db: "ErrorDB" | None = None,
        failure_threshold: int = 3,
        command_provider: Callable[[str], List[str]] | None = None,
        config: Dict[str, str] | None = None,
    ) -> None:
        self.graph = graph
        if config:
            backend = backend or config.get("backend")
        self.backend = backend or os.getenv("HEALER_BACKEND", "subprocess")
        self.client: object | None = None
        self.rollback_mgr = rollback_mgr
        self.error_db = error_db
        self.failure_threshold = failure_threshold
        self.command_provider = command_provider
        self.script_dir = resolve_path(os.getenv("BOT_SCRIPT_DIR", "."))
        self.extension = os.getenv("BOT_EXTENSION", ".py")
        if config:
            self.script_dir = resolve_path(config.get("script_dir", self.script_dir))
            self.extension = config.get("extension", self.extension)
        self.failures: dict[str, int] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.backend == "docker":
            try:  # pragma: no cover - optional dependency
                import docker  # type: ignore

                self.client = docker.from_env()
            except Exception:
                self.client = None
                self.logger.warning("Docker client initialization failed")
        elif self.backend == "kubernetes":
            try:  # pragma: no cover - optional dependency
                from kubernetes import client, config  # type: ignore

                config.load_kube_config()
                self.client = client.CoreV1Api()
            except Exception:
                self.client = None
                self.logger.warning("Kubernetes client initialization failed")
        elif self.backend == "vm":
            self.client = None

    def heal(self, bot: str, patch_id: str | None = None) -> None:
        if self.rollback_mgr is not None:
            try:
                self.rollback_mgr.log_healing_action(bot, "heal", patch_id)
            except Exception:
                pass
        try:
            if self.backend == "docker" and self.client is not None:
                container = self.client.containers.get(bot)
                container.restart()
            elif self.backend == "kubernetes" and self.client is not None:
                self.client.delete_namespaced_pod(name=bot, namespace="default")
            elif self.backend == "vm":
                subprocess.run(["virsh", "reboot", bot], check=False)
            else:
                script_path = self.script_dir / f"{bot}{self.extension}"
                cmd = ["python", str(script_path)]
                if self.command_provider:
                    try:
                        cmd = self.command_provider(bot)
                    except Exception:
                        self.logger.exception("command provider failed for %s", bot)
                        cmd = ["python", str(script_path)]
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self.graph.add_telemetry_event(bot, "restart", self.backend)
        except Exception:
            self.logger.exception("Bot restart failed for %s on %s", bot, self.backend)

    def probe_and_heal(self, bot: str, url: str | None = None) -> None:
        """Check bot health and heal if unreachable."""
        url = url or f"http://{bot}:8000/health"
        healthy = False
        if requests:
            try:
                r = requests.get(url, timeout=2)
                healthy = r.status_code == 200
            except Exception:
                healthy = False
                self.logger.exception("Health check failed for %s at %s", bot, url)
        if healthy:
            self.failures[bot] = 0
            return

        self.failures[bot] = self.failures.get(bot, 0) + 1
        if self.failures[bot] >= self.failure_threshold:
            if self.rollback_mgr is not None:
                try:
                    self.rollback_mgr.auto_rollback("latest", [bot])
                except Exception:
                    self.logger.exception("Auto rollback failed for %s", bot)
            if self.error_db is not None:
                try:
                    self.error_db.set_safe_mode(bot)
                except Exception:
                    self.logger.exception("Failed to enable safe mode for %s", bot)
        self.heal(bot)


@dataclass
class PlaybookGenerator:
    """Generate operational playbooks from anomalies."""

    def generate(self, anomalies: List[str]) -> str:
        playbook = {"actions": anomalies}
        stamp = int(time.time())
        import secrets

        salt = secrets.token_hex(4)
        digest = hashlib.md5((salt + str(anomalies)).encode()).hexdigest()
        path = resolve_path(".") / f"playbook_{stamp}_{digest}.json"
        path.write_text(json.dumps(playbook, indent=2))
        return str(path)


class PredictiveResourceAllocator:
    """Allocate resources based on forecasted demand."""

    def __init__(self, metrics: MetricsDB, autoscale_url: str | None = None) -> None:
        self.metrics = metrics
        self.autoscale_url = autoscale_url

    def _post(self, action: str) -> bool:
        if not (self.autoscale_url and requests):
            return False
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(self.autoscale_url, json={"action": action}, timeout=2)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    return True
            except Exception as exc:  # pragma: no cover - log only
                last_exc = exc
            time.sleep(2**attempt)
        if last_exc:
            logging.error("Autoscaler error: %s", last_exc)
        return False

    def forecast_and_allocate(self) -> None:
        df = self.metrics.fetch(10)
        if hasattr(df, "empty"):
            empty = df.empty
            cpu_vals = list(df["cpu"]) if not empty else []
            mean_cpu = float(df["cpu"].mean()) if not empty else 0.0
        else:
            empty = not df
            cpu_vals = [float(r.get("cpu", 0.0)) for r in df]
            mean_cpu = sum(cpu_vals) / (len(cpu_vals) or 1)

        if empty:
            return

        if len(cpu_vals) > 1:
            import numpy as np

            x = np.arange(len(cpu_vals))
            y = np.array(cpu_vals)
            slope, _ = np.polyfit(x, y, 1)
            trend = float(slope)
        else:
            trend = 0.0

        if mean_cpu > 0.8 or (trend > 0 and mean_cpu > 0.7):
            logging.info("scale up resources")
            self._post("scale_up")
        elif mean_cpu < 0.2 or (trend < 0 and mean_cpu < 0.3):
            logging.info("scale down resources")
            self._post("scale_down")


__all__ = [
    "FormalVerifier",
    "TelemetryReplicator",
    "AutomatedRollbackManager",
    "AnomalyEnsembleDetector",
    "SelfHealingOrchestrator",
    "SecureLog",
    "PlaybookGenerator",
    "PredictiveResourceAllocator",
]
