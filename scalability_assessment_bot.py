"""Scalability Assessment Bot for load testing and optimisation."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router


@dataclass
class TaskInfo:
    """Basic metrics for a single task."""

    name: str
    cpu: float
    memory: float


@dataclass
class ScalabilityReport:
    """Result of scalability analysis."""

    tasks: List[TaskInfo]
    bottlenecks: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class PerformanceDB:
    """Storage for performance metrics using :class:`DBRouter`."""

    def __init__(self, *, router: DBRouter | None = None) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router("scalability_assessment")
        self.conn = self.router.get_connection("sandbox_metrics")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics(task TEXT, cpu REAL, memory REAL)"
        )

    def log(self, task: str, cpu: float, memory: float) -> None:
        self.conn.execute(
            "INSERT INTO metrics(task, cpu, memory) VALUES (?, ?, ?)",
            (task, cpu, memory),
        )
        self.conn.commit()


@self_coding_managed
class ScalabilityAssessmentBot:
    """Analyse blueprints and simulate high load to find bottlenecks."""

    def __init__(
        self,
        db: PerformanceDB | None = None,
        rp_url: str | None = None,
        db_router: DBRouter | None = None,
    ) -> None:
        self.db_router = db_router or GLOBAL_ROUTER or init_db_router(
            "scalability_assessment"
        )
        perf_router = self.db_router if hasattr(self.db_router, "get_connection") else None
        self.db = db or PerformanceDB(router=perf_router)
        self.rp_url = rp_url
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ScalabilityBot")

    def parse_blueprint(self, data: str) -> Dict[str, Any]:
        """Parse blueprint JSON or YAML text."""
        try:
            obj = json.loads(data)
        except Exception:
            if yaml:
                obj = yaml.safe_load(data)
            else:
                raise
        return obj if isinstance(obj, dict) else {}

    def simulate_load(self, tasks: List[Any]) -> List[TaskInfo]:
        """Simulate load and produce more realistic metrics."""
        metrics: List[TaskInfo] = []
        for t in tasks:
            if isinstance(t, dict):
                name = t.get("name", "")
                ops = t.get("operations", [])
                cpu = float(t.get("cpu", len(name) * 0.1 + 0.05 * len(ops)))
                memory = float(
                    t.get("memory", len(name) * 0.2 + 0.1 * len(" ".join(map(str, ops))))
                )
            else:
                name = str(t)
                cpu = len(name) * 0.1
                memory = len(name) * 0.2
            metrics.append(TaskInfo(name=name, cpu=cpu, memory=memory))
            self.db.log(name, cpu, memory)
        return metrics

    def analyse(self, data: str) -> ScalabilityReport:
        """Run analysis on blueprint text."""
        if self.db_router:
            try:
                _ = self.db_router.query_all("scalability")
            except Exception:
                self.logger.exception("Failed to query scalability data")
        bp = self.parse_blueprint(data)
        tasks_raw = bp.get("tasks", [])
        metrics = self.simulate_load(tasks_raw)
        if metrics:
            avg_cpu = sum(m.cpu for m in metrics) / len(metrics)
            avg_mem = sum(m.memory for m in metrics) / len(metrics)
        else:
            avg_cpu = avg_mem = 0.0
        bottlenecks = [
            m.name
            for m in metrics
            if m.cpu > avg_cpu * 1.2 or m.memory > avg_mem * 1.2
        ]
        suggestions = [f"Reduce load of {n}" for n in bottlenecks]
        return ScalabilityReport(tasks=metrics, bottlenecks=bottlenecks, suggestions=suggestions)

    def send_report(self, report: ScalabilityReport) -> None:
        """Post suggestions to the Resource Prediction Bot."""
        if self.db_router:
            try:
                _ = self.db_router.query_all("scalability")
            except Exception:
                self.logger.exception("Failed to query scalability data")
        if self.rp_url and requests:
            try:
                requests.post(
                    self.rp_url,
                    json={
                        "bottlenecks": report.bottlenecks,
                        "suggestions": report.suggestions,
                    },
                    timeout=3,
                )
            except Exception:  # pragma: no cover - external failures
                self.logger.exception("Failed to send report")


__all__ = [
    "TaskInfo",
    "ScalabilityReport",
    "PerformanceDB",
    "ScalabilityAssessmentBot",
]
