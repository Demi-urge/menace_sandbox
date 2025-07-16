from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, List, Tuple, Any
import importlib.metadata as metadata

from .failure_learning_system import DiscrepancyDB
from .resource_allocation_optimizer import ResourceAllocationOptimizer


@dataclass
class Detection:
    """Result from a discrepancy rule."""

    message: str
    severity: float
    workflow: str | None = None
    ts: str = datetime.utcnow().isoformat()


RuleFunc = Callable[[Any, Any, Any], Iterable[Tuple[str, float, str | None]]]


class DiscrepancyDetectionBot:
    """Run registered discrepancy rules and log findings."""

    def __init__(
        self,
        db: DiscrepancyDB | None = None,
        *,
        optimizer: ResourceAllocationOptimizer | None = None,
        models_db: Any = None,
        workflows_db: Any = None,
        storage: Any = None,
        severity_threshold: float = 1.0,
    ) -> None:
        self.db = db or DiscrepancyDB()
        self.optimizer = optimizer
        self.models_db = models_db
        self.workflows_db = workflows_db
        self.storage = storage
        self.severity_threshold = severity_threshold
        self.rules: List[Tuple[str, RuleFunc]] = []
        self._load_rules()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DiscrepancyDetectionBot")

    def _load_rules(self) -> None:
        try:
            eps = metadata.entry_points()
        except Exception as exc:
            self.logger.exception("failed to gather discrepancy rule entry points")
            raise
        for ep in eps.select(group="menace.discrepancy_rules"):
            try:
                func = ep.load()
                if callable(func):
                    self.rules.append((ep.name, func))
            except Exception as exc:
                self.logger.exception("failed loading rule %s", ep.name, exc_info=exc)
                raise

    def scan(self) -> List[Detection]:
        findings: List[Detection] = []
        for name, rule in self.rules:
            try:
                results = rule(self.models_db, self.workflows_db, self.storage)
            except Exception as e:
                self.logger.warning("rule %s failed: %s", name, e)
                continue
            if not results:
                continue
            if isinstance(results, tuple):
                results = [results]
            for msg, sev, wf in results:
                det = Detection(msg, float(sev), wf)
                findings.append(det)
                self.db.log_detection(name, det.severity, det.message, det.workflow)
                if self.optimizer and det.severity >= self.severity_threshold:
                    try:
                        self.optimizer.scale_down(str(det.workflow))
                    except Exception as e:
                        self.logger.warning(
                            "scale_down failed for workflow %s: %s", det.workflow, e
                        )
        return findings


__all__ = ["Detection", "DiscrepancyDetectionBot"]
