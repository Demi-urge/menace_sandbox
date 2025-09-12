from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, List, Tuple, Any
import importlib.metadata as metadata
import uuid

from .resilience import CircuitBreaker, retry_with_backoff
from .logging_utils import set_correlation_id, get_logger

from .failure_learning_system import DiscrepancyDB
from .resource_allocation_optimizer import ResourceAllocationOptimizer


registry = BotRegistry()
data_bot = DataBot(start_server=False)

@dataclass
class Detection:
    """Result from a discrepancy rule."""

    message: str
    severity: float
    workflow: str | None = None
    ts: str = datetime.utcnow().isoformat()


RuleFunc = Callable[[Any, Any, Any], Iterable[Tuple[str, float, str | None]]]


class DiscrepancyError(Exception):
    """Base class for discrepancy detection errors."""


class RuleLoadError(DiscrepancyError):
    """Raised when loading discrepancy rules fails."""


class RuleExecutionError(DiscrepancyError):
    """Raised when a discrepancy rule execution fails."""


class OptimizationError(DiscrepancyError):
    """Raised when post-detection optimisation fails."""


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
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
        self._rule_circuit = CircuitBreaker()
        self._opt_circuit = CircuitBreaker()
        self._load_rules()
        logging.basicConfig(level=logging.INFO)
        self.logger = get_logger(__name__)

    def _load_rules(self) -> None:
        try:
            eps = metadata.entry_points()
        except Exception as exc:
            self.logger.exception("failed to gather discrepancy rule entry points")
            raise RuleLoadError("entry point discovery failed") from exc
        for ep in eps.select(group="menace.discrepancy_rules"):
            try:
                func = ep.load()
                if callable(func):
                    self.rules.append((ep.name, func))
            except Exception as exc:
                self.logger.exception("failed loading rule %s", ep.name, exc_info=exc)
                raise RuleLoadError(f"failed loading rule {ep.name}") from exc

    def scan(self) -> List[Detection]:
        cid = uuid.uuid4().hex
        set_correlation_id(cid)
        findings: List[Detection] = []
        for name, rule in self.rules:
            try:
                results = retry_with_backoff(
                    lambda: self._rule_circuit.call(
                        lambda: rule(self.models_db, self.workflows_db, self.storage)
                    ),
                    attempts=3,
                    logger=self.logger,
                )
            except Exception as e:
                self.logger.warning("rule %s failed after retries: %s", name, e)
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
                        retry_with_backoff(
                            lambda: self._opt_circuit.call(
                                lambda: self.optimizer.scale_down(str(det.workflow))
                            ),
                            attempts=3,
                            logger=self.logger,
                        )
                    except Exception as e:
                        self.logger.warning(
                            "scale_down failed for workflow %s: %s", det.workflow, e
                        )
        set_correlation_id(None)
        return findings


__all__ = [
    "Detection",
    "DiscrepancyDetectionBot",
    "DiscrepancyError",
    "RuleLoadError",
    "RuleExecutionError",
    "OptimizationError",
]