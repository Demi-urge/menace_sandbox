from __future__ import annotations

"""Coordinate incremental training from event bus activity."""

from typing import Optional
import logging

from .unified_event_bus import EventBus
from .data_bot import MetricsDB
from .neuroplasticity import PathwayRecord, Outcome
from .learning_engine import LearningEngine
from .unified_learning_engine import UnifiedLearningEngine
from .action_learning_engine import ActionLearningEngine
from .evaluation_manager import EvaluationManager
from .error_bot import ErrorBot
from .curriculum_builder import CurriculumBuilder


logger = logging.getLogger(__name__)


class SelfLearningCoordinator:
    """Listen for events and incrementally update learning engines."""

    def __init__(
        self,
        event_bus: EventBus,
        *,
        learning_engine: LearningEngine | None = None,
        unified_engine: UnifiedLearningEngine | None = None,
        action_engine: ActionLearningEngine | None = None,
        eval_interval: int = 0,
        metrics_db: MetricsDB | None = None,
        error_bot: ErrorBot | None = None,
        summary_interval: int = 0,
        curriculum_builder: CurriculumBuilder | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.learning_engine = learning_engine
        self.unified_engine = unified_engine
        self.action_engine = action_engine
        self.running = False
        self.eval_interval = eval_interval
        self.metrics_db = metrics_db
        self.error_bot = error_bot
        self.summary_interval = summary_interval
        self.curriculum_builder = curriculum_builder
        self._summary_count = 0
        self._train_count = 0
        self.evaluation_manager = EvaluationManager(
            learning_engine, unified_engine, action_engine
        )
        self.best_engine: Optional[object] = None

    # --------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        topics = [
            ("memory:new", self._on_memory),
            ("code:new", self._on_code),
            ("pathway:new", self._on_pathway),
            ("workflows:new", self._on_workflow),
            ("errors:new", self._on_error),
            ("telemetry:new", self._on_telemetry),
            ("metrics:new", self._on_metrics),
            ("transactions:new", self._on_transaction),
            ("curriculum:new", self._on_curriculum),
        ]
        for topic, cb in topics:
            try:
                self.event_bus.subscribe(topic, cb)
            except Exception as exc:
                log = getattr(self.error_bot, "logger", logger)
                log.warning("failed to subscribe %s: %s", topic, exc)

    def stop(self) -> None:
        self.running = False

    # --------------------------------------------------------------
    def _on_memory(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        actions = str(payload.get("key", payload.get("data", "")))
        rec = PathwayRecord(
            actions=actions,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=0.0,
        )
        self._train_all(rec)

    def _on_code(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        actions = str(payload.get("summary", payload.get("code", "")))
        rec = PathwayRecord(
            actions=actions,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=float(payload.get("complexity_score", 0.0)),
        )
        self._train_all(rec)

    def _on_workflow(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        steps = payload.get("workflow")
        if isinstance(steps, list):
            actions = "->".join(str(s) for s in steps)
        else:
            actions = str(steps or payload.get("title", ""))
        status = str(payload.get("status", "")).lower()
        oc = Outcome.FAILURE if "fail" in status or "reject" in status else Outcome.SUCCESS
        rec = PathwayRecord(
            actions=actions,
            inputs="",
            outputs="",
            exec_time=float(payload.get("workflow_duration", 0.0)),
            resources="",
            outcome=oc,
            roi=float(payload.get("estimated_profit_per_bot", 0.0)),
        )
        self._train_all(rec)

    def _on_pathway(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            rec = PathwayRecord(
                actions=payload.get("actions", ""),
                inputs=payload.get("inputs", ""),
                outputs=payload.get("outputs", ""),
                exec_time=float(payload.get("exec_time", 0.0)),
                resources=payload.get("resources", ""),
                outcome=Outcome(payload.get("outcome", "FAILURE")),
                roi=float(payload.get("roi", 0.0)),
                ts=payload.get("ts", ""),
            )
        except Exception:
            return
        self._train_all(rec)

    def _on_error(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        msg = str(payload.get("message", payload.get("error_type", "")))
        rec = PathwayRecord(
            actions=msg,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.0,
        )
        self._train_all(rec)

    def _on_telemetry(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        msg = str(payload.get("error_type", payload.get("stack_trace", "")))
        rec = PathwayRecord(
            actions=msg,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.0,
        )
        self._train_all(rec)

    def _on_metrics(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            expense = float(payload.get("expense", 0.0))
            revenue = float(payload.get("revenue", 0.0))
            roi = (revenue - expense) / (expense or 1.0)
        except Exception:
            roi = 0.0
        resources = (
            f"cpu={payload.get('cpu', 0.0)},mem={payload.get('memory', 0.0)},"
            f"disk={payload.get('disk_io', 0.0)},net={payload.get('net_io', 0.0)}"
        )
        oc = Outcome.FAILURE if int(payload.get("errors", 0)) > 0 else Outcome.SUCCESS
        rec = PathwayRecord(
            actions=str(payload.get("bot", "")),
            inputs="",
            outputs="",
            exec_time=float(payload.get("response_time", 0.0)),
            resources=resources,
            outcome=oc,
            roi=roi,
            ts=str(payload.get("ts", "")),
        )
        self._train_all(rec)

    def _on_transaction(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            amount = float(payload.get("amount", 0.0))
        except Exception:
            amount = 0.0
        result = str(payload.get("result", ""))
        oc = Outcome.SUCCESS if "success" in result.lower() else Outcome.FAILURE
        rec = PathwayRecord(
            actions=str(payload.get("model_id", "")),
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=oc,
            roi=amount,
        )
        self._train_all(rec)

    def _on_curriculum(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        err = str(payload.get("error_type", ""))
        if not err:
            return
        rec = PathwayRecord(
            actions=err,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.0,
        )
        self._train_all(rec)

    # --------------------------------------------------------------
    def _train_all(self, rec: PathwayRecord) -> None:
        self._train_record(rec)
        self._train_count += 1
        if self.eval_interval and self._train_count >= self.eval_interval:
            self._evaluate_all()
            self._train_count = 0
        if self.summary_interval and self.error_bot:
            self._summary_count += 1
            if self._summary_count >= self.summary_interval:
                self._train_from_summary()
                self._summary_count = 0

    def _train_record(self, rec: PathwayRecord) -> None:
        if self.learning_engine:
            try:
                self.learning_engine.partial_train(rec)
            except Exception as exc:
                logger.exception(
                    "learning_engine failed during partial_train: %s", exc
                )
        if self.unified_engine:
            try:
                self.unified_engine.partial_train(rec)
            except Exception as exc:
                logger.exception(
                    "unified_engine failed during partial_train: %s", exc
                )
        if self.action_engine:
            try:
                self.action_engine.partial_train(rec)
            except Exception as exc:
                logger.exception(
                    "action_engine failed during partial_train: %s", exc
                )

    def _train_from_summary(self) -> None:
        if self.curriculum_builder:
            self.curriculum_builder.publish()
            return
        if not self.error_bot:
            return
        summary = self.error_bot.summarize_telemetry()
        summary.sort(key=lambda s: s.get("success_rate", 0.0))
        for item in summary:
            rec = PathwayRecord(
                actions=str(item.get("error_type", "")),
                inputs="",
                outputs="",
                exec_time=0.0,
                resources="",
                outcome=Outcome.FAILURE
                if item.get("success_rate", 0.0) < 0.5
                else Outcome.SUCCESS,
                roi=0.0,
            )
            self._train_record(rec)

    def _evaluate_all(self) -> None:
        try:
            self.evaluation_manager.evaluate_all()
            self.best_engine = self.evaluation_manager.best_engine()
        except Exception as exc:
            logger.exception(
                "evaluation_manager failed during evaluate_all: %s", exc
            )


__all__ = ["SelfLearningCoordinator", "CurriculumBuilder"]
