from __future__ import annotations

"""Worker that executes evaluation tasks from :class:`UnifiedEventBus`."""

from .unified_event_bus import UnifiedEventBus
from .evaluation_manager import EvaluationManager
from .evaluation_history_db import EvaluationRecord
import logging
from .error_flags import RAISE_ERRORS

logger = logging.getLogger(__name__)


class EvaluationWorker:
    """Listen for ``evaluation:run`` tasks and publish results."""

    def __init__(self, event_bus: UnifiedEventBus, manager: EvaluationManager) -> None:
        self.event_bus = event_bus
        self.manager = manager
        self.event_bus.subscribe("evaluation:run", self._on_task)

    def _on_task(self, topic: str, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        name = payload.get("engine")
        if not isinstance(name, str):
            return
        engine = self.manager.engines.get(name)
        if engine is None:
            return
        evaluate = getattr(engine, "evaluate", None)
        if not callable(evaluate):
            return
        try:
            result = evaluate()
        except Exception as exc:  # pragma: no cover - evaluation failures ignored
            logger.exception("evaluation failed for %s: %s", name, exc)
            result = {"error": str(exc)}
            if self.manager.db is not None:
                try:
                    rec = EvaluationRecord(engine=name, cv_score=0.0, passed=False, error=str(exc))
                    self.manager.db.add(rec)
                except Exception:
                    logger.exception("db add failed")
            if RAISE_ERRORS:
                raise
        else:
            self.manager.history.setdefault(name, []).append(result)
            if self.manager.db is not None:
                try:
                    rec = EvaluationRecord(engine=name, cv_score=float(result.get("cv_score", 0.0)))
                    self.manager.db.add(rec)
                except Exception:
                    logger.exception("db add failed")
            persist = getattr(engine, "persist_evaluation", None)
            if callable(persist):
                try:
                    persist(result)
                except Exception:
                    logger.exception("persist failed")
            score = float(result.get("cv_score", 0.0))
            if score > self.manager._best_score:
                self.manager._best_score = score
                self.manager._best_name = name
        try:
            self.event_bus.publish("evaluation:result", {"engine": name, "result": result})
        except Exception:
            logger.exception("result publish failed")


__all__ = ["EvaluationWorker"]
