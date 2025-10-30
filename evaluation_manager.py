from __future__ import annotations

"""Manage evaluation results from multiple learning engines."""

from typing import Dict, List, Optional, TYPE_CHECKING
import logging
import threading
import time

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .error_cluster_analyzer import ErrorClusterAnalyzer
    from .error_cluster_predictor import ErrorClusterPredictor
    from .knowledge_graph import KnowledgeGraph
    from .error_bot import ErrorDB
    from .unified_event_bus import UnifiedEventBus

from .evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
from .error_flags import RAISE_ERRORS

from .learning_engine import LearningEngine
from .unified_learning_engine import UnifiedLearningEngine
from .action_learning_engine import ActionLearningEngine


logger = logging.getLogger(__name__)


class EvaluationManager:
    """Collect and compare evaluation results from multiple engines."""

    def __init__(
        self,
        learning_engine: LearningEngine | None = None,
        unified_engine: UnifiedLearningEngine | None = None,
        action_engine: ActionLearningEngine | None = None,
        history_db: EvaluationHistoryDB | None = None,
    ) -> None:
        self.engines: dict[str, object | None] = {
            "learning_engine": learning_engine,
            "unified_engine": unified_engine,
            "action_learning_engine": action_engine,
        }
        self.db: EvaluationHistoryDB | None = history_db
        self.history: Dict[str, List[Dict[str, float]]] = {
            name: [] for name, eng in self.engines.items() if eng is not None
        }
        self._best_name: str | None = None
        self._best_score: float = float("-inf")

    # --------------------------------------------------------------
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all configured engines and persist their results."""
        results: Dict[str, Dict[str, float]] = {}
        for name, eng in self.engines.items():
            if eng is None:
                continue
            evaluate = getattr(eng, "evaluate", None)
            if not callable(evaluate):
                continue
            try:
                res: Dict[str, float] = evaluate()
            except Exception as exc:
                logger.exception("evaluation failed for %s: %s", name, exc)
                self.history.setdefault(name, []).append({"error": str(exc)})
                if self.db is not None:
                    try:
                        rec = EvaluationRecord(
                            engine=name,
                            cv_score=0.0,
                            passed=False,
                            error=str(exc),
                        )
                        self.db.add(rec)
                    except Exception as db_exc:  # pragma: no cover - database failures
                        logger.error("Failed to add evaluation record for %s: %s", name, db_exc)
                if RAISE_ERRORS:
                    raise
                continue
            results[name] = res
            self.history.setdefault(name, []).append(res)
            if self.db is not None:
                try:
                    rec = EvaluationRecord(engine=name, cv_score=float(res.get("cv_score", 0.0)))
                    self.db.add(rec)
                except Exception as exc:  # pragma: no cover - database failures
                    logger.error("Failed to add evaluation record for %s: %s", name, exc)
            persist = getattr(eng, "persist_evaluation", None)
            if callable(persist):
                try:
                    persist(res)
                except Exception as exc:  # pragma: no cover - persistence failures
                    logger.error("Failed to persist evaluation for %s: %s", name, exc)
            score = float(res.get("cv_score", 0.0))
            if score > self._best_score:
                self._best_score = score
                self._best_name = name
        return results

    # --------------------------------------------------------------
    def best_engine(self) -> Optional[object]:
        """Return the engine with the highest observed ``cv_score``."""
        if self._best_name is None:
            return None
        return self.engines.get(self._best_name)


class ErrorClusterScheduler:
    """Run error cluster analysis at fixed intervals."""

    def __init__(
        self,
        analyzer: ErrorClusterAnalyzer,
        predictor: ErrorClusterPredictor,
        db: ErrorDB,
        graph: KnowledgeGraph,
        event_bus: UnifiedEventBus,
        *,
        interval: float = 300.0,
    ) -> None:
        self.analyzer = analyzer
        self.predictor = predictor
        self.db = db
        self.graph = graph
        self.event_bus = event_bus
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                clusters = self.analyzer.analyze()
                if clusters:
                    mapping = {
                        etype: cid
                        for cid, errors in clusters.items()
                        for etype in errors
                    }
                    self.db.set_error_clusters(mapping)
                # ensure graph stats stay current
                self.graph.update_error_stats(self.db)
                high_risk = self.predictor.predict_high_risk_modules()
                payload = {
                    "clusters": clusters,
                    "high_risk_modules": high_risk,
                }
                self.event_bus.publish("errors:cluster_update", payload)
            except Exception:  # pragma: no cover - background loop
                logger.exception("cluster scheduler iteration failed")
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.1)


__all__ = ["EvaluationManager", "ErrorClusterScheduler"]
