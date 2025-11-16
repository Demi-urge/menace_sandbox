from __future__ import annotations

"""Rank trained models and redeploy the top performer."""

import logging
from typing import Optional, Dict, TYPE_CHECKING

from .neuroplasticity import PathwayDB
if TYPE_CHECKING:  # pragma: no cover - avoid heavy import at runtime
    from .advanced_error_management import AutomatedRollbackManager
from .evaluation_history_db import EvaluationHistoryDB
from .model_deployer import ModelDeployer


class CrossModelComparator:
    """Compare model metrics and redeploy the winner."""

    def __init__(
        self,
        pathways: PathwayDB,
        history: EvaluationHistoryDB,
        deployer: Optional[ModelDeployer] = None,
        rollback_mgr: "AutomatedRollbackManager" | None = None,
    ) -> None:
        self.pathways = pathways
        self.history = history
        self.deployer = deployer
        self.rollback_mgr = rollback_mgr
        self.logger = logging.getLogger("CrossModelComparator")
        self._last_best: Optional[str] = None

    # ------------------------------------------------------------------
    def aggregated_weights(self) -> Dict[str, float]:
        """Return deployment weights aggregated across all nodes."""
        return self.history.deployment_weights()

    def best_model(self) -> Optional[str]:
        weights = self.aggregated_weights()
        if not weights:
            return None

        scores: Dict[str, float] = {}
        path_max = 0.0
        if self.pathways:
            try:
                path_max = self.pathways.highest_myelination_score()
            except Exception:
                path_max = 0.0
            if path_max <= 0.0:
                path_max = 1.0
        for name, w in weights.items():
            p_w = 0.0
            if self.pathways:
                try:
                    sim = self.pathways.similar_actions(
                        f"run_cycle:{name}", limit=1
                    )
                    if sim:
                        p_w = float(sim[0][1]) / path_max
                except Exception:
                    p_w = 0.0
            scores[name] = w + p_w
        return max(scores, key=scores.get)

    def rank_and_deploy(self) -> Optional[str]:
        best = self.best_model()
        if best and best != self._last_best and self.deployer:
            if hasattr(self.deployer, "clone_model"):
                try:
                    self.deployer.clone_model(best)
                except Exception:
                    self.logger.exception("clone failed")
            if hasattr(self.deployer, "deploy_model"):
                try:
                    self.deployer.deploy_model(best)
                except Exception:
                    self.logger.exception("deploy failed")
            self._last_best = best
        return best

    # ------------------------------------------------------------------
    def evaluate_and_rollback(self, threshold: float = 0.5) -> Optional[str]:
        """Rollback if current best drops below ``threshold``"""
        best = self.rank_and_deploy()
        if not best or not self.rollback_mgr:
            return best
        weights = self.history.deployment_weights()
        if weights.get(best, 1.0) < threshold:
            if hasattr(self.rollback_mgr, "auto_rollback"):
                try:
                    self.rollback_mgr.auto_rollback("latest", [best])
                except Exception:
                    self.logger.exception("auto rollback failed")
        return best


__all__ = ["CrossModelComparator"]

