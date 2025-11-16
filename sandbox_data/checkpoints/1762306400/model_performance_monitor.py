from __future__ import annotations

"""Monitor models across the cluster and retire underperformers."""

import logging
from typing import Iterable, Optional

from .cross_model_comparator import CrossModelComparator
from .evaluation_history_db import EvaluationHistoryDB
from .model_deployer import ModelDeployer


class ModelPerformanceMonitor:
    """Continuously compare models and drop poor performers."""

    def __init__(
        self,
        comparator: CrossModelComparator,
        history: EvaluationHistoryDB,
        deployer: ModelDeployer | None = None,
        *,
        threshold: float = 0.2,
    ) -> None:
        self.comparator = comparator
        self.history = history
        self.deployer = deployer or ModelDeployer()
        self.threshold = threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def evaluate(self) -> Optional[str]:
        best = self.comparator.rank_and_deploy()
        if not best:
            return None
        weights = self.history.deployment_weights()
        for name, score in weights.items():
            if name != best and score < self.threshold:
                self.logger.info("retiring underperforming model %s", name)
                try:
                    self.deployer.retire_model(name)
                except Exception:
                    self.logger.exception("retire failed")
        return best


__all__ = ["ModelPerformanceMonitor"]
