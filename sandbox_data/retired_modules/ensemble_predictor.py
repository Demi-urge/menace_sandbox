from __future__ import annotations

"""Combine multiple ``EvolutionPredictor`` models for more robust forecasts."""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .evolution_predictor import EvolutionPredictor


@dataclass
class EnsemblePrediction:
    """Mean ROI prediction with variance."""

    action: str
    expected_roi: float
    variance: float


class EnsemblePredictor:
    """Maintain several ``EvolutionPredictor`` instances and aggregate predictions."""

    def __init__(
        self,
        count: int = 3,
        *,
        history_db: object | None = None,
        data_bot: object | None = None,
        capital_bot: object | None = None,
    ) -> None:
        self.predictors: List[EvolutionPredictor] = [
            EvolutionPredictor(history_db=history_db, data_bot=data_bot, capital_bot=capital_bot)
            for _ in range(max(1, count))
        ]

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Retrain all ensemble members."""
        for p in self.predictors:
            try:
                p.train()
            except Exception:
                continue

    def predict(self, action: str, before_metric: float) -> Tuple[float, float]:
        """Return mean and variance of ROI predictions."""
        preds: List[float] = []
        for p in self.predictors:
            try:
                preds.append(float(p.predict(action, before_metric)))
            except Exception:
                continue
        if not preds:
            return 0.0, 0.0
        mean = sum(preds) / len(preds)
        var = sum((x - mean) ** 2 for x in preds) / len(preds)
        return mean, var


__all__ = ["EnsemblePrediction", "EnsemblePredictor"]
