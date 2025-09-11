from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import logging

logger = logging.getLogger(__name__)
import math
from typing import Iterable, List

import numpy as np

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .prediction_manager_bot import PredictionManager


@self_coding_managed
class EnergyForecastBot:
    """Predict near-term energy score from basic metrics."""

    prediction_profile = {"scope": ["energy"], "risk": ["medium"]}

    def __init__(
        self,
        *,
        history_limit: int = 20,
        data_bot: DataBot | None = None,
        capital_bot: CapitalManagementBot | None = None,
        prediction_manager: PredictionManager | None = None,
    ) -> None:
        self.history_limit = history_limit
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.prediction_manager = prediction_manager
        self.history: List[List[float]] = []
        self.targets: List[float] = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EnergyForecastBot")
        self.last_training_exception: Exception | None = None
        self.last_prediction_exception: Exception | None = None
        try:
            from sklearn.linear_model import LinearRegression  # type: ignore

            self.model: LinearRegression | None = LinearRegression()
        except Exception:  # pragma: no cover - optional dependency
            class _SimpleLinearRegression:
                """Minimal ordinary least squares implementation."""

                def __init__(self) -> None:
                    self.coef_: np.ndarray | None = None
                    self.intercept_: float = 0.0

                def fit(self, X: np.ndarray, y: np.ndarray) -> None:
                    X = np.asarray(X, dtype=float)
                    y = np.asarray(y, dtype=float)
                    X_b = np.c_[np.ones(len(X)), X]
                    beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
                    self.intercept_ = float(beta[0])
                    self.coef_ = beta[1:]

                def predict(self, X: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
                    arr = np.asarray(list(X), dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return arr @ (self.coef_ or np.zeros(arr.shape[1])) + self.intercept_

            self.model = _SimpleLinearRegression()
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)

    # ------------------------------------------------------------------
    def learn(self, feats: Iterable[float], target: float) -> None:
        """Store a training example and update the model."""
        vec = list(feats)
        self.history.append(vec)
        self.targets.append(target)
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]
            self.targets = self.targets[-self.history_limit :]
        if self.model and len(self.targets) >= 5:
            try:
                self.model.fit(np.array(self.history), np.array(self.targets))
                self.last_training_exception = None
            except Exception as exc:
                self.last_training_exception = exc
                self.logger.exception(
                    "training failed for input %s -> %s", vec, target
                )

    def predict(self, feats: Iterable[float]) -> float:
        """Return an energy estimate in the ``[0,1]`` range."""
        vec = list(feats)
        base = sum(vec) / len(vec) if vec else 0.0
        if self.model and len(self.targets) >= 5:
            try:
                base = float(self.model.predict([vec])[0])
                self.last_prediction_exception = None
            except Exception as exc:
                self.last_prediction_exception = exc
                self.logger.exception("prediction failed for input %s", vec)
        return 1.0 / (1.0 + math.exp(-base))


__all__ = ["EnergyForecastBot"]
