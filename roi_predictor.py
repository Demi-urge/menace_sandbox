from __future__ import annotations

"""Optional gradient boosting predictor for ROI values."""

from typing import Iterable, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class ROIPredictor:
    """Forecast future ROI values using gradient boosting."""

    def __init__(self) -> None:
        self._model = GradientBoostingRegressor(random_state=0)

    def forecast(
        self,
        history: Iterable[float],
        exog: np.ndarray | None = None,
    ) -> Tuple[float, Tuple[float, float]]:
        """Return next ROI prediction and naive confidence interval."""
        hist = [float(h) for h in history]
        if not hist:
            return 0.0, (0.0, 0.0)
        if len(hist) < 2:
            val = hist[-1]
            return val, (val, val)

        X = np.arange(len(hist)).reshape(-1, 1)
        if exog is not None and exog.shape[0] >= len(hist):
            X = np.hstack([X, exog[-len(hist) :]])
            next_row = np.hstack([[len(hist)], exog[-1]]).reshape(1, -1)
        else:
            next_row = np.array([[len(hist)]])

        y = np.array(hist)
        self._model.fit(X, y)
        mean = float(self._model.predict(next_row)[0])
        resid = y - self._model.predict(X)
        se = float(resid.std(ddof=1)) if resid.size > 1 else 0.0
        delta = 1.96 * se
        return mean, (mean - delta, mean + delta)


__all__ = ["ROIPredictor"]
