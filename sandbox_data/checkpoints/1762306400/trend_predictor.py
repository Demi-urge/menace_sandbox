from __future__ import annotations

"""Forecast ROI and error rate trends."""

from dataclasses import dataclass
from typing import List, Tuple

from .evolution_history_db import EvolutionHistoryDB
from .data_bot import MetricsDB

try:  # optional dependencies
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore


@dataclass
class TrendPrediction:
    """Predicted metrics over a future window."""

    roi: float
    errors: float


class TrendPredictor:
    """Train simple time series models on ROI and error history."""

    def __init__(
        self,
        history_db: EvolutionHistoryDB | None = None,
        metrics_db: MetricsDB | None = None,
    ) -> None:
        self.history = history_db or EvolutionHistoryDB()
        self.metrics = metrics_db or MetricsDB()
        self._roi_model: object | None = None
        self._roi_len = 0
        self._err_model: object | None = None
        self._err_len = 0

    # ------------------------------------------------------------------
    def _fit_model(self, series: List[float]) -> Tuple[object | None, int]:
        if len(series) < 2:
            return None, 0
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(series, order=(1, 1, 1)).fit()
            return model, len(series)
        except Exception:
            try:
                from sklearn.linear_model import LinearRegression
                import numpy as np

                X = np.arange(len(series)).reshape(-1, 1)
                y = np.array(series)
                model = LinearRegression().fit(X, y)
                model._is_lr = True  # type: ignore[attr-defined]
                return model, len(series)
            except Exception:
                return None, 0

    @staticmethod
    def _forecast(model: object | None, length: int, steps: int) -> float:
        if model is None:
            return 0.0
        try:
            if getattr(model, "_is_lr", False):
                import numpy as np

                X = np.array([[length + steps - 1]])
                return float(model.predict(X)[0])
            return float(model.forecast(steps)[-1])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def train(self, limit: int = 50) -> None:
        """Fit models from the latest metrics and evolution history."""
        roi_series: List[float] = []
        err_series: List[float] = []
        if pd is None:
            rows = self.metrics.fetch(limit)
            if isinstance(rows, list):
                roi_series.extend(
                    float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in rows
                )
                err_series.extend(float(r.get("errors", 0.0)) for r in rows)
        else:
            df = self.metrics.fetch(limit)
            if not getattr(df, "empty", True):
                df["roi"] = df["revenue"] - df["expense"]
                roi_series.extend([float(v) for v in df["roi"].tolist()])
                err_series.extend([float(v) for v in df["errors"].tolist()])
        hist = self.history.fetch(limit)
        for row in hist:
            roi_series.append(float(row[3]))
            err_series.append(float(row[6]))
        self._roi_model, self._roi_len = self._fit_model(roi_series)
        self._err_model, self._err_len = self._fit_model(err_series)

    def predict_future_metrics(self, cycles: int = 1) -> TrendPrediction:
        """Return expected ROI and error rate for ``cycles`` ahead."""
        if self._roi_model is None or self._err_model is None:
            self.train()
        roi = self._forecast(self._roi_model, self._roi_len, cycles)
        err = self._forecast(self._err_model, self._err_len, cycles)
        return TrendPrediction(roi=roi, errors=err)


__all__ = ["TrendPrediction", "TrendPredictor"]
