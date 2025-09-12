from __future__ import annotations

import math
from statistics import NormalDist
from typing import List, Tuple

try:  # optional dependency
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except Exception:  # pragma: no cover - optional
    ARIMA = None  # type: ignore


class ForecastModel:
    """Base interface for forecast models."""

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def fit(self, history: List[float]) -> "ForecastModel":  # pragma: no cover - interface
        raise NotImplementedError

    def forecast(self) -> Tuple[float, float, float]:  # pragma: no cover - interface
        raise NotImplementedError


class SimpleLinearForecast(ForecastModel):
    """Linear regression forecaster with configurable confidence interval."""

    def __init__(self, confidence: float = 0.95) -> None:
        super().__init__(confidence)
        self.history: List[float] = []
        self.n = 0
        self.slope = 0.0
        self.intercept = 0.0
        self.std_err = 0.0

    def fit(self, history: List[float]) -> "SimpleLinearForecast":
        self.history = [float(h) for h in history]
        self.n = len(self.history)
        if self.n >= 2:
            x_vals = list(range(self.n))
            x_mean = sum(x_vals) / self.n
            y_mean = sum(self.history) / self.n
            s_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, self.history))
            s_xx = sum((x - x_mean) ** 2 for x in x_vals)
            self.slope = s_xy / s_xx if s_xx else 0.0
            self.intercept = y_mean - self.slope * x_mean
            residuals = [
                y - (self.slope * x + self.intercept)
                for x, y in zip(x_vals, self.history)
            ]
            dof = max(self.n - 2, 1)
            self.std_err = math.sqrt(sum(r * r for r in residuals) / dof)
        elif self.n == 1:
            self.intercept = self.history[0]
            self.slope = 0.0
            self.std_err = 0.0
        else:
            self.intercept = 0.0
            self.slope = 0.0
            self.std_err = 0.0
        return self

    def forecast(self) -> Tuple[float, float, float]:
        if self.n == 0:
            return 0.0, 0.0, 0.0
        x_pred = self.n
        pred = self.slope * x_pred + self.intercept
        if self.n < 2:
            return pred, pred, pred
        x_vals = list(range(self.n))
        x_mean = sum(x_vals) / self.n
        s_xx = sum((x - x_mean) ** 2 for x in x_vals)
        if s_xx == 0:
            se_pred = self.std_err
        else:
            se_pred = self.std_err * math.sqrt(1 + 1 / self.n + (x_pred - x_mean) ** 2 / s_xx)
        z = NormalDist().inv_cdf((1 + self.confidence) / 2)
        ci = z * se_pred
        return pred, pred - ci, pred + ci


class ExponentialSmoothingModel(ForecastModel):
    """Simple exponential smoothing forecaster."""

    def __init__(self, alpha: float = 0.5, confidence: float = 0.95) -> None:
        super().__init__(confidence)
        self.alpha = alpha
        self.level = 0.0
        self.var = 0.0
        self._fit = False

    def fit(self, history: List[float]) -> "ExponentialSmoothingModel":
        if not history:
            self.level = 0.0
            self.var = 0.0
            self._fit = False
            return self
        self.level = float(history[0])
        residuals: List[float] = []
        level = self.level
        for value in history[1:]:
            prev_level = level
            level = self.alpha * value + (1 - self.alpha) * level
            residuals.append(value - prev_level)
        self.level = level
        if residuals:
            self.var = sum(r * r for r in residuals) / len(residuals)
        else:
            self.var = 0.0
        self._fit = True
        return self

    def forecast(self) -> Tuple[float, float, float]:
        if not self._fit:
            return 0.0, 0.0, 0.0
        pred = self.level
        z = NormalDist().inv_cdf((1 + self.confidence) / 2)
        std = math.sqrt(self.var)
        ci = z * std
        return pred, pred - ci, pred + ci


class ARIMAForecastModel(ForecastModel):
    """ARIMA forecaster using statsmodels (if available)."""

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0), confidence: float = 0.95) -> None:
        super().__init__(confidence)
        self.order = order
        self._result = None
        self._history: List[float] = []

    def fit(self, history: List[float]) -> "ARIMAForecastModel":
        self._history = [float(h) for h in history]
        if ARIMA is None or len(self._history) < sum(self.order):  # pragma: no cover - optional
            self._result = None
            return self
        model = ARIMA(self._history, order=self.order)
        try:
            self._result = model.fit()
        except Exception:  # pragma: no cover - statsmodels can fail on bad data
            self._result = None
        return self

    def forecast(self) -> Tuple[float, float, float]:
        if self._result is None:
            if not self._history:
                return 0.0, 0.0, 0.0
            pred = self._history[-1]
            return pred, pred, pred
        forecast = self._result.get_forecast()
        pred = float(forecast.predicted_mean[-1])
        try:
            conf_int = forecast.conf_int(alpha=1 - self.confidence)
            low = float(conf_int[-1, 0])
            high = float(conf_int[-1, 1])
        except Exception:  # pragma: no cover
            low = high = pred
        return pred, low, high


def create_model(name: str, confidence: float = 0.95) -> ForecastModel:
    name = (name or "").lower()
    if name in {"exponential", "holt"}:
        return ExponentialSmoothingModel(confidence=confidence)
    if name == "arima" and ARIMA is not None:
        return ARIMAForecastModel(confidence=confidence)
    # fallback to simple linear
    return SimpleLinearForecast(confidence=confidence)


__all__ = [
    "ForecastModel",
    "SimpleLinearForecast",
    "ExponentialSmoothingModel",
    "ARIMAForecastModel",
    "create_model",
]
