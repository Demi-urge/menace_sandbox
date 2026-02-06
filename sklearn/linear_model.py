"""Minimal linear_model stubs used by ROI tracking."""
from __future__ import annotations

from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy unavailable
    np = None  # type: ignore[assignment]

__all__ = ["LinearRegression", "SGDRegressor"]


class LinearRegression:
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self._mean: float | List[float] = 0.0
        self.coef_: Optional["np.ndarray"] = None
        self.intercept_: Optional["np.ndarray"] = None

    def fit(self, X: Iterable, y: Iterable[float]) -> "LinearRegression":
        if np is None:
            return self._fit_mean_baseline(y)

        X_arr = np.asarray(list(X), dtype=float)
        y_arr = np.asarray(list(y), dtype=float)
        if X_arr.size == 0 or y_arr.size == 0:
            return self._fit_mean_baseline(y_arr)

        X_arr = np.atleast_2d(X_arr)
        if self.fit_intercept:
            bias = np.ones((X_arr.shape[0], 1), dtype=float)
            X_design = np.hstack([X_arr, bias])
        else:
            X_design = X_arr

        coef, *_ = np.linalg.lstsq(X_design, y_arr, rcond=None)
        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = np.zeros_like(coef[:1])
        return self

    def predict(self, X: Iterable) -> "np.ndarray | List[float]":
        rows = list(X)
        if np is None or self.coef_ is None:
            return self._predict_mean_baseline(rows)
        X_arr = np.asarray(rows, dtype=float)
        X_arr = np.atleast_2d(X_arr)
        preds = X_arr @ self.coef_
        if self.intercept_ is not None:
            preds = preds + self.intercept_
        return np.asarray(preds)

    def _fit_mean_baseline(self, y: Iterable[float]) -> "LinearRegression":
        data = list(y)
        if data and isinstance(data[0], (list, tuple)):
            columns = list(zip(*data))
            self._mean = [sum(col) / len(col) for col in columns]
        else:
            self._mean = sum(data) / len(data) if data else 0.0
        return self

    def _predict_mean_baseline(self, rows: Iterable) -> List[float]:
        baseline = self._mean
        if isinstance(baseline, list):
            return [list(baseline) for _ in rows]
        return [baseline for _ in rows]

    def get_params(self, deep: bool = True) -> dict:
        return {"fit_intercept": self.fit_intercept}


class SGDRegressor(LinearRegression):
    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        fit_intercept: bool = True,
        **_kwargs: object,
    ) -> None:
        super().__init__(fit_intercept=fit_intercept)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict:
        return {
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "fit_intercept": self.fit_intercept,
        }
