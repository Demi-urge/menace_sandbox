"""Minimal linear_model stubs used by ROI tracking."""
from __future__ import annotations

from typing import Iterable, List


class LinearRegression:
    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, _X: Iterable, y: Iterable[float]) -> "LinearRegression":
        data = list(y)
        self._mean = sum(data) / len(data) if data else 0.0
        return self

    def predict(self, X: Iterable) -> List[float]:
        rows = list(X)
        return [self._mean for _ in rows]
