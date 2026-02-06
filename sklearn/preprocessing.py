"""Minimal preprocessing stubs used by ROI tracking."""
from __future__ import annotations

from typing import Iterable, List, Sequence


class PolynomialFeatures:
    def __init__(self, degree: int = 2) -> None:
        self.degree = degree
        self.n_features_in_: int | None = None

    def fit(self, X: Iterable) -> "PolynomialFeatures":
        rows = self._normalize_rows(X)
        self.n_features_in_ = len(rows[0]) if rows else 0
        return self

    def fit_transform(self, X: Iterable) -> List[List[float]]:
        self.fit(X)
        return self.transform(X)

    def transform(self, X: Iterable) -> List[List[float]]:
        rows = self._normalize_rows(X)
        return [self._expand_row(row) for row in rows]

    def _normalize_rows(self, X: Iterable) -> List[List[float]]:
        if isinstance(X, (int, float)):
            return [[float(X)]]
        if isinstance(X, Sequence) and X:
            first = X[0]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                return [self._to_float_list(row) for row in X]
            return [[float(value)] for value in X]
        return []

    def _to_float_list(self, row: Iterable) -> List[float]:
        return [float(value) for value in row]

    def _expand_row(self, row: Sequence[float]) -> List[float]:
        expanded = [1.0]
        for value in row:
            for power in range(1, self.degree + 1):
                expanded.append(value ** power)
        return expanded
