"""Minimal preprocessing stubs used by ROI tracking."""
from __future__ import annotations

from typing import Iterable, List


class PolynomialFeatures:
    def __init__(self, degree: int = 2) -> None:
        self.degree = degree

    def fit_transform(self, X: Iterable) -> List[List[float]]:
        return [self._expand(row) for row in X]

    def transform(self, X: Iterable) -> List[List[float]]:
        return [self._expand(row) for row in X]

    def _expand(self, row: object) -> List[float]:
        if isinstance(row, (list, tuple)) and row:
            value = float(row[0])
        else:
            value = float(row)
        expanded = [1.0]
        for power in range(1, self.degree + 1):
            expanded.append(value ** power)
        return expanded
