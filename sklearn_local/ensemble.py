"""Minimal ensemble stubs used by sandbox workflows."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

__all__ = ["RandomForestClassifier", "GradientBoostingRegressor"]


class RandomForestClassifier:
    """Barebones RandomForestClassifier using a majority-class baseline."""

    def __init__(self, *args, **kwargs) -> None:
        self._majority_class = None

    def fit(self, _X: Iterable, y: Iterable) -> "RandomForestClassifier":
        labels = list(y)
        if labels:
            counts = Counter(labels)
            self._majority_class = counts.most_common(1)[0][0]
        else:
            self._majority_class = None
        return self

    def predict(self, X: Iterable) -> List:
        rows = list(X)
        return [self._majority_class for _ in rows]


class GradientBoostingRegressor:
    """Barebones GradientBoostingRegressor using a mean-target baseline."""

    def __init__(self, random_state: int | None = None, *args, **kwargs) -> None:
        self.random_state = random_state
        self._baseline = 0.0
        self.baseline_ = 0.0

    def fit(self, _X: Iterable, y: Iterable) -> "GradientBoostingRegressor":
        values = list(y)
        if values:
            try:
                self._baseline = float(sum(values) / len(values))
            except TypeError:
                self._baseline = 0.0
        else:
            self._baseline = 0.0
        self.baseline_ = self._baseline
        return self

    def predict(self, X: Iterable) -> List[float]:
        rows = list(X)
        return [self._baseline for _ in rows]
