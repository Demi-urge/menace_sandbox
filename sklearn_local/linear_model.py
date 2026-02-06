"""Minimal linear_model stubs used by sandbox workflows."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

__all__ = ["LogisticRegression"]


class LogisticRegression:
    """Barebones LogisticRegression classifier using a majority-class baseline."""

    def __init__(self, *args, **kwargs) -> None:
        self._majority_class = None

    def fit(self, _X: Iterable, y: Iterable) -> "LogisticRegression":
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
