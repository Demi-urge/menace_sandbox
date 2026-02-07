"""Minimal pairwise metrics stubs for local usage."""
from __future__ import annotations

from typing import Iterable, Sequence

import importlib.util

if importlib.util.find_spec("numpy") is not None:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
else:  # pragma: no cover - numpy unavailable
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import sklearn_local.metrics as _local_metrics  # type: ignore
except Exception:  # pragma: no cover - missing local metrics
    _local_metrics = None

__all__ = ["cosine_similarity"]


def _as_2d(data: Iterable) -> list[list[float]]:
    rows = list(data)
    if not rows:
        return []
    if isinstance(rows[0], (list, tuple)):
        return [list(map(float, row)) for row in rows]
    return [list(map(float, rows))]


def cosine_similarity(X: Iterable, Y: Iterable | None = None) -> list[list[float]]:
    """Compute cosine similarity between rows of X and Y."""
    if _local_metrics is not None:
        local = getattr(_local_metrics, "cosine_similarity", None)
        if local is not None:
            return local(X, Y)

    X_rows = _as_2d(X)
    Y_rows = _as_2d(X if Y is None else Y)

    if np is not None:
        X_arr = np.asarray(X_rows, dtype=float)
        Y_arr = np.asarray(Y_rows, dtype=float)
        if X_arr.size == 0 or Y_arr.size == 0:
            return np.zeros((len(X_rows), len(Y_rows)), dtype=float).tolist()
        X_norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
        Y_norms = np.linalg.norm(Y_arr, axis=1, keepdims=True)
        denom = X_norms @ Y_norms.T
        denom = np.where(denom == 0, 1.0, denom)
        sims = (X_arr @ Y_arr.T) / denom
        return sims.tolist()

    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(float(x) * float(y) for x, y in zip(a, b))

    def _norm(a: Sequence[float]) -> float:
        return sum(float(x) ** 2 for x in a) ** 0.5

    result: list[list[float]] = []
    for row in X_rows:
        row_norm = _norm(row)
        row_scores = []
        for other in Y_rows:
            denom = row_norm * _norm(other)
            score = 0.0 if denom == 0 else _dot(row, other) / denom
            row_scores.append(score)
        result.append(row_scores)
    return result
