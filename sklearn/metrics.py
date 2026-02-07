"""Shim module for ``sklearn.metrics`` backed by local implementations."""
from __future__ import annotations

from sklearn_local.metrics import accuracy_score, mean_squared_error, r2_score

__all__ = ["accuracy_score", "mean_squared_error", "r2_score"]
