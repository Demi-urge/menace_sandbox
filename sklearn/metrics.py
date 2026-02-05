"""Minimal metrics utilities for sandbox usage."""
from __future__ import annotations

from typing import Iterable

__all__ = ["accuracy_score"]


def accuracy_score(y_true: Iterable, y_pred: Iterable) -> float:
    """Return the fraction of equal elements in ``y_true`` and ``y_pred``."""
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")

    if not true_list:
        return 0.0

    matches = sum(1 for true, pred in zip(true_list, pred_list) if true == pred)
    return matches / len(true_list)
