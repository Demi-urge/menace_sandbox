"""Minimal metrics utilities for sandbox usage."""
from __future__ import annotations

from typing import Iterable

__all__ = ["accuracy_score", "mean_squared_error", "r2_score"]


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


def mean_squared_error(y_true: Iterable, y_pred: Iterable) -> float:
    """Return the mean squared error between ``y_true`` and ``y_pred``."""
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")

    if not true_list:
        return 0.0

    squared_errors = [(true - pred) ** 2 for true, pred in zip(true_list, pred_list)]
    return sum(squared_errors) / len(true_list)


def r2_score(y_true: Iterable, y_pred: Iterable) -> float:
    """Return the R^2 (coefficient of determination) score."""
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")

    if not true_list:
        return 0.0

    mean_true = sum(true_list) / len(true_list)
    ss_res = sum((true - pred) ** 2 for true, pred in zip(true_list, pred_list))
    ss_tot = sum((true - mean_true) ** 2 for true in true_list)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)
