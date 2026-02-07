"""Minimal metrics stubs for local usage."""
from __future__ import annotations

from typing import Iterable, Sequence

import importlib.util

if importlib.util.find_spec("numpy") is not None:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
else:  # pragma: no cover - numpy unavailable
    np = None  # type: ignore

__all__ = [
    "accuracy_score",
    "mean_squared_error",
    "r2_score",
    "roc_auc_score",
    "silhouette_score",
]


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


def roc_auc_score(y_true: Sequence, y_score: Sequence) -> float:
    """Compute a simple ROC AUC for binary classification."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    if not y_true:
        return 0.0

    labels = list(y_true)
    scores = list(y_score)
    if len(set(labels)) < 2:
        raise ValueError("roc_auc_score is undefined with a single class present")

    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    pos_count = sum(1 for _, label in pairs if label == 1)
    neg_count = len(pairs) - pos_count
    if pos_count == 0 or neg_count == 0:
        raise ValueError("roc_auc_score is undefined with a single class present")

    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / pos_count
        fpr = fp / neg_count
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)


def silhouette_score(X: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Compute a basic silhouette score using Euclidean distance."""
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same length")

    if len(X) < 2:
        return 0.0

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return 0.0

    clusters = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        if np is not None:
            diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
            return float(np.sqrt(np.sum(diff * diff)))
        return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5

    scores = []
    for idx, label in enumerate(labels):
        same_cluster = [i for i in clusters[label] if i != idx]
        if same_cluster:
            a = sum(_distance(X[idx], X[i]) for i in same_cluster) / len(same_cluster)
        else:
            a = 0.0

        b = None
        for other_label, indices in clusters.items():
            if other_label == label or not indices:
                continue
            dist = sum(_distance(X[idx], X[i]) for i in indices) / len(indices)
            if b is None or dist < b:
                b = dist

        if b is None:
            scores.append(0.0)
            continue

        denom = max(a, b)
        score = 0.0 if denom == 0 else (b - a) / denom
        scores.append(score)

    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))
