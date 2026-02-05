"""Minimal model_selection utilities for sandbox usage."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, TypeVar

import numpy as np

__all__ = ["train_test_split"]

T = TypeVar("T")


def _as_sequence(data: Sequence | Iterable) -> Sequence:
    return data if isinstance(data, Sequence) else list(data)


def _index_data(data: T, indices: list[int]) -> T:
    try:
        return data[indices]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        return [data[i] for i in indices]  # type: ignore[return-value]


def _normalize_test_size(test_size: float | int, n_samples: int) -> int:
    if n_samples <= 0:
        return 0

    if isinstance(test_size, float):
        if test_size <= 0:
            n_test = 0
        else:
            n_test = int(round(n_samples * test_size))
    else:
        n_test = int(test_size)

    if n_samples > 1:
        return max(1, min(n_test, n_samples - 1))
    return min(n_test, n_samples)


def train_test_split(
    X: Sequence | Iterable,
    y: Sequence | Iterable,
    test_size: float | int = 0.25,
    random_state: int | None = None,
    shuffle: bool = True,
) -> Tuple[Sequence, Sequence, Sequence, Sequence]:
    X_seq = _as_sequence(X)
    y_seq = _as_sequence(y)
    n_samples = len(X_seq)
    if n_samples != len(y_seq):
        raise ValueError("X and y must contain the same number of samples")

    n_test = _normalize_test_size(test_size, n_samples)
    indices = list(range(n_samples))

    if shuffle and n_samples > 1:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        indices = rng.permutation(indices).tolist()

    split_index = n_samples - n_test
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = _index_data(X_seq, train_indices)
    X_test = _index_data(X_seq, test_indices)
    y_train = _index_data(y_seq, train_indices)
    y_test = _index_data(y_seq, test_indices)

    return X_train, X_test, y_train, y_test
