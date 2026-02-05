"""Minimal model_selection stubs used by ROI tracking."""
from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple, TypeVar


class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: Sequence) -> Iterable[Tuple[List[int], List[int]]]:
        n_samples = len(X)
        fold_size = max(1, n_samples // self.n_splits)
        indices = list(range(n_samples))
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size
            test_idx = indices[start:end]
            train_idx = indices[:start] + indices[end:]
            if not test_idx:
                break
            yield train_idx, test_idx


def cross_val_score(_estimator: object, X: Sequence, y: Sequence, cv: KFold, scoring: str | None = None) -> List[float]:
    return [0.0 for _ in range(cv.n_splits)]


T = TypeVar("T")


def _index_data(data: T, indices: List[int]) -> T:
    try:
        return data[indices]  # type: ignore[index]
    except (TypeError, KeyError):
        return [data[i] for i in indices]  # type: ignore[return-value]


def train_test_split(
    X: Sequence,
    y: Sequence,
    *,
    test_size: float | int = 0.25,
    random_state: int | None = None,
) -> Tuple[Sequence, Sequence, Sequence, Sequence]:
    n_samples = len(X)
    if n_samples != len(y):
        raise ValueError("X and y must contain the same number of samples")

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1 when a float")
        n_test = int(round(n_samples * test_size))
    else:
        n_test = int(test_size)

    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("test_size results in an invalid split")

    indices = list(range(n_samples))
    rng = random.Random(random_state)
    rng.shuffle(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = _index_data(X, train_indices)
    X_test = _index_data(X, test_indices)
    y_train = _index_data(y, train_indices)
    y_test = _index_data(y, test_indices)

    return X_train, X_test, y_train, y_test
