"""Minimal model_selection stubs used by ROI tracking."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


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
