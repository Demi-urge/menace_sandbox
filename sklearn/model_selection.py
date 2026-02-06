"""Minimal model_selection stubs used by ROI tracking."""
from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence, Tuple, TypeVar


class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: Sequence, y: Sequence | None = None) -> Iterable[Tuple[List[int], List[int]]]:
        n_samples = len(X)
        indices = list(range(n_samples))
        if self.shuffle:
            rng = random.Random(self.random_state)
            rng.shuffle(indices)
        fold_size = max(1, n_samples // self.n_splits)
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size
            test_idx = indices[start:end]
            train_idx = indices[:start] + indices[end:]
            if not test_idx:
                break
            yield train_idx, test_idx


def cross_val_score(
    estimator: object,
    X: Sequence,
    y: Sequence,
    cv: KFold,
    scoring: str | None = None,
) -> List[float]:
    scores: List[float] = []
    for train_idx, test_idx in cv.split(X, y):
        model = _clone_estimator(estimator)
        X_train = _index_data(X, train_idx)
        y_train = _index_data(y, train_idx)
        X_test = _index_data(X, test_idx)
        y_test = _index_data(y, test_idx)
        model.fit(X_train, y_train)
        score = _score_estimator(model, X_test, y_test, scoring)
        scores.append(float(score))
    return scores


T = TypeVar("T")


def _index_data(data: T, indices: List[int]) -> T:
    try:
        return data[indices]  # type: ignore[index]
    except (TypeError, KeyError):
        return [data[i] for i in indices]  # type: ignore[return-value]


def _clone_estimator(estimator: object) -> object:
    try:
        return estimator.__class__(**estimator.get_params())  # type: ignore[call-arg]
    except Exception:
        try:
            import pickle

            return pickle.loads(pickle.dumps(estimator))
        except Exception:
            return estimator


def _score_estimator(estimator: object, X_test: Sequence, y_test: Sequence, scoring: str | None) -> float:
    if scoring is None:
        return float(estimator.score(X_test, y_test))
    if scoring == "r2":
        return float(estimator.score(X_test, y_test))
    preds = estimator.predict(X_test)
    if scoring == "neg_mean_absolute_error":
        errors = [abs(float(p) - float(t)) for p, t in zip(preds, y_test)]
        return -float(sum(errors) / max(1, len(errors)))
    if scoring == "accuracy":
        correct = [int(p == t) for p, t in zip(preds, y_test)]
        return float(sum(correct) / max(1, len(correct)))
    return float(estimator.score(X_test, y_test))


class GridSearchCV:
    def __init__(
        self,
        estimator: object,
        param_grid: dict,
        cv: int | KFold = 5,
        scoring: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_: dict | None = None
        self.best_score_: float | None = None
        self.best_estimator_: object | None = None

    def _iter_param_grid(self) -> Iterable[dict]:
        if not self.param_grid:
            yield {}
            return
        items = sorted(self.param_grid.items())
        keys = [k for k, _ in items]
        values = [v if isinstance(v, list) else [v] for _, v in items]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def fit(self, X: Sequence, y: Sequence) -> "GridSearchCV":
        cv = self.cv if isinstance(self.cv, KFold) else KFold(n_splits=int(self.cv))
        best_score = float("-inf")
        best_params: dict | None = None
        best_estimator: object | None = None

        for params in self._iter_param_grid():
            estimator = _clone_estimator(self.estimator)
            if hasattr(estimator, "set_params"):
                estimator.set_params(**params)
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=self.scoring)
            if not scores:
                continue
            mean_score = float(sum(scores) / len(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = dict(params)
                best_estimator = estimator

        if best_estimator is None:
            best_estimator = _clone_estimator(self.estimator)
            best_estimator.fit(X, y)
            best_score = _score_estimator(best_estimator, X, y, self.scoring)
            best_params = {}
        else:
            best_estimator.fit(X, y)

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_estimator
        return self


def train_test_split(
    X: Sequence | Iterable,
    y: Sequence | Iterable,
    *,
    test_size: float | int = 0.25,
    random_state: int | None = None,
    shuffle: bool = True,
) -> Tuple[Sequence, Sequence, Sequence, Sequence]:
    X_seq = X if isinstance(X, Sequence) else list(X)
    y_seq = y if isinstance(y, Sequence) else list(y)
    n_samples = len(X_seq)
    if n_samples != len(y_seq):
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
    if shuffle:
        rng = random.Random(random_state)
        rng.shuffle(indices)

    split_index = n_samples - n_test
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = _index_data(X_seq, train_indices)
    X_test = _index_data(X_seq, test_indices)
    y_train = _index_data(y_seq, train_indices)
    y_test = _index_data(y_seq, test_indices)

    return X_train, X_test, y_train, y_test
