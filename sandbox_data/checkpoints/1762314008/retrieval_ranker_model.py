from __future__ import annotations

"""Training routine for the retrieval ranking model.

This module trains a model on data gathered by
:mod:`analytics.retrieval_ranker_dataset`.  The preferred implementation uses
:class:`lightgbm.LGBMRanker` when available and falls back to a logistic
regression model otherwise.  Cross-validation and holdout scores are exported
via :mod:`metrics_exporter` gauges and the final fitted model is persisted to
``analytics/models/retrieval_ranker.pkl``.
"""

from pathlib import Path
from typing import Any, Iterable, Tuple

import joblib
import pandas as pd

from .retrieval_ranker_dataset import build_dataset
from ..metrics_exporter import learning_cv_score, learning_holdout_score
from ..dynamic_path_router import resolve_path

try:  # optional dependency
    from lightgbm import LGBMRanker  # type: ignore
except Exception:  # pragma: no cover - lightgbm is optional
    LGBMRanker = None  # type: ignore

try:  # scikit-learn is optional
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        cross_val_score,
        train_test_split,
    )  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    LogisticRegression = None  # type: ignore
    GroupKFold = None  # type: ignore
    GroupShuffleSplit = None  # type: ignore
    cross_val_score = None  # type: ignore
    train_test_split = None  # type: ignore

from ..learning_engine import _SimpleLogReg

MODEL_DIR = resolve_path("analytics/models")
MODEL_PATH = MODEL_DIR / "retrieval_ranker.pkl"


def _prepare_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return feature matrix, labels and grouping keys for training."""

    df = build_dataset()
    if df.empty:
        raise RuntimeError("training dataset is empty")

    # Label column – "hit" if available, otherwise derive from ``prior_hits``
    if "hit" in df:
        y = df["hit"].astype(int)
    else:
        y = (df.get("prior_hits", 0) > 0).astype(int)

    # Use ``session_id`` as the grouping key for cross-validation.  When the
    # column is missing we still need an index-aligned placeholder series so
    # that downstream splits work reliably.
    groups = df.get("session_id", pd.Series([0] * len(df), index=df.index))

    db_col = "db_type" if "db_type" in df.columns else "origin_db"
    df = pd.get_dummies(df, columns=[db_col], prefix="db", dtype=float)
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("db_")
        or c
        in {
            "similarity",
            "age",
            "exec_freq",
            "roi_delta",
            "prior_hits",
            "alignment_severity",
            "win",
            "regret",
        }
    ]
    X = df[feature_cols].fillna(0.0)
    return X, y, groups


def _score(model: Any, X: Any, y: Iterable[int]) -> float:
    """Return accuracy-like score for *model* on *(X, y)*."""

    try:
        return float(model.score(X, y))  # type: ignore[attr-defined]
    except Exception:
        pass
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if hasattr(proba, "__iter__") and not isinstance(proba, float):
            preds = [1 if p[1] >= 0.5 else 0 for p in proba]
            correct = sum(int(a == b) for a, b in zip(preds, y))
            return correct / max(len(preds), 1)
    return 0.0

def train(save_path: Path | str = MODEL_PATH) -> Any:
    """Train the retrieval ranker and persist the fitted model."""

    X, y, groups = _prepare_dataset()

    if LGBMRanker is not None:
        model: Any = LGBMRanker(random_state=0)
        cv_scores: list[float] = []
        if GroupKFold is not None and groups.nunique() >= 2:
            gkf = GroupKFold(n_splits=min(3, groups.nunique()))
            for tr, te in gkf.split(X, y, groups):
                group_train = groups.iloc[tr].value_counts().tolist()
                m = LGBMRanker(random_state=0)
                m.fit(X.iloc[tr], y.iloc[tr], group=group_train)
                cv_scores.append(_score(m, X.iloc[te], y.iloc[te]))
        cv_score = float(sum(cv_scores) / len(cv_scores)) if cv_scores else 0.0

        holdout_score = cv_score
        if GroupShuffleSplit is not None and groups.nunique() > 1:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
            tr, te = next(gss.split(X, y, groups))
            group_train = groups.iloc[tr].value_counts().tolist()
            m = LGBMRanker(random_state=0)
            m.fit(X.iloc[tr], y.iloc[tr], group=group_train)
            holdout_score = _score(m, X.iloc[te], y.iloc[te])

        full_group = groups.groupby(groups).size().tolist()
        model.fit(X, y, group=full_group)

    else:
        if LogisticRegression is not None:
            model = LogisticRegression(max_iter=1000)
            scores = (
                cross_val_score(model, X, y, cv=3) if cross_val_score else []
            )
            cv_score = float(scores.mean()) if len(scores) else 0.0
            if train_test_split is not None:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=0
                )
                model.fit(X_tr, y_tr)
                holdout_score = _score(model, X_te, y_te)
            else:
                model.fit(X, y)
                holdout_score = cv_score
        else:  # use simple logistic regression fallback
            model = _SimpleLogReg()
            n = len(X)
            k = 3
            indices = list(range(n))
            fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
            current = 0
            scores: list[float] = []
            for fold in fold_sizes:
                start, stop = current, current + fold
                test_idx = indices[start:stop]
                train_idx = indices[:start] + indices[stop:]
                m = _SimpleLogReg()
                m.fit(
                    X.iloc[train_idx].to_numpy().tolist(),
                    y.iloc[train_idx].tolist(),
                )
                score = _score(
                    m,
                    X.iloc[test_idx].to_numpy().tolist(),
                    y.iloc[test_idx].tolist(),
                )
                scores.append(score)
                current = stop
            cv_score = float(sum(scores) / len(scores)) if scores else 0.0
            holdout_score = scores[0] if scores else 0.0
            model.fit(X.to_numpy().tolist(), y.tolist())

    try:
        if learning_cv_score:
            learning_cv_score.set(float(cv_score))
        if learning_holdout_score:
            learning_holdout_score.set(float(holdout_score))
    except Exception:  # pragma: no cover - metrics export is best effort
        pass

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model


def load(model_path: Path | str = MODEL_PATH) -> Any:
    """Load a previously trained retrieval ranker."""

    return joblib.load(model_path)


__all__ = ["MODEL_DIR", "MODEL_PATH", "train", "load"]
