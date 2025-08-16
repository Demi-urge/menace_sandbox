"""Utilities for training a lightweight retrieval ranking model.

This module builds a training dataframe from
``retrieval_training_dataset.build_dataset`` and fits a small binary
classifier.  The preferred implementation uses
``sklearn.linear_model.LogisticRegression``; if that is unavailable the code
falls back to ``lightgbm.LGBMClassifier`` when installed and finally to a very
small NumPy based logistic regression variant.  The resulting model weights are
serialised to JSON so that other components can consume them without a heavy
machine learning dependency.

Example
-------

Retrain the model and store it under ``analytics/retrieval_ranker.model``::

    python retrieval_ranker.py train \
        --vector-db vector_metrics.db \
        --patch-db metrics.db \
        --model-path analytics/retrieval_ranker.model

The JSON file contains ``coef``, ``intercept`` and ``features`` fields that are
compatible with :func:`menace.universal_retriever.load_ranker`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from retrieval_training_dataset import build_dataset

try:  # optional dependencies
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:  # pragma: no cover - scikit-learn not installed
    LogisticRegression = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover - lightgbm not installed
    LGBMClassifier = None  # type: ignore


# ---------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """Container for a fitted model and associated feature names."""

    model: object
    feature_names: list[str]


# ---------------------------------------------------------------------------
class _SimpleLogReg:
    """Very small logistic regression implementation using NumPy.

    It is intentionally tiny and only supports the subset of the scikit-learn
    interface required by the tests.  The implementation uses a basic gradient
    descent optimiser and therefore should only be used on small datasets.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 2000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.classes_ = np.array([0, 1])

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SimpleLogReg":
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        for _ in range(self.epochs):
            z = X @ w + b
            preds = self._sigmoid(z)
            dw = (preds - y) @ X / n_samples
            db = float(np.mean(preds - y))
            w -= self.lr * dw
            b -= self.lr * db
        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coef_[0] + self.intercept_[0]
        p = self._sigmoid(z)
        return np.vstack([1 - p, p]).T


# ---------------------------------------------------------------------------
def load_training_data(
    *, vector_db: Path | str = "vector_metrics.db", patch_db: Path | str = "metrics.db"
) -> pd.DataFrame:
    """Load the training dataframe from the metrics databases."""

    return build_dataset(vector_db=vector_db, patch_db=patch_db)


# ---------------------------------------------------------------------------
def train(df: pd.DataFrame) -> TrainedModel:
    """Fit a ranking model on ``df`` and return the fitted model."""

    feature_cols = [
        "db_type",
        "age",
        "similarity",
        "exec_freq",
        "roi_delta",
        "prior_hits",
        "win_rate",
        "regret_rate",
        "stale_cost",
        "sample_count",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = pd.get_dummies(
        df[available], columns=["db_type"], prefix="db", dtype=float
    ).fillna(0.0)
    y = df["label"].astype(int)

    model: object
    if LogisticRegression is not None:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
    elif LGBMClassifier is not None:
        model = LGBMClassifier()
        model.fit(X, y)
    else:  # pragma: no cover - exercised when dependencies missing
        model = _SimpleLogReg().fit(X.to_numpy(), y.to_numpy())

    return TrainedModel(model=model, feature_names=list(X.columns))


# ---------------------------------------------------------------------------
def save_model(tm: TrainedModel, path: Path | str) -> Path:
    """Persist ``tm`` to ``path`` in JSON format."""

    model = tm.model
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        coef = getattr(model, "coef_")
        intercept = getattr(model, "intercept_")
        classes = getattr(model, "classes_", [0, 1])
        data = {
            "coef": np.asarray(coef).tolist(),
            "intercept": np.asarray(intercept).tolist(),
            "classes": np.asarray(classes).tolist(),
            "features": tm.feature_names,
        }
    else:  # pragma: no cover - used for models that expose ``booster_`` etc.
        data = {
            "booster": getattr(model, "booster_", None),
            "features": tm.feature_names,
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
def load_model(path: Path | str) -> dict:
    """Return the JSON model stored at ``path``."""

    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Command line interface for training the ranker."""

    parser = argparse.ArgumentParser(description="Train retrieval ranking model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train model")
    t.add_argument("--vector-db", default="vector_metrics.db")
    t.add_argument("--patch-db", default="metrics.db")
    t.add_argument(
        "--model-path",
        default="analytics/retrieval_ranker.model",
        help="Where to store the serialized model",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "train":
        df = load_training_data(vector_db=args.vector_db, patch_db=args.patch_db)
        tm = train(df)
        save_model(tm, args.model_path)
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

