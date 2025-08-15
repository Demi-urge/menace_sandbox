from __future__ import annotations

"""Training utilities for retrieval ranking models.

This module pulls training samples from :mod:`vector_metrics_analytics`,
optionally augments them with extra features and trains a simple logistic
regression classifier to predict whether a retrieval results in a
*win* (or regret).  Model weights are exported to JSON so other
components can reuse them without depending on scikit-learn.

A small CLI is provided for periodic retraining from fresh logs::

    python -m menace.retrieval_ranker --db-path metrics.db \
        --model-path ranker.json
"""

from pathlib import Path
from typing import Iterable, Sequence, Any, Optional
import argparse
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .vector_metrics_db import VectorMetricsDB
from . import vector_metrics_analytics as vma


# ---------------------------------------------------------------------------
def prepare_training_dataframe(
    db: VectorMetricsDB,
    extra_features: Iterable[dict[str, Any]] | pd.DataFrame | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load retrieval samples and return a DataFrame ready for training."""

    samples = vma.retrieval_training_samples(db, limit=limit)
    df = pd.DataFrame(samples)
    if extra_features is not None:
        if isinstance(extra_features, pd.DataFrame):
            extra_df = extra_features.reset_index(drop=True)
        else:
            extra_df = pd.DataFrame(list(extra_features))
        df = pd.concat([df, extra_df.reindex(df.index)], axis=1)
    df = pd.get_dummies(df, columns=["db"], dtype=float)
    df = df.fillna(0.0)
    for col in ["hit", "win", "regret"]:
        if col in df:
            df[col] = df[col].astype(int)
    return df


# ---------------------------------------------------------------------------
def train_retrieval_ranker(
    df: pd.DataFrame, *, target: str = "win"
) -> tuple[LogisticRegression, list[str]]:
    """Train a logistic-regression classifier on ``df``."""

    if target not in {"win", "regret"}:
        raise ValueError("target must be 'win' or 'regret'")
    X = df.drop(columns=["win", "regret"], errors="ignore")
    y = df[target]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, list(X.columns)


# ---------------------------------------------------------------------------
def save_model(
    model: LogisticRegression,
    feature_names: Sequence[str],
    path: str | Path,
) -> None:
    """Persist ``model`` coefficients to ``path`` as JSON."""

    data = {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
        "features": list(feature_names),
    }
    Path(path).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for model retraining."""

    p = argparse.ArgumentParser(description="Train retrieval ranking model")
    p.add_argument("--db-path", default="vector_metrics.db")
    p.add_argument("--model-path", default="retrieval_ranker.json")
    p.add_argument("--target", choices=["win", "regret"], default="win")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--extra-features", type=Path, help="Optional CSV file with extra features"
    )
    args = p.parse_args(argv)

    db = VectorMetricsDB(args.db_path)
    extra = None
    if args.extra_features:
        extra = pd.read_csv(args.extra_features)
    df = prepare_training_dataframe(db, extra_features=extra, limit=args.limit)
    model, feats = train_retrieval_ranker(df, target=args.target)
    save_model(model, feats, args.model_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
