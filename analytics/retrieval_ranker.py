from __future__ import annotations

"""Utilities for training a retrieval ranking model.

The training routine joins retrieval statistics with patch outcome labels and
fits a binary classifier.  The resulting model and feature metadata are stored
as a pickle file for consumption by other components.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

try:  # optional dependency
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - lightgbm is optional
    lgb = None  # type: ignore


# ---------------------------------------------------------------------------
@dataclass
class TrainingSet:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]


# ---------------------------------------------------------------------------
def build_training_set(
    stats_path: str | Path,
    labels_path: str | Path,
    *,
    join_on: str = "session_id",
) -> TrainingSet:
    """Load retrieval statistics and patch outcome labels.

    ``stats_path`` and ``labels_path`` should point to JSON Lines files.  The
    stats file is expected to include columns such as ``origin_db`` (database
    type), ``age`` (seconds since last update), ``similarity``, ``roi_delta``,
    ``exec_freq`` (execution frequency), ``prior_hits`` (historical hit count),
    aggregate ``win_rate`` / ``regret_rate`` scores, and optional
    ``stale_cost`` / ``sample_count`` reliability metrics.  The labels file must
    provide the training target under a ``win`` column.
    """

    stats = pd.read_json(stats_path, lines=True)
    labels = pd.read_json(labels_path, lines=True)
    df = stats.merge(labels, on=join_on, how="inner")

    numeric_features = [
        "age",
        "similarity",
        "roi_delta",
        "exec_freq",
        "prior_hits",
        "win_rate",
        "regret_rate",
        "stale_cost",
        "sample_count",
    ]
    for col in numeric_features:
        if col not in df:
            df[col] = 0.0

    df = pd.get_dummies(df, columns=["origin_db"], prefix="db", dtype=float)

    feature_cols = [
        c for c in df.columns if c.startswith("db_") or c in numeric_features
    ]
    X = df[feature_cols].fillna(0.0)

    if "win" not in df:
        raise ValueError("labels file must include a 'win' column")
    y = df["win"].astype(int)

    return TrainingSet(X=X, y=y, feature_names=list(feature_cols))


# ---------------------------------------------------------------------------
def train_ranker(ts: TrainingSet, *, model: str = "logistic"):
    """Train a ranking model and return the fitted instance."""

    if model == "lightgbm" and lgb is not None:
        train_data = lgb.Dataset(ts.X, label=ts.y)
        params = {"objective": "binary", "verbosity": -1}
        booster = lgb.train(params, train_data, num_boost_round=50)
        return booster

    clf = LogisticRegression(max_iter=1000)
    clf.fit(ts.X, ts.y)
    return clf


# ---------------------------------------------------------------------------
def save_model(model, feature_names: list[str], path: str | Path) -> None:
    """Persist model and associated feature names to ``path``."""

    joblib.dump({"model": model, "features": feature_names}, path)


# ---------------------------------------------------------------------------
def retrain_loop(args: argparse.Namespace) -> None:
    """Run a single training cycle or loop periodically."""

    while True:
        ts = build_training_set(
            args.stats_path, args.labels_path, join_on=args.join_on
        )
        model = train_ranker(ts, model=args.model)
        save_model(model, ts.feature_names, args.model_path)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train retrieval ranking model")
    p.add_argument(
        "--stats-path",
        default="analytics/retrieval_outcomes.jsonl",
        help="Path to retrieval statistics JSONL",
    )
    p.add_argument(
        "--labels-path",
        default="analytics/patch_outcomes.jsonl",
        help="Path to patch outcome labels JSONL",
    )
    p.add_argument(
        "--model-path",
        default="analytics/retrieval_ranker.pkl",
        help="Where to store the trained model",
    )
    p.add_argument(
        "--model",
        choices=["logistic", "lightgbm"],
        default="logistic",
        help="Model type to train",
    )
    p.add_argument(
        "--join-on",
        default="session_id",
        help="Column used to join stats and labels",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Seconds between retraining runs; 0 performs a single pass",
    )

    args = p.parse_args(argv)
    retrain_loop(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

