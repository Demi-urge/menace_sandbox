from __future__ import annotations

"""Utilities for building retrieval training datasets.

This module reads retrieval events from :mod:`vector_metrics_db` and joins
with ``patch_outcomes`` records to produce labeled samples ready for model
training.  It can also export the resulting dataset to CSV or Parquet files
for offline processing.
"""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Literal

import pandas as pd


@dataclass
class TrainingSample:
    """Single retrieval training example."""

    session_id: str
    vector_id: str
    db_type: str
    age: float
    similarity: float
    exec_freq: float
    roi_delta: float
    prior_hits: int
    label: int


# ---------------------------------------------------------------------------
def build_dataset(
    vector_db: Path | str = "vector_metrics.db",
    patch_db: Path | str = "metrics.db",
) -> pd.DataFrame:
    """Return a ``pandas.DataFrame`` with training features and labels."""

    v_path = Path(vector_db)
    if not v_path.exists():
        raise FileNotFoundError(v_path)

    with sqlite3.connect(v_path) as vconn:
        retrieval_df = pd.read_sql(
            """
            SELECT session_id, vector_id, db AS db_type, age, similarity,
                   contribution, hit, ts
              FROM vector_metrics
             WHERE event_type='retrieval'
            """,
            vconn,
        )

    p_path = Path(patch_db)
    patch_df = pd.DataFrame()
    if p_path.exists():
        with sqlite3.connect(p_path) as pconn:
            patch_df = pd.read_sql(
                """
                SELECT session_id, vector_id,
                       CASE WHEN success=1 AND COALESCE(reverted,0)=0 THEN 1 ELSE 0 END AS label
                  FROM patch_outcomes
                """,
                pconn,
            )

    # Merge retrieval metrics with patch outcomes.
    df = retrieval_df.merge(patch_df, on=["session_id", "vector_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    # Feature engineering
    df["age"] = df["age"].fillna(0)
    df["similarity"] = df["similarity"].fillna(0)
    df["roi_delta"] = df["contribution"].fillna(0)
    df["hit"] = df["hit"].fillna(0).astype(int)

    df = df.sort_values("ts")
    grp = df.groupby("vector_id", sort=False)

    df["prior_hits"] = grp["hit"].cumsum() - df["hit"]
    df["exec_freq"] = grp.cumcount()

    return df[
        [
            "session_id",
            "vector_id",
            "db_type",
            "age",
            "similarity",
            "exec_freq",
            "roi_delta",
            "prior_hits",
            "label",
        ]
    ]


# ---------------------------------------------------------------------------
def export_dataset(
    out_path: Path | str,
    fmt: Literal["csv", "parquet"] = "csv",
    *,
    vector_db: Path | str = "vector_metrics.db",
    patch_db: Path | str = "metrics.db",
) -> Path:
    """Export the training dataset to ``out_path``.

    Parameters
    ----------
    out_path:
        Destination file.  The directory is created if needed.
    fmt:
        ``"csv"`` or ``"parquet"``.
    vector_db, patch_db:
        Paths to the metrics databases.
    """

    df = build_dataset(vector_db=vector_db, patch_db=patch_db)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":  # pragma: no cover - optional dependency
        df.to_parquet(out_path, index=False)
    else:  # pragma: no cover - invalid format
        raise ValueError("fmt must be 'csv' or 'parquet'")
    return out_path


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export retrieval training dataset")
    parser.add_argument("--vector-db", default="vector_metrics.db", help="Path to vector metrics DB")
    parser.add_argument("--patch-db", default="metrics.db", help="Path to patch outcomes DB")
    parser.add_argument("--out", required=True, help="Output file path")
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "parquet"],
        help="Export format",
    )
    args = parser.parse_args()
    export_dataset(args.out, args.format, vector_db=args.vector_db, patch_db=args.patch_db)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
