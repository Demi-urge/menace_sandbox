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

from db_router import GLOBAL_ROUTER, init_db_router

# Allow reuse of an existing router but fall back to a local initialisation
# when the module is executed directly.
router = GLOBAL_ROUTER or init_db_router("retrieval_training_dataset")


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

    vconn = router.get_connection("vector_metrics")
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
    stats_df = pd.DataFrame(
        columns=[
            "origin_db",
            "win_rate",
            "regret_rate",
            "stale_cost",
            "sample_count",
            "roi",
        ]
    )
    if p_path.exists():
        pconn = router.get_connection("patch_outcomes")
        patch_df = pd.read_sql(
                """
                SELECT session_id, vector_id,
                       CASE WHEN success=1 AND COALESCE(reverted,0)=0 THEN 1 ELSE 0 END AS label
                  FROM patch_outcomes
                """,
                pconn,
            )
        try:
            stats_df = pd.read_sql(
                "SELECT origin_db, win_rate, regret_rate, stale_cost, sample_count, roi FROM retriever_stats",
                pconn,
            )
        except Exception:
            try:
                tmp = pd.read_sql(
                    "SELECT origin_db, wins, regrets FROM retriever_stats",
                    pconn,
                )
                total = tmp["wins"].fillna(0) + tmp["regrets"].fillna(0)
                tmp["win_rate"] = tmp["wins"].fillna(0) / total.where(total > 0, 1)
                tmp["regret_rate"] = tmp["regrets"].fillna(0) / total.where(total > 0, 1)
                tmp["stale_cost"] = 0.0
                tmp["sample_count"] = total
                tmp["roi"] = 0.0
                stats_df = tmp[
                    [
                        "origin_db",
                        "win_rate",
                        "regret_rate",
                        "stale_cost",
                        "sample_count",
                        "roi",
                    ]
                ]
            except Exception:
                pass

    # Merge retrieval metrics with patch outcomes and reliability statistics.
    df = retrieval_df.merge(patch_df, on=["session_id", "vector_id"], how="left")
    df = df.merge(
        stats_df,
        left_on="db_type",
        right_on="origin_db",
        how="left",
    ).drop(columns=["origin_db"], errors="ignore")
    df["label"] = df["label"].fillna(0).astype(int)

    # Feature engineering
    df["age"] = df["age"].fillna(0)
    df["similarity"] = df["similarity"].fillna(0)
    df["roi_delta"] = df["contribution"].fillna(0)
    df["hit"] = df["hit"].fillna(0).astype(int)
    df["win_rate"] = df["win_rate"].fillna(0.0)
    df["regret_rate"] = df["regret_rate"].fillna(0.0)
    df["stale_cost"] = df["stale_cost"].fillna(0.0)
    df["sample_count"] = df["sample_count"].fillna(0.0)
    df["roi"] = df["roi"].fillna(0.0)

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
            "win_rate",
            "regret_rate",
            "stale_cost",
            "sample_count",
            "roi",
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
