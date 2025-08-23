from __future__ import annotations

"""Assemble training data combining ROI deltas, performance metrics and GPT scores."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from db_router import GLOBAL_ROUTER, init_db_router
from evaluation_history_db import EvaluationHistoryDB


@dataclass
class DatasetRecord:
    module: str
    ts: pd.Timestamp
    roi_delta: float
    performance_delta: float
    gpt_score: float


router = GLOBAL_ROUTER or init_db_router("analytics")


def _load_roi_events() -> pd.DataFrame:
    """Return ROI deltas per module."""

    conn = router.get_connection("roi_events")
    df = pd.read_sql(
        "SELECT action AS module, roi_before, roi_after, ts FROM roi_events",
        conn,
    )
    if df.empty:
        return pd.DataFrame(columns=["module", "ts", "roi_delta"])
    df["ts"] = pd.to_datetime(df["ts"])  # type: ignore[call-arg]
    df["roi_delta"] = df["roi_after"].astype(float) - df["roi_before"].astype(float)
    return df[["module", "ts", "roi_delta"]]


def _load_performance() -> pd.DataFrame:
    """Return performance deltas based on profitability per module."""

    conn = router.get_connection("metrics")
    df = pd.read_sql(
        "SELECT bot AS module, profitability, ts FROM metrics",
        conn,
    )
    if df.empty:
        return pd.DataFrame(columns=["module", "ts", "performance_delta"])
    df["ts"] = pd.to_datetime(df["ts"])  # type: ignore[call-arg]
    df.sort_values(["module", "ts"], inplace=True)
    df["performance_delta"] = df.groupby("module")["profitability"].diff()
    return df[["module", "ts", "performance_delta"]]


def _load_eval_scores(path: str | Path) -> pd.DataFrame:
    """Return GPT evaluation scores per module."""

    db = EvaluationHistoryDB(path)
    rows: list[dict[str, object]] = []
    for eng in db.engines():
        for score, ts, _passed, _err in db.history(eng, limit=1_000_000):
            rows.append({"module": eng, "ts": pd.to_datetime(ts), "gpt_score": float(score)})
    if not rows:
        return pd.DataFrame(columns=["module", "ts", "gpt_score"])
    return pd.DataFrame(rows)


def build_dataset(
    *,
    evaluation_path: str | Path = "evaluation_history.db",
) -> pd.DataFrame:
    """Load, merge and normalise ROI, performance and evaluation data.

    Returns a :class:`~pandas.DataFrame` with columns ``module``, ``ts``,
    ``roi_delta``, ``performance_delta`` and ``gpt_score``. Numerical columns are
    normalised to zero mean and unit variance.
    """

    roi_df = _load_roi_events()
    perf_df = _load_performance()
    eval_df = _load_eval_scores(evaluation_path)

    if roi_df.empty or perf_df.empty or eval_df.empty:
        return pd.DataFrame(columns=["module", "ts", "roi_delta", "performance_delta", "gpt_score"])

    merged = pd.merge_asof(
        roi_df.sort_values("ts"),
        perf_df.sort_values("ts"),
        on="ts",
        by="module",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("ts"),
        eval_df.sort_values("ts"),
        on="ts",
        by="module",
        direction="backward",
    )
    merged.dropna(subset=["roi_delta", "performance_delta", "gpt_score"], inplace=True)

    for col in ["roi_delta", "performance_delta", "gpt_score"]:
        if not merged[col].empty:
            merged[col] = (merged[col] - merged[col].mean()) / (merged[col].std() or 1.0)
    return merged.reset_index(drop=True)


__all__ = ["DatasetRecord", "build_dataset"]
