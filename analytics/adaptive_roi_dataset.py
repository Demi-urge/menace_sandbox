from __future__ import annotations

"""Assemble training data combining ROI deltas, performance metrics and GPT scores."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - executed in minimal envs
    pd = None  # type: ignore[assignment]
    _MISSING_PANDAS = exc
else:
    _MISSING_PANDAS = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import Timestamp as _Timestamp
else:
    _Timestamp = Any

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from evaluation_history_db import EvaluationHistoryDB


_LOGGER = logging.getLogger(__name__)
_PANDAS_WARNING_EMITTED = False


@dataclass
class DatasetRecord:
    module: str
    ts: _Timestamp
    roi_delta: float
    performance_delta: float
    gpt_score: float


def _ensure_pandas() -> None:
    global _PANDAS_WARNING_EMITTED
    if _MISSING_PANDAS is not None:
        if not _PANDAS_WARNING_EMITTED:
            _LOGGER.warning(
                "adaptive ROI dataset disabled; pandas dependency missing"
            )
            _PANDAS_WARNING_EMITTED = True
        raise ModuleNotFoundError(
            "pandas is required for adaptive ROI datasets. Install menace_sandbox[analytics]"
        ) from _MISSING_PANDAS


def _get_router(router: DBRouter | None = None) -> DBRouter:
    """Return an initialised :class:`DBRouter` instance."""

    return router or GLOBAL_ROUTER or init_db_router("analytics")


def _load_roi_events(router: DBRouter) -> pd.DataFrame:
    """Return ROI deltas per module."""

    _ensure_pandas()
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


def _load_performance(router: DBRouter) -> pd.DataFrame:
    """Return performance deltas based on profitability per module."""

    _ensure_pandas()
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


def _load_eval_scores(router: DBRouter) -> pd.DataFrame:
    """Return GPT evaluation scores per module."""

    _ensure_pandas()
    db = EvaluationHistoryDB(router=router)
    rows: list[dict[str, object]] = []
    for eng in db.engines():
        for score, ts, _passed, _err in db.history(eng, limit=1_000_000):
            rows.append({"module": eng, "ts": pd.to_datetime(ts), "gpt_score": float(score)})
    if not rows:
        return pd.DataFrame(columns=["module", "ts", "gpt_score"])
    return pd.DataFrame(rows)


def build_dataset(*, router: DBRouter | None = None) -> pd.DataFrame:
    """Load, merge and normalise ROI, performance and evaluation data.

    Parameters
    ----------
    router:
        Optional :class:`DBRouter` instance. Falls back to the global router
        when not provided.

    Returns
    -------
    pandas.DataFrame
        Columns ``module``, ``ts``, ``roi_delta``, ``performance_delta`` and
        ``gpt_score`` normalised to zero mean and unit variance.
    """

    _ensure_pandas()
    router = _get_router(router)
    roi_df = _load_roi_events(router)
    perf_df = _load_performance(router)
    eval_df = _load_eval_scores(router)

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
