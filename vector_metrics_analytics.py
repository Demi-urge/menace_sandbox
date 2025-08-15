from __future__ import annotations

"""Analytical utilities for vector operation metrics.

This module exposes helpers to:

* calculate the time-based cost of serving stale embeddings,
* aggregate retrieval metrics by database to evaluate ROI contribution,
* surface retrieval events for ranking or model training pipelines.
"""

from datetime import datetime, timedelta
from typing import Any

from .vector_metrics_db import VectorMetricsDB


# ---------------------------------------------------------------------------
def stale_embedding_cost(
    db: VectorMetricsDB,
    max_age: timedelta,
    *,
    now: datetime | None = None,
) -> float:
    """Return token-seconds cost for embeddings older than ``max_age``.

    For each embedding whose age exceeds ``max_age`` this computes
    ``(age - max_age).total_seconds() * tokens`` and sums the results.
    ``now`` may be supplied for deterministic testing.
    """

    now = now or datetime.utcnow()
    cur = db.conn.execute(
        "SELECT tokens, ts FROM vector_metrics WHERE event_type='embedding'"
    )
    cost = 0.0
    for tokens, ts_str in cur.fetchall():
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        age = now - ts
        excess = age - max_age
        if excess > timedelta(0):
            cost += excess.total_seconds() * int(tokens)
    return cost


# ---------------------------------------------------------------------------
def roi_by_database(db: VectorMetricsDB) -> dict[str, float]:
    """Aggregate retrieval contributions grouped by database."""

    cur = db.conn.execute(
        """
        SELECT db, COALESCE(SUM(contribution),0)
          FROM vector_metrics
         WHERE event_type='retrieval'
      GROUP BY db
        """
    )
    return {name: float(contrib) for name, contrib in cur.fetchall()}


# ---------------------------------------------------------------------------
def retrieval_training_samples(
    db: VectorMetricsDB,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return retrieval events as training samples.

    The resulting list can be fed into ranking algorithms or other
    model-training pipelines to improve retrieval quality. Each sample
    contains ``db``, ``rank``, ``hit``, ``contribution``, ``win`` and
    ``regret`` fields. ``limit`` restricts the number of rows returned,
    newest first.
    """

    sql = (
        "SELECT db, rank, hit, contribution, win, regret FROM vector_metrics "
        "WHERE event_type='retrieval' ORDER BY ts DESC"
    )
    cur = db.conn.execute(sql + (" LIMIT ?" if limit is not None else ""), (limit,) if limit is not None else ())
    samples: list[dict[str, Any]] = []
    for db_name, rank, hit, contrib, win, regret in cur.fetchall():
        samples.append(
            {
                "db": db_name,
                "rank": int(rank) if rank is not None else None,
                "hit": bool(hit) if hit is not None else False,
                "contribution": float(contrib) if contrib is not None else 0.0,
                "win": bool(win) if win is not None else False,
                "regret": bool(regret) if regret is not None else False,
            }
        )
    return samples


__all__ = [
    "stale_embedding_cost",
    "roi_by_database",
    "retrieval_training_samples",
]
