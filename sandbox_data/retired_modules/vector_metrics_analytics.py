from __future__ import annotations

"""Analytical utilities for vector operation metrics.

This module exposes helpers to:

* calculate the time-based cost of serving stale embeddings,
* aggregate retrieval metrics by database to evaluate ROI contribution,
* surface retrieval events for ranking or model training pipelines.
"""

import argparse
import json
from datetime import datetime, timedelta
from typing import Any

from dynamic_path_router import resolve_path

try:  # pragma: no cover - allow running as a script
    from .vector_metrics_db import VectorMetricsDB, default_vector_metrics_path
except Exception:  # pragma: no cover - fallback when executed directly
    from vector_metrics_db import VectorMetricsDB, default_vector_metrics_path  # type: ignore


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
    cur = db.conn.execute(
        sql + (" LIMIT ?" if limit is not None else ""),
        (limit,) if limit is not None else (),
    )
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


# ---------------------------------------------------------------------------
def roi_trends(
    db: VectorMetricsDB,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Return ROI totals per day for each origin database.

    The returned mapping contains a list of ``(date, roi)`` tuples for each
    database.  ``days`` limits the history window while ``now`` may be supplied
    for deterministic testing.
    """

    now = now or datetime.utcnow()
    start = now - timedelta(days=days)
    cur = db.conn.execute(
        """
        SELECT db, date(ts) AS day, COALESCE(SUM(contribution),0)
          FROM vector_metrics
         WHERE event_type='retrieval' AND ts >= ?
      GROUP BY db, day
         ORDER BY day
        """,
        (start.isoformat(),),
    )
    trends: dict[str, list[tuple[str, float]]] = {}
    for db_name, day, roi in cur.fetchall():
        trends.setdefault(str(db_name), []).append((str(day), float(roi or 0.0)))
    return trends


# ---------------------------------------------------------------------------
def ranking_weight_changes(
    db: VectorMetricsDB,
    *,
    limit: int | None = None,
) -> dict[str, list[dict[str, float | str]]]:
    """Return history of ranking weight adjustments grouped by origin.

    Each entry in the returned mapping contains dictionaries with ``ts``,
    ``delta`` and ``weight`` keys.  ``limit`` restricts the total number of
    rows considered, newest first.
    """

    sql = (
        "SELECT db, ts, contribution, COALESCE(similarity, context_score) "
        "FROM vector_metrics WHERE event_type='ranker' ORDER BY ts DESC"
    )
    cur = db.conn.execute(
        sql + (" LIMIT ?" if limit is not None else ""),
        (limit,) if limit is not None else (),
    )
    changes: dict[str, list[dict[str, float | str]]] = {}
    for db_name, ts, delta, weight in cur.fetchall():
        changes.setdefault(str(db_name), []).append(
            {
                "ts": str(ts),
                "delta": float(delta or 0.0),
                "weight": float(weight or 0.0),
            }
        )
    return changes


# ---------------------------------------------------------------------------
def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Vector metrics analytics helpers"
    )
    parser.add_argument(
        "--db",
        default=str(default_vector_metrics_path()),
        help="Path to VectorMetricsDB",
    )
    parser.add_argument(
        "--roi-summary",
        action="store_true",
        help="Print ROI trends per origin database",
    )
    parser.add_argument(
        "--weight-summary",
        action="store_true",
        help="Print ranking weight change history per origin",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of history for ROI trends",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of ranking weight change rows to show",
    )
    args = parser.parse_args(argv)

    vdb = VectorMetricsDB(args.db)
    if args.roi_summary:
        trend = roi_trends(vdb, days=args.days)
        print(json.dumps(trend, indent=2))
    if args.weight_summary:
        changes = ranking_weight_changes(vdb, limit=args.limit)
        print(json.dumps(changes, indent=2))
    if not (args.roi_summary or args.weight_summary):
        parser.print_help()


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()


__all__ = [
    "stale_embedding_cost",
    "roi_by_database",
    "retrieval_training_samples",
    "roi_trends",
    "ranking_weight_changes",
]
