from __future__ import annotations

"""Correlate retrieval sessions with patch outcomes.

This module analyses :class:`vector_metrics_db.VectorMetricsDB` records and
summarises patch outcomes per origin database.  For every origin it returns the
fraction of successful patches and the average ROI delta contributed by that
origin.  Results can be consumed directly or exported as JSON via the CLI.
"""

from pathlib import Path
import argparse
import json
from typing import Dict, Union

from vector_metrics_db import VectorMetricsDB


# ---------------------------------------------------------------------------

def per_origin_stats(db: Union[VectorMetricsDB, str, Path]) -> Dict[str, Dict[str, float]]:
    """Return success rates and ROI deltas grouped by origin.

    ``db`` may be an existing :class:`VectorMetricsDB` instance or the path to
    the metrics database.  Only retrieval events with recorded outcomes are
    considered.
    """

    if not isinstance(db, VectorMetricsDB):
        db = VectorMetricsDB(db)

    cur = db.conn.execute(
        """
        SELECT db,
               AVG(COALESCE(win,0)) AS success_rate,
               AVG(COALESCE(contribution,0)) AS roi_delta
          FROM vector_metrics
         WHERE event_type='retrieval' AND win IS NOT NULL
      GROUP BY db
        """
    )
    return {
        (name or ""): {"success_rate": float(rate or 0.0), "roi_delta": float(roi or 0.0)}
        for name, rate, roi in cur.fetchall()
    }


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate retrieval metadata with patch outcomes"
    )
    parser.add_argument(
        "--db", default="vector_metrics.db", help="Path to VectorMetricsDB"
    )
    parser.add_argument(
        "--json", default=None, help="Optional JSON output file"
    )
    args = parser.parse_args()

    stats = per_origin_stats(args.db)
    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
