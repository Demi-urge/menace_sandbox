from __future__ import annotations

"""Aggregate vector operation metrics into heatmap-friendly files."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Dict, Tuple, List
import os
import uuid

from db_router import init_db_router
from dynamic_path_router import resolve_path

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

from analytics.session_roi import per_origin_stats  # noqa: E402
from vector_metrics_db import VectorMetricsDB  # noqa: E402

router = GLOBAL_ROUTER


class VectorMetricsAggregator:
    """Summarise :class:`VectorMetricsDB` records by time period."""

    def __init__(self, db_path: Path | str = "vector_metrics.db") -> None:
        self.db_path = Path(db_path)

    # ------------------------------------------------------------------
    def _rows(self) -> Iterable[Tuple[str, str, int, float, str]]:
        """Yield ``(event_type, db, tokens, contribution, ts)`` rows from the database."""
        if not self.db_path.exists():
            return []
        cur = router.get_connection("vector_metrics").execute(
            "SELECT event_type, db, tokens, COALESCE(contribution,0), ts FROM vector_metrics"
        )
        return cur.fetchall()

    # ------------------------------------------------------------------
    def aggregate(self, period: str = "hourly") -> List[Dict[str, object]]:
        """Return aggregated metrics for ``period`` (``hourly`` or ``daily``)."""
        if period not in {"hourly", "daily"}:
            raise ValueError("period must be 'hourly' or 'daily'")

        rows = self._rows()
        if not rows:
            return []

        agg: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        for event, db, tokens, contrib, ts in rows:
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                # Skip malformed timestamps
                continue
            bucket = dt.strftime("%Y-%m-%d %H:00" if period == "hourly" else "%Y-%m-%d")
            key = (bucket, db or "", event or "")
            if key not in agg:
                agg[key] = {"count": 0, "tokens_total": 0, "roi_total": 0.0}
            agg[key]["count"] += 1
            agg[key]["tokens_total"] += int(tokens or 0)
            agg[key]["roi_total"] += float(contrib or 0.0)

        result = [
            {
                "period": k[0],
                "db": k[1],
                "event_type": k[2],
                "count": v["count"],
                "tokens_total": int(v["tokens_total"]),
                "roi_total": float(v["roi_total"]),
            }
            for k, v in sorted(agg.items())
        ]
        return result

    # ------------------------------------------------------------------
    def origin_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-origin success rates and ROI deltas."""
        if not self.db_path.exists():
            return {}
        db = VectorMetricsDB(self.db_path)
        return per_origin_stats(db)

    # ------------------------------------------------------------------
    def export(
        self,
        data: List[Dict[str, object]],
        json_file: Path | str,
        csv_file: Path | str,
    ) -> None:
        """Write ``data`` to ``json_file`` and ``csv_file``."""
        Path(json_file).write_text(json.dumps(data, indent=2))
        with open(csv_file, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["period", "db", "event_type", "count", "tokens_total", "roi_total"],
            )
            writer.writeheader()
            writer.writerows(data)

    # ------------------------------------------------------------------
    def run(
        self,
        period: str = "hourly",
        json_file: Path | str | None = None,
        csv_file: Path | str | None = None,
    ) -> List[Dict[str, object]]:
        """Aggregate metrics and export them to files.

        If ``json_file``/``csv_file`` are not provided the files are named
        ``vector_metrics_heatmap_<period>.json`` and ``.csv`` so hourly and
        daily jobs do not clobber each other's outputs.

        Returns the aggregated data for convenience.
        """
        if json_file is None:
            json_file = f"vector_metrics_heatmap_{period}.json"
        if csv_file is None:
            csv_file = f"vector_metrics_heatmap_{period}.csv"

        data = self.aggregate(period)
        self.export(data, json_file, csv_file)
        stats = self.origin_stats()
        Path("vector_origin_stats.json").write_text(json.dumps(stats, indent=2))
        return data


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate vector metrics into heatmap-friendly files"
    )
    parser.add_argument(
        "--db", default="vector_metrics.db", help="Path to VectorMetricsDB"
    )
    parser.add_argument(
        "--period",
        choices=("hourly", "daily"),
        default="hourly",
        help="Aggregation period",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="JSON output file (defaults to vector_metrics_heatmap_<period>.json)"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV output file (defaults to vector_metrics_heatmap_<period>.csv)"
    )
    args = parser.parse_args()

    agg = VectorMetricsAggregator(args.db)
    agg.run(args.period, args.json, args.csv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
