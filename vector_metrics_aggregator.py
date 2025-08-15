from __future__ import annotations

"""Aggregate vector operation metrics into heatmap-friendly files."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Iterable, Dict, Tuple, List


class VectorMetricsAggregator:
    """Summarise :class:`VectorMetricsDB` records by time period."""

    def __init__(self, db_path: Path | str = "vector_metrics.db") -> None:
        self.db_path = Path(db_path)

    # ------------------------------------------------------------------
    def _rows(self) -> Iterable[Tuple[str, str, int, str]]:
        """Yield ``(event_type, db, tokens, ts)`` rows from the database."""
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT event_type, db, tokens, ts FROM vector_metrics"
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

        agg: Dict[Tuple[str, str, str], Dict[str, int]] = {}
        for event, db, tokens, ts in rows:
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                # Skip malformed timestamps
                continue
            bucket = dt.strftime("%Y-%m-%d %H:00" if period == "hourly" else "%Y-%m-%d")
            key = (bucket, db or "", event or "")
            if key not in agg:
                agg[key] = {"count": 0, "tokens_total": 0}
            agg[key]["count"] += 1
            agg[key]["tokens_total"] += int(tokens or 0)

        result = [
            {
                "period": k[0],
                "db": k[1],
                "event_type": k[2],
                "count": v["count"],
                "tokens_total": v["tokens_total"],
            }
            for k, v in sorted(agg.items())
        ]
        return result

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
                fh, fieldnames=["period", "db", "event_type", "count", "tokens_total"]
            )
            writer.writeheader()
            writer.writerows(data)

    # ------------------------------------------------------------------
    def run(
        self,
        period: str = "hourly",
        json_file: Path | str = "vector_metrics_heatmap.json",
        csv_file: Path | str = "vector_metrics_heatmap.csv",
    ) -> List[Dict[str, object]]:
        """Aggregate metrics and export them to files.

        Returns the aggregated data for convenience.
        """
        data = self.aggregate(period)
        self.export(data, json_file, csv_file)
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
        "--json", default="vector_metrics_heatmap.json", help="JSON output file"
    )
    parser.add_argument(
        "--csv", default="vector_metrics_heatmap.csv", help="CSV output file"
    )
    args = parser.parse_args()

    agg = VectorMetricsAggregator(args.db)
    agg.run(args.period, args.json, args.csv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
