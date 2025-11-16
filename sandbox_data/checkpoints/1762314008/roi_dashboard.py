"""Simple CLI/dashboard for workflow-level ROI results.

This version pulls data from ``roi_results.db`` and can visualise multiple
metrics over time for one or more workflows.  Results can be filtered by
``workflow_id`` and timestamp range.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Sequence

try:
    from ..roi_results_db import ROIResultsDB
except ImportError:  # pragma: no cover - allow running as script
    from roi_results_db import ROIResultsDB

try:  # optional dependency for plotting
    import matplotlib.pyplot as plt  # pragma: no cover - plotting only
except Exception:  # pragma: no cover - matplotlib may be missing
    plt = None  # type: ignore


# ---------------------------------------------------------------------------
def _fetch_results(
    db: ROIResultsDB,
    workflows: Sequence[str] | None,
    start: str | None,
    end: str | None,
) -> List[Dict[str, object]]:
    """Return ROI result rows honouring the provided filters."""

    query = (
        "SELECT workflow_id, timestamp, roi_gain, success_rate, "
        "workflow_synergy_score, bottleneck_index, patchability_score "
        "FROM workflow_results"
    )
    clauses: List[str] = []
    params: List[object] = []

    if workflows:
        placeholders = ",".join(["?"] * len(workflows))
        clauses.append(f"workflow_id IN ({placeholders})")
        params.extend(workflows)
    if start:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end:
        clauses.append("timestamp <= ?")
        params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp"

    cur = db.conn.execute(query, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
def _plot(records: List[Dict[str, object]], *, output: str | None = None) -> None:
    """Render metric trends for the provided records."""

    if plt is None:  # pragma: no cover - plotting optional
        print("matplotlib not available; skipping visualisation")
        return

    if not records:
        print("No data matching filters")
        return

    metrics = [
        "roi_gain",
        "success_rate",
        "workflow_synergy_score",
        "bottleneck_index",
        "patchability_score",
    ]
    workflows = sorted({str(r["workflow_id"]) for r in records})
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for wf in workflows:
            wf_records = [r for r in records if r["workflow_id"] == wf]
            times = [datetime.fromisoformat(str(r["timestamp"])) for r in wf_records]
            values = [float(r[metric]) for r in wf_records]
            ax.plot(times, values, marker="o", label=wf)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(metric)
        if len(workflows) > 1:
            ax.legend()

    fig.tight_layout()
    if output:
        fig.savefig(output)
    else:  # pragma: no cover - interactive display
        plt.show()


# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    """Entry point for the ROI dashboard CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-db", default="roi_results.db", help="Path to ROI results DB")
    parser.add_argument(
        "--workflow-id",
        action="append",
        dest="workflows",
        help="Workflow ID to filter (can be repeated)",
    )
    parser.add_argument("--start", help="Start timestamp (inclusive)")
    parser.add_argument("--end", help="End timestamp (inclusive)")
    parser.add_argument("--output", help="Optional path to save the plot instead of displaying")
    args = parser.parse_args(argv)

    db = ROIResultsDB(args.roi_db)
    records = _fetch_results(db, args.workflows, args.start, args.end)
    _plot(records, output=args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
