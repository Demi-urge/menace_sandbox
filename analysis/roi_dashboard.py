"""Simple CLI/dashboard for ROI and retrieval metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

from vector_metrics_db import VectorMetricsDB
from roi_tracker import ROITracker

try:  # optional dependency for plotting
    import matplotlib.pyplot as plt  # pragma: no cover - plotting only
except Exception:  # pragma: no cover - matplotlib may be missing
    plt = None  # type: ignore


# ---------------------------------------------------------------------------
def _patch_outcomes(db: VectorMetricsDB, limit: int = 10) -> List[Tuple[str, float]]:
    """Return top ``limit`` patches by cumulative ROI contribution."""

    cur = db.conn.execute(
        """
        SELECT patch_id, COALESCE(SUM(contribution),0) AS roi
          FROM vector_metrics
         WHERE event_type='retrieval' AND patch_id != ''
      GROUP BY patch_id
      ORDER BY roi DESC
         LIMIT ?
        """,
        (limit,),
    )
    return [(str(patch), float(roi)) for patch, roi in cur.fetchall()]


# ---------------------------------------------------------------------------
def _plot(
    roi_history: List[float],
    db_metrics: List[Dict[str, float]],
    patch_metrics: List[Tuple[str, float]],
    *,
    output: str | None = None,
) -> None:
    """Render plots for ROI trend, retrieval success and patch outcomes."""

    if plt is None:  # pragma: no cover - plotting optional
        print("matplotlib not available; skipping visualisation")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    if roi_history:
        axes[0].plot(range(len(roi_history)), roi_history, marker="o")
        axes[0].set_title("ROI Delta Trend")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("ROI Δ")

    if db_metrics:
        names = [m["origin_db"] for m in db_metrics]
        win_rates = [m.get("win_rate", 0.0) for m in db_metrics]
        regret_rates = [m.get("regret_rate", 0.0) for m in db_metrics]
        x = range(len(names))
        axes[1].bar([i - 0.2 for i in x], win_rates, width=0.4, label="Win rate")
        axes[1].bar([i + 0.2 for i in x], regret_rates, width=0.4, label="Regret rate")
        axes[1].set_title("Retrieval Success by Database")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(names, rotation=45, ha="right")
        axes[1].legend()

    if patch_metrics:
        patches = [p for p, _ in patch_metrics]
        roi_vals = [r for _, r in patch_metrics]
        axes[2].bar(patches, roi_vals)
        axes[2].set_title("Patch ROI Contribution")
        axes[2].set_xticklabels(patches, rotation=45, ha="right")
        axes[2].set_ylabel("ROI")

    fig.tight_layout()
    if output:
        fig.savefig(output)
    else:  # pragma: no cover - interactive display
        plt.show()


# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    """Entry point for the ROI dashboard CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector-db",
        default="vector_metrics.db",
        help="Path to vector metrics SQLite DB",
    )
    parser.add_argument(
        "--roi-history",
        default="sandbox_data/roi_history.json",
        help="Path to ROI history JSON or SQLite file",
    )
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=5,
        help="Iterations considered when checking for plateau",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the generated plot instead of displaying",
    )
    args = parser.parse_args(argv)

    vec_db = VectorMetricsDB(Path(args.vector_db))
    tracker = ROITracker()
    tracker.load_history(str(args.roi_history))

    # Aggregate per-database metrics and ROI deltas.
    tracker.ingest_vector_metrics_db(vec_db)
    db_metrics = tracker.db_roi_report()
    patches = _patch_outcomes(vec_db)

    # Plateau detection.
    threshold = tracker.diminishing()
    recent = tracker.roi_history[-args.plateau_window :]
    if recent and all(abs(x) <= threshold for x in recent):
        print(
            f"Warning: ROI improvement plateau detected (|Δ| <= {threshold:.4f} for last {len(recent)} iterations)",
        )

    _plot(tracker.roi_history, db_metrics, patches, output=args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
