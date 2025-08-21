from __future__ import annotations

"""Readiness index helpers.

This module provides small utilities for combining risk adjusted ROI with
system level quality metrics.  The resulting value is used by dashboards and
deployment governance to gauge whether a workflow is ready for promotion.
"""

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .roi_tracker import ROITracker

# Optional global tracker used by :func:`readiness_summary`.
_tracker: "ROITracker" | None = None


def bind_tracker(tracker: "ROITracker") -> None:
    """Register ``tracker`` for global readiness summaries."""

    global _tracker
    _tracker = tracker


def compute_readiness(
    raroi: float, reliability: float, safety: float, resilience: float
) -> float:
    """Return combined readiness score.

    All inputs are expected to be normalised to the 0-1 range.  The score is a
    simple product of the four metrics.
    """

    return float(raroi) * float(reliability) * float(safety) * float(resilience)


def evaluate_cycle(
    tracker: "ROITracker", reliability: float, safety: float, resilience: float
) -> float:
    """Aggregate the latest metrics into a readiness score.

    Parameters
    ----------
    tracker:
        ROI tracker holding the most recent risk-adjusted ROI value.
    reliability, safety, resilience:
        Normalised quality metrics for the current cycle.
    """

    raroi = float(tracker.last_raroi or 0.0)
    return compute_readiness(raroi, reliability, safety, resilience)


def readiness_summary(workflow_id: str) -> Dict[str, float]:
    """Return a readiness snapshot for ``workflow_id``.

    The summary includes the latest metrics from the registered tracker and the
    resulting readiness score.  :func:`bind_tracker` must be called before this
    function is used.
    """

    if _tracker is None:
        raise RuntimeError("ROITracker not configured")

    raroi = float(_tracker.last_raroi or 0.0)
    reliability = float(_tracker.reliability())
    safety_hist = (
        _tracker.metrics_history.get("security_score")
        or _tracker.metrics_history.get("synergy_safety_rating")
        or [1.0]
    )
    safety = float(safety_hist[-1]) if safety_hist else 1.0
    res_hist = (
        _tracker.metrics_history.get("synergy_resilience")
        or _tracker.metrics_history.get("resilience")
        or [1.0]
    )
    resilience = float(res_hist[-1]) if res_hist else 1.0
    readiness = compute_readiness(raroi, reliability, safety, resilience)
    return {
        "workflow_id": str(workflow_id),
        "raroi": raroi,
        "reliability": reliability,
        "safety": safety,
        "resilience": resilience,
        "readiness": readiness,
    }
