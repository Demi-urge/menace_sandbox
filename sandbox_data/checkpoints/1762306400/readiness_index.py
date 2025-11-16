from __future__ import annotations

"""Readiness index helpers.

This module provides small utilities for combining risk adjusted ROI with
system level quality metrics.  The resulting value is used by dashboards and
deployment governance to gauge whether a workflow is ready for promotion.

The module now also maintains an in-memory history of readiness snapshots which
can be serialised for telemetry logging.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple
import json
from .scope_utils import Scope

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .roi_tracker import ROITracker
    from .telemetry_backend import TelemetryBackend

# Optional global tracker used by :func:`readiness_summary`.
_tracker: "ROITracker" | None = None

# In-memory readiness history used for telemetry serialisation.
_history: List[Dict[str, float]] = []


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


def _log_history(
    raroi: float, reliability: float, safety: float, resilience: float, readiness: float
) -> None:
    """Append a readiness snapshot to the module level history."""

    _history.append(
        {
            "raroi": float(raroi),
            "reliability": float(reliability),
            "safety": float(safety),
            "resilience": float(resilience),
            "readiness": float(readiness),
        }
    )


def readiness_history() -> List[Dict[str, float]]:
    """Return recorded readiness snapshots."""

    return list(_history)


def serialize_readiness_history() -> str:
    """Return readiness history serialised as JSON."""

    return json.dumps(_history)


def military_grade_readiness(
    raroi: float, reliability: float, safety: float, resilience: float
) -> float:
    """Return hardened readiness score.

    The "military grade" index currently mirrors :func:`compute_readiness` but
    is provided as a dedicated entry point for future weighting schemes.
    All inputs must be normalised to ``0-1``.
    """

    return compute_readiness(raroi, reliability, safety, resilience)


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
    readiness = compute_readiness(raroi, reliability, safety, resilience)
    _log_history(raroi, reliability, safety, resilience, readiness)
    return readiness


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
    _log_history(raroi, reliability, safety, resilience, readiness)
    return {
        "workflow_id": str(workflow_id),
        "raroi": raroi,
        "reliability": reliability,
        "safety": safety,
        "resilience": resilience,
        "readiness": readiness,
    }


class ReadinessIndex:
    """Compute readiness from live metrics or telemetry.

    The class can operate on an :class:`ROITracker` instance, a
    :class:`TelemetryBackend` or both.  When both are provided the tracker takes
    precedence.  Reliability is scaled by the most recent confidence to ensure
    drift and ROI drops immediately impact readiness.
    """

    def __init__(
        self,
        tracker: "ROITracker" | None = None,
        telemetry: "TelemetryBackend" | None = None,
        *,
        scope: Scope | str = "local",
    ) -> None:
        self.tracker = tracker
        self.telemetry = telemetry
        self.scope = scope
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def _latest_metrics(
        self, workflow_id: str | None = None
    ) -> Tuple[float, float, float, float]:
        """Return ``(raroi, reliability, safety, resilience)``."""

        if self.tracker is not None:
            raroi = float(self.tracker.last_raroi or 0.0)
            conf = float(getattr(self.tracker, "last_confidence", 1.0) or 1.0)
            drift_flags = getattr(self.tracker, "drift_flags", [])
            if drift_flags and drift_flags[-1] and conf > 0.5:
                conf *= 0.5
            raroi_hist = getattr(self.tracker, "raroi_history", [])
            if (
                len(raroi_hist) >= 2
                and raroi_hist[-1] < raroi_hist[-2]
                and conf > 0.5
            ):
                conf *= 0.5
            reliability = float(self.tracker.synergy_reliability()) * conf
            safety_hist = self.tracker.metrics_history.get(
                "synergy_safety_rating", [1.0]
            )
            resilience_hist = self.tracker.metrics_history.get(
                "synergy_resilience", [1.0]
            )
            safety = float(safety_hist[-1]) if safety_hist else 1.0
            resilience = float(resilience_hist[-1]) if resilience_hist else 1.0
            return raroi, reliability, safety, resilience

        if self.telemetry is not None:
            history = self.telemetry.fetch_history(workflow_id, scope=self.scope)
            if history:
                last = history[-1]
                raroi = float(last.get("actual") or 0.0)
                conf = float(last.get("confidence") or 1.0)
                if bool(last.get("drift_flag")) and conf > 0.5:
                    conf *= 0.5
                pred = last.get("predicted")
                act = last.get("actual")
                try:
                    if (
                        pred is not None
                        and act is not None
                        and float(act) < float(pred)
                        and conf > 0.5
                    ):
                        conf *= 0.5
                except Exception:
                    pass
                deltas = last.get("scenario_deltas") or {}
                reliability = float(deltas.get("synergy_reliability", 1.0)) * conf
                safety = float(deltas.get("synergy_safety_rating", 1.0))
                resilience = float(deltas.get("synergy_resilience", 1.0))
                return raroi, reliability, safety, resilience

        return 0.0, 1.0, 1.0, 1.0

    # ------------------------------------------------------------------
    def readiness(self, workflow_id: str | None = None) -> float:
        """Return readiness for the latest cycle."""

        snap = self.snapshot(workflow_id)
        return snap["readiness"]

    # ------------------------------------------------------------------
    def snapshot(self, workflow_id: str | None = None) -> Dict[str, float]:
        """Return readiness and underlying metrics."""

        raroi, reliability, safety, resilience = self._latest_metrics(workflow_id)
        readiness = military_grade_readiness(raroi, reliability, safety, resilience)
        snap = {
            "raroi": raroi,
            "reliability": reliability,
            "safety": safety,
            "resilience": resilience,
            "readiness": readiness,
        }
        self.history.append(snap)
        _log_history(raroi, reliability, safety, resilience, readiness)
        return snap

    # ------------------------------------------------------------------
    def serialize_history(self) -> List[Dict[str, float]]:
        """Return the collected readiness history for this instance."""

        return list(self.history)
