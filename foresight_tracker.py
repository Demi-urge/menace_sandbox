from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple
import statistics


class ForesightTracker:
    """Track workflow foresight metrics and evaluate stability."""

    def __init__(self, max_cycles: int = 10) -> None:
        self.max_cycles = max_cycles
        self.history: Dict[str, Deque[Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    def record_cycle_metrics(self, workflow_id: str, metrics: Dict[str, float]) -> None:
        """Record metrics for a workflow cycle.

        Parameters
        ----------
        workflow_id:
            Identifier for the workflow.
        metrics:
            Dictionary containing ``roi_delta``, ``raroi_delta``,
            ``confidence``, ``resilience`` and ``scenario_degradation``.
        """

        entry = {
            key: float(metrics.get(key, 0.0))
            for key in (
                "roi_delta",
                "raroi_delta",
                "confidence",
                "resilience",
                "scenario_degradation",
            )
        }
        q = self.history.setdefault(workflow_id, deque(maxlen=self.max_cycles))
        q.append(entry)

    # ------------------------------------------------------------------
    def get_trend_curve(self, workflow_id: str) -> Tuple[float, float, float]:
        """Return slope, second derivative and rolling stability for ROI delta."""

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            return 0.0, 0.0, 0.0

        roi = [entry["roi_delta"] for entry in data]
        n = len(roi)
        x = list(range(n))
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(roi)
        denom = sum((xi - mean_x) ** 2 for xi in x)
        slope = (
            sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, roi)) / denom
            if denom
            else 0.0
        )

        first_diff = [roi[i + 1] - roi[i] for i in range(n - 1)]
        second_diff = [first_diff[i + 1] - first_diff[i] for i in range(len(first_diff) - 1)]
        second_derivative = statistics.mean(second_diff) if second_diff else 0.0
        rolling_stability = statistics.stdev(roi) if len(roi) > 1 else 0.0
        return slope, second_derivative, rolling_stability

    # ------------------------------------------------------------------
    def is_stable(self, workflow_id: str, volatility_threshold: float = 1.0) -> bool:
        """Check if ROI trend is positive and volatility below threshold."""

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            return False

        slope, _, volatility = self.get_trend_curve(workflow_id)
        return slope > 0 and volatility < volatility_threshold


__all__ = ["ForesightTracker"]
