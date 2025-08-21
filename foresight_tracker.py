from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np


class ForesightTracker:
    """Track workflow foresight metrics and evaluate stability."""

    def __init__(
        self,
        N: int = 10,
        *,
        volatility_threshold: float = 1.0,
    ) -> None:
        """Create a tracker storing up to ``N`` cycle snapshots per workflow."""

        self.max_cycles = N
        self.volatility_threshold = volatility_threshold
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
    def get_history(self, workflow_id: str) -> List[Dict[str, float]]:
        """Return a copy of the stored history for ``workflow_id``."""

        return list(self.history.get(workflow_id, []))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, List[Dict[str, float]]]:
        """Serialize all histories to a dictionary."""

        return {wid: list(deq) for wid, deq in self.history.items()}

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Iterable[Dict[str, float]]],
        N: int = 10,
        *,
        volatility_threshold: float = 1.0,
    ) -> "ForesightTracker":
        """Create a :class:`ForesightTracker` from serialized ``data``."""

        tracker = cls(N=N, volatility_threshold=volatility_threshold)
        for wid, entries in data.items():
            tracker.history[wid] = deque(entries, maxlen=N)
        return tracker

    # ------------------------------------------------------------------
    def get_trend_curve(self, workflow_id: str) -> Tuple[float, float, float, float]:
        """Return slope, second derivative, volatility and stability for ROI delta."""

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            # No meaningful trend information
            return 0.0, 0.0, 0.0, 1.0

        roi = np.array([entry["roi_delta"] for entry in data], dtype=float)
        x = np.arange(len(roi))

        # First derivative: slope from linear regression
        slope = float(np.polyfit(x, roi, 1)[0])

        # Second derivative: curvature from quadratic fit
        if roi.size >= 3:
            a = float(np.polyfit(x, roi, 2)[0])
            second_derivative = 2.0 * a
        else:
            second_derivative = 0.0

        # Rolling volatility/stability over a window of last 5 points (or fewer)
        window = min(5, roi.size)
        volatility = float(np.std(roi[-window:], ddof=1)) if window > 1 else 0.0
        stability = 1.0 / (1.0 + volatility)
        return slope, second_derivative, volatility, stability

    # ------------------------------------------------------------------
    def is_stable(self, workflow_id: str, threshold: float | None = None) -> bool:
        """Evaluate the latest trend's slope and volatility.

        A workflow is considered stable when the most recent ROI trend is
        positive and its volatility stays below a configurable ``threshold``.
        If ``threshold`` is omitted, the value provided during initialization is
        used.  The method relies on :meth:`get_trend_curve` to obtain the
        current slope and rolling standard deviation (treated as volatility).
        """

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            return False

        slope, _, volatility, _ = self.get_trend_curve(workflow_id)
        limit = self.volatility_threshold if threshold is None else threshold
        return slope > 0 and volatility < limit


__all__ = ["ForesightTracker"]
