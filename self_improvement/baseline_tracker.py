from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Iterable


class BaselineTracker:
    """Maintain rolling statistics for numeric metrics.

    Metrics are tracked using fixed-size deques so both the moving average and
    variance can be computed efficiently for recent values.  The tracker is
    metric-agnostic and will automatically create histories for new metric
    names on first update.
    """

    def __init__(self, window: int = 10, metrics: Iterable[str] | None = None) -> None:
        self.window = window
        self._history: Dict[str, Deque[float]] = {
            m: deque(maxlen=window) for m in (metrics or [])
        }

    # ------------------------------------------------------------------
    def update(self, **metrics: float) -> None:
        """Record new metric values.

        Parameters
        ----------
        **metrics:
            Mapping of metric name to numeric value.

        Notes
        -----
        Updating the ``roi`` metric also records the delta from the previous
        value under ``roi_delta`` to provide a history of per-cycle ROI changes.
        """

        for name, value in metrics.items():
            hist = self._history.setdefault(name, deque(maxlen=self.window))
            if name == "roi":
                prev = hist[-1] if hist else 0.0
                delta_hist = self._history.setdefault(
                    "roi_delta", deque(maxlen=self.window)
                )
                delta_hist.append(float(value) - prev)
            hist.append(float(value))

    # ------------------------------------------------------------------
    def get(self, metric: str) -> float:
        """Return the moving average for *metric*."""
        hist = self._history.get(metric)
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    # ------------------------------------------------------------------
    def variance(self, metric: str) -> float:
        """Return the population variance for *metric*."""
        hist = self._history.get(metric)
        if not hist:
            return 0.0
        mean = self.get(metric)
        return sum((x - mean) ** 2 for x in hist) / len(hist)

    # ------------------------------------------------------------------
    def std(self, metric: str) -> float:
        """Return the standard deviation for *metric*."""
        return math.sqrt(self.variance(metric))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, list[float]]:
        """Expose raw metric histories (primarily for testing)."""
        return {k: list(v) for k, v in self._history.items()}

    # ------------------------------------------------------------------
    def delta_history(self, metric: str) -> list[float]:
        """Return recorded deltas for *metric* if available."""
        return list(self._history.get(f"{metric}_delta", []))


# Shared tracker used across self-improvement modules
TRACKER = BaselineTracker()
