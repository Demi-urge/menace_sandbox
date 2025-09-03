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

    def __init__(
        self,
        window: int = 10,
        metrics: Iterable[str] | None = None,
        **initial: Iterable[float],
    ) -> None:
        self.window = window
        self._history: Dict[str, Deque[float]] = {
            m: deque(maxlen=window) for m in (metrics or [])
        }
        self._success_history: Deque[bool] = deque(maxlen=window)
        for name, values in initial.items():
            hist = self._history.setdefault(name, deque(maxlen=window))
            for v in values:
                hist.append(float(v))

    # ------------------------------------------------------------------
    def update(self, **metrics: float) -> None:
        """Record new metric values.

        Parameters
        ----------
        **metrics:
            Mapping of metric name to numeric value.

        Notes
        -----
        Updating the ``roi`` or ``pass_rate`` metrics also records the delta
        from the previous value under ``roi_delta`` or ``pass_rate_delta`` to
        provide a history of per-cycle changes.
        """

        for name, value in metrics.items():
            hist = self._history.setdefault(name, deque(maxlen=self.window))
            if name == "roi":
                prev = hist[-1] if hist else 0.0
                delta_hist = self._history.setdefault(
                    "roi_delta", deque(maxlen=self.window)
                )
                delta = float(value) - prev
                delta_hist.append(delta)
                self._success_history.append(delta > 0)
            elif name == "pass_rate":
                prev = hist[-1] if hist else 0.0
                delta_hist = self._history.setdefault(
                    "pass_rate_delta", deque(maxlen=self.window)
                )
                delta = float(value) - prev
                delta_hist.append(delta)
            elif name == "entropy":
                avg = sum(hist) / len(hist) if hist else 0.0
                delta_hist = self._history.setdefault(
                    "entropy_delta", deque(maxlen=self.window)
                )
                delta_hist.append(float(value) - avg)
            hist.append(float(value))

        # Record current momentum so moving averages and deviations can be
        # computed like other metrics.  This is appended after processing the
        # provided metrics so ``roi`` updates influence the momentum history in
        # the same cycle.
        momentum_hist = self._history.setdefault("momentum", deque(maxlen=self.window))
        momentum_hist.append(self.momentum)

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

    # ------------------------------------------------------------------
    @property
    def success_count(self) -> int:
        """Number of positive ROI deltas within the window."""
        return sum(1 for s in self._success_history if s)

    # ------------------------------------------------------------------
    @property
    def cycle_count(self) -> int:
        """Total ROI cycles considered in the window."""
        return len(self._success_history)

    # ------------------------------------------------------------------
    @property
    def momentum(self) -> float:
        """Recent success ratio normalised by window size."""
        if not self._success_history:
            return 0.0
        return self.success_count / self.window


# Shared tracker used across self-improvement modules
TRACKER = BaselineTracker()
