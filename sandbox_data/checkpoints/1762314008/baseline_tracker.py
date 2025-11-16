from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Iterable, Mapping


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
        # ``_last_entropy`` stores the most recent entropy value prior to the
        # latest update so that callers can easily compute deltas between
        # successive cycles.  This is preferable to reaching into the history
        # deque and simplifies overfitting checks in callers.
        self._last_entropy: float = 0.0
        for name, values in initial.items():
            hist = self._history.setdefault(name, deque(maxlen=window))
            for v in values:
                hist.append(float(v))
            if name == "entropy" and values:
                # When initial entropy history is provided the last value is
                # considered the "previous" entropy for delta calculations.
                self._last_entropy = float(list(values)[-1])

    # ------------------------------------------------------------------
    def update(self, *, record_momentum: bool = True, **metrics: float) -> None:
        """Record new metric values.

        Parameters
        ----------
        record_momentum:
            When ``True`` (the default) the current momentum value is recorded
            after processing ``metrics`` so that moving averages and deviations
            can be computed for the momentum series.  Set to ``False`` when
            adding auxiliary metrics that should not advance the momentum
            history.
        **metrics:
            Mapping of metric name to numeric value.

        Notes
        -----
        For every metric the delta from the current moving average is recorded
        under ``<metric>_delta`` before the new value is added to the history.
        This includes commonly used metrics like ``roi`` and ``pass_rate``.
        ``roi`` deltas also update the internal success history used for
        momentum calculations.
        """

        for name, value in metrics.items():
            hist = self._history.setdefault(name, deque(maxlen=self.window))
            avg = sum(hist) / len(hist) if hist else 0.0
            delta_hist = self._history.setdefault(
                f"{name}_delta", deque(maxlen=self.window)
            )
            delta = float(value) - avg
            delta_hist.append(delta)
            if name == "roi":
                self._success_history.append(delta > 0)
            if name == "entropy":
                # Record the previous entropy value so callers can compute the
                # change between consecutive updates without inspecting the
                # history deque directly.
                self._last_entropy = hist[-1] if hist else self._last_entropy
            hist.append(float(value))

        if record_momentum:
            # Record current momentum so moving averages and deviations can be
            # computed like other metrics.  This is appended after processing the
            # provided metrics so ``roi`` updates influence the momentum history in
            # the same cycle.
            momentum_hist = self._history.setdefault(
                "momentum", deque(maxlen=self.window)
            )
            momentum_hist.append(self.momentum)

    # ------------------------------------------------------------------
    def get(self, metric: str) -> float:
        """Return the moving average for *metric*."""
        hist = self._history.get(metric)
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    # ------------------------------------------------------------------
    def current(self, metric: str) -> float:
        """Return the most recent value recorded for *metric*."""
        hist = self._history.get(metric)
        if not hist:
            return 0.0
        return hist[-1]

    # ------------------------------------------------------------------
    def delta(self, metric: str) -> float:
        """Return the deviation of the latest value from ``get(metric)``."""
        hist = self._history.get(metric)
        if not hist:
            return 0.0
        return hist[-1] - self.get(metric)

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
    def to_state(self) -> Dict[str, object]:
        """Serialise tracker histories including success records."""

        return {
            "history": self.to_dict(),
            "success": list(self._success_history),
            "last_entropy": self._last_entropy,
        }

    # ------------------------------------------------------------------
    def load_state(self, state: Mapping[str, object]) -> None:
        """Restore histories from :meth:`to_state` output."""

        history = state.get("history", {})
        if isinstance(history, Mapping):
            self._history = {
                k: deque((float(v) for v in values), maxlen=self.window)
                for k, values in history.items()
            }
        success = state.get("success", [])
        self._success_history = deque(
            (bool(x) for x in success), maxlen=self.window
        )
        self._last_entropy = float(state.get("last_entropy", self._last_entropy))

    # ------------------------------------------------------------------
    def delta_history(self, metric: str) -> list[float]:
        """Return recorded deltas for *metric* if available."""
        return list(self._history.get(f"{metric}_delta", []))

    # ------------------------------------------------------------------
    @property
    def last_entropy(self) -> float:
        """Return the entropy value from the previous update."""
        return self._last_entropy

    # ------------------------------------------------------------------
    @property
    def entropy_delta(self) -> float:
        """Return the change in entropy since the previous update."""
        hist = self._history.get("entropy")
        if not hist:
            return 0.0
        return hist[-1] - self._last_entropy

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
