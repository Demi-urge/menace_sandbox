"""Utilities for tracking workflow metrics and assessing trend stability."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Mapping, Tuple

import numpy as np


class ForesightTracker:
    """Maintain recent cycle metrics for workflows and evaluate stability."""

    def __init__(self, window: int = 10, volatility_threshold: float = 1.0) -> None:
        """Create a new tracker.

        Parameters
        ----------
        window:
            Number of recent cycles to retain per workflow.
        volatility_threshold:
            Maximum standard deviation across stored metrics considered stable.
        """

        self.window = window
        self.volatility_threshold = volatility_threshold
        self.history: Dict[str, Deque[Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    def record_cycle_metrics(self, workflow_id: str, metrics: Mapping[str, float]) -> None:
        """Append ``metrics`` for ``workflow_id`` and cap history length."""

        entry = {k: float(v) for k, v in metrics.items()}
        queue = self.history.setdefault(workflow_id, deque(maxlen=self.window))
        queue.append(entry)

    # ------------------------------------------------------------------
    def get_trend_curve(self, workflow_id: str) -> Tuple[float, float, float]:
        """Return slope, second derivative and average window stability.

        This is part of the public API so callers can inspect the raw trend
        information used by :meth:`is_stable`. The trend is computed from the
        mean of the metric values for each recorded cycle.
        ``avg_window_stability`` equals ``1 / (1 + std)`` where ``std`` is the
        standard deviation over the retained window.
        """

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            return 0.0, 0.0, 0.0

        averages = np.array(
            [np.mean(list(entry.values())) for entry in data], dtype=float
        )
        x = np.arange(len(averages))
        slope = float(np.polyfit(x, averages, 1)[0])

        if len(averages) >= 3:
            coeff = np.polyfit(x, averages, 2)[0]
            second_derivative = float(2.0 * coeff)
        else:
            second_derivative = 0.0

        window = min(self.window, len(averages))
        std = float(np.std(averages[-window:], ddof=1)) if window > 1 else 0.0
        avg_stability = 1.0 / (1.0 + std)
        return slope, second_derivative, avg_stability

    # ------------------------------------------------------------------
    def is_stable(self, workflow_id: str) -> bool:
        """Return ``True`` when slope is positive and volatility is low.

        This public helper combines :meth:`get_trend_curve` with a volatility
        check to determine whether ``workflow_id`` is operating within the
        allowed threshold.
        """

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            return False

        slope, _, _ = self.get_trend_curve(workflow_id)
        all_values = np.array([v for entry in data for v in entry.values()], dtype=float)
        std = float(np.std(all_values, ddof=1)) if all_values.size > 1 else 0.0
        return slope > 0 and std < self.volatility_threshold


__all__ = ["ForesightTracker"]

