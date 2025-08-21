"""Utilities for tracking workflow metrics and assessing trend stability."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Mapping, Tuple

import numpy as np


class ForesightTracker:
    """Maintain recent cycle metrics for workflows and evaluate stability."""

    def __init__(
        self,
        window: int = 10,
        volatility_threshold: float = 1.0,
        N: int | None = None,
    ) -> None:
        """Create a new tracker.

        Parameters
        ----------
        window, N:
            ``N`` is an alias for ``window`` specifying the number of recent
            cycles to retain per workflow.
        volatility_threshold:
            Maximum standard deviation across stored metrics considered stable.
        """

        if N is not None:
            window = N

        self.max_cycles = window
        self.volatility_threshold = volatility_threshold
        self.history: Dict[str, Deque[Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    @property
    def window(self) -> int:
        """Legacy alias for :attr:`max_cycles`.

        The tracker historically exposed the maximum retention window under
        the name ``window``.  The public attribute :attr:`max_cycles` now holds
        this value directly; this property preserves backwards compatibility.
        """

        return self.max_cycles

    # ------------------------------------------------------------------
    def record_cycle_metrics(self, workflow_id: str, metrics: Mapping[str, float]) -> None:
        """Append ``metrics`` for ``workflow_id`` and cap history length."""

        entry = {k: float(v) for k, v in metrics.items()}
        queue = self.history.setdefault(workflow_id, deque(maxlen=self.max_cycles))
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

        window = min(self.max_cycles, len(averages))
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

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON‑serialisable representation of the tracker.

        The history for each workflow is emitted in chronological order so that
        the oldest entry appears first in the list.  Only the currently retained
        window is serialised.  The ``max_cycles`` field mirrors the old
        ``window`` key for backwards compatibility.
        """

        return {
            "max_cycles": self.max_cycles,
            "window": self.max_cycles,  # legacy key for backwards compatibility
            "volatility_threshold": self.volatility_threshold,
            "history": {
                wf_id: [
                    {k: float(v) for k, v in entry.items()}
                    for entry in list(entries)
                ]
                for wf_id, entries in self.history.items()
            },
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: dict,
        window: int | None = None,
        volatility_threshold: float | None = None,
        N: int | None = None,
    ) -> "ForesightTracker":
        """Reconstruct a tracker from :meth:`to_dict` output.

        Parameters
        ----------
        data:
            Dictionary produced by :meth:`to_dict`.
        window, N:
            Optional override for the maximum number of entries to keep per
            workflow. ``N`` is an alias for ``window``. When both are ``None``
            the value stored in ``data`` is used or the class default (``10``)
            if unavailable.
        volatility_threshold:
            Optional override for the volatility threshold.  When ``None`` the
            value stored in ``data`` is used or the class default (``1.0``) if
            missing.
        """

        if N is not None:
            window = N
        if window is None:
            window = int(
                data.get(
                    "max_cycles",
                    data.get("N", data.get("window", 10)),
                )
            )
        if volatility_threshold is None:
            volatility_threshold = float(data.get("volatility_threshold", 1.0))

        tracker = cls(window=window, volatility_threshold=volatility_threshold)
        raw_history = data.get("history", {})
        for wf_id, entries in raw_history.items():
            queue: Deque[Dict[str, float]] = deque(maxlen=tracker.max_cycles)
            for entry in list(entries)[-tracker.max_cycles:]:
                queue.append({k: float(v) for k, v in entry.items()})
            tracker.history[wf_id] = queue
        return tracker


__all__ = ["ForesightTracker"]

