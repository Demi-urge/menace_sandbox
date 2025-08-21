"""Utilities for tracking workflow metrics and assessing trend stability."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Mapping, Tuple

import numpy as np
import yaml


class ForesightTracker:
    """Maintain recent cycle metrics for workflows and evaluate stability."""

    def __init__(
        self,
        max_cycles: int = 10,
        volatility_threshold: float = 1.0,
        window: int | None = None,
        N: int | None = None,
        template_config_path: str | Path = "configs/foresight_templates.yaml",
    ) -> None:
        """Create a new tracker.

        Parameters
        ----------
        max_cycles, window, N:
            Aliases specifying the number of recent cycles to retain per
            workflow.
        volatility_threshold:
            Maximum standard deviation across stored metrics considered stable.
        """

        if window is not None:
            max_cycles = window
        if N is not None:
            max_cycles = N

        self.max_cycles = max_cycles
        self.volatility_threshold = volatility_threshold
        self.history: Dict[str, Deque[Dict[str, float]]] = {}
        self.template_config_path = Path(template_config_path)
        self.templates: Dict[str, list[float]] | None = None

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
    def capture_from_roi(self, tracker: "ROITracker" | None, workflow_id: str) -> None:
        """Record metrics extracted from an :class:`ROITracker` instance.

        Parameters
        ----------
        tracker:
            ROI tracker providing ``roi_history``, ``raroi_history``,
            ``confidence_history`` and ``metrics_history``.
        workflow_id:
            Identifier of the workflow for which the metrics should be
            recorded.

        The helper pulls the latest ROI delta, RAROI delta, resilience,
        confidence and scenario degradation metrics from ``tracker`` and
        forwards them to :meth:`record_cycle_metrics`.
        """

        if tracker is None:
            return

        try:
            roi_hist = getattr(tracker, "roi_history", [])
            history = self.history.get(workflow_id)
            profile_map = getattr(self, "workflow_profiles", None)
            if not isinstance(profile_map, Mapping):
                profile_map = getattr(self, "profile_map", {})
            profile = profile_map.get(workflow_id, workflow_id)
            real_roi = float(roi_hist[-1]) if roi_hist else 0.0
            if self.is_cold_start(workflow_id):
                cycles = len(history) if history else 0
                template_val = self._get_template_curve(profile, cycles)
                if self.templates and profile in self.templates:
                    alpha = min(cycles / 5.0, 1.0)
                    effective_roi = alpha * real_roi + (1.0 - alpha) * template_val
                    roi_delta = effective_roi
                    raroi_delta = effective_roi
                else:
                    roi_delta = real_roi
                    raroi_hist = getattr(tracker, "raroi_history", [])
                    if len(raroi_hist) >= 2:
                        raroi_delta = float(raroi_hist[-1] - raroi_hist[-2])
                    elif raroi_hist:
                        raroi_delta = float(raroi_hist[-1])
                    else:
                        raroi_delta = 0.0
            else:
                roi_delta = real_roi
                raroi_hist = getattr(tracker, "raroi_history", [])
                if len(raroi_hist) >= 2:
                    raroi_delta = float(raroi_hist[-1] - raroi_hist[-2])
                elif raroi_hist:
                    raroi_delta = float(raroi_hist[-1])
                else:
                    raroi_delta = 0.0

            conf_hist = getattr(tracker, "confidence_history", [])
            confidence = float(conf_hist[-1]) if conf_hist else 0.0

            metrics_hist = getattr(tracker, "metrics_history", {})
            res_hist = metrics_hist.get("synergy_resilience") or metrics_hist.get(
                "resilience", []
            )
            resilience = float(res_hist[-1]) if res_hist else 0.0

            try:
                scenario_deg = float(
                    getattr(tracker, "scenario_degradation", lambda: 0.0)()
                )
            except Exception:
                scenario_deg = 0.0

            self.record_cycle_metrics(
                workflow_id,
                {
                    "roi_delta": roi_delta,
                    "raroi_delta": raroi_delta,
                    "confidence": confidence,
                    "resilience": resilience,
                    "scenario_degradation": scenario_deg,
                },
            )
        except Exception:
            # best effort; failing to record foresight metrics shouldn't break
            # the calling workflow
            pass

    # ------------------------------------------------------------------
    def is_cold_start(self, workflow_id: str) -> bool:
        """Return ``True`` when insufficient ROI history exists."""

        history = self.history.get(workflow_id)
        if not history or len(history) < 3:
            return True
        if not any("roi_delta" in entry for entry in history):
            return True
        return False

    # ------------------------------------------------------------------
    def _get_template_curve(self, profile: str, cycle_index: int) -> float:
        """Return synthetic ROI for ``profile`` at ``cycle_index``."""

        if self.templates is None:
            try:
                with self.template_config_path.open("r", encoding="utf8") as fh:
                    self.templates = yaml.safe_load(fh) or {}
            except Exception:
                self.templates = {}
        curve = self.templates.get(profile, []) if self.templates else []
        if cycle_index < 0:
            cycle_index = 0
        if cycle_index < len(curve):
            return float(curve[cycle_index])
        return float(curve[-1]) if curve else 0.0

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

        Only the retained window for each workflow is serialised.  History is
        emitted in chronological order so that the first element of each list is
        the oldest entry.
        """

        return {
            "window": self.max_cycles,
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
        data: Mapping,
        N: int | None = None,
        volatility_threshold: float | None = None,
        max_cycles: int | None = None,
    ) -> "ForesightTracker":
        """Reconstruct a tracker from :meth:`to_dict` output.

        Parameters
        ----------
        data:
            Mapping produced by :meth:`to_dict`.
        N:
            Optional override for the number of history entries to retain per
            workflow.  When ``None`` the value stored in ``data`` is used or the
            class default (``10``) if unavailable.
        volatility_threshold:
            Optional override for the volatility threshold.  When ``None`` the
            value stored in ``data`` is used or the class default (``1.0``) if
            missing.
        """

        if max_cycles is not None and N is None:
            N = max_cycles
        if N is None:
            N = int(data.get("window", data.get("N", 10)))
        if volatility_threshold is None:
            volatility_threshold = float(data.get("volatility_threshold", 1.0))

        tracker = cls(max_cycles=N, volatility_threshold=volatility_threshold)

        raw_history = data.get("history", {})
        for wf_id, entries in raw_history.items():
            queue: Deque[Dict[str, float]] = deque(maxlen=tracker.max_cycles)
            for entry in list(entries)[-tracker.max_cycles:]:
                queue.append({k: float(v) for k, v in entry.items()})
            tracker.history[wf_id] = queue

        return tracker


__all__ = ["ForesightTracker"]

