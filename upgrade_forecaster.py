"""Project near-term upgrade performance.

This helper exposes :class:`UpgradeForecaster` which uses historical
metrics collected by :class:`foresight_tracker.ForesightTracker` and a
lightweight temporal simulation to project the likely return on
investment and stability for the next few improvement cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping
import json
import math

import numpy as np

from foresight_tracker import ForesightTracker
from sandbox_runner.environment import simulate_temporal_trajectory


@dataclass
class CycleProjection:
    """Forecasted metrics for a single upcoming cycle."""

    cycle: int
    roi: float
    risk: float
    confidence: float
    decay: float


class UpgradeForecaster:
    """Combine historic trends with sandbox simulations to forecast upgrades."""

    def __init__(self, tracker: ForesightTracker, records_base: str | Path = "forecast_records") -> None:
        self.tracker = tracker
        self.records_base = Path(records_base)
        try:
            self.records_base.mkdir(parents=True, exist_ok=True)
        except Exception:
            # best effort – failures shouldn't break callers
            pass

    # ------------------------------------------------------------------
    def _recent_values(self, workflow_id: str, key: str) -> List[float]:
        profile = self.tracker.get_temporal_profile(workflow_id)
        return [float(entry.get(key, 0.0)) for entry in profile]

    # ------------------------------------------------------------------
    def forecast(self, workflow_id: str, patch: Iterable[str] | str, cycles: int = 5) -> List[CycleProjection]:
        """Return projections for the next ``cycles`` (3–5 typical).

        Parameters
        ----------
        workflow_id:
            Identifier of the workflow being patched.
        patch:
            Workflow steps or identifier executed for the temporal simulation.
        cycles:
            Number of future cycles to forecast. Clamped to the range 3–5.
        """

        cycles = max(3, min(5, int(cycles)))

        # Pull historical metrics and compute trend characteristics
        slope, _, stability = self.tracker.get_trend_curve(str(workflow_id))
        recent_roi = self._recent_values(str(workflow_id), "roi_delta")
        variance = float(np.var(recent_roi)) if recent_roi else 0.0
        latest = self.tracker.get_temporal_profile(str(workflow_id))[-1] if recent_roi else {}
        base_roi = float(latest.get("roi_delta", 0.0))
        base_conf = float(latest.get("confidence", 0.0))
        resilience = float(latest.get("resilience", 0.0))
        base_decay = float(latest.get("scenario_degradation", 0.0))

        # Run a simulated trajectory of the patched workflow
        roi_tracker = simulate_temporal_trajectory(str(workflow_id), patch, foresight_tracker=self.tracker)
        roi_hist = getattr(roi_tracker, "roi_history", [])
        sim_roi = float(roi_hist[-1]) if roi_hist else base_roi
        metrics_hist = getattr(roi_tracker, "metrics_history", {})
        ent_hist = metrics_hist.get("synergy_shannon_entropy", [])
        entropy = float(ent_hist[-1]) if ent_hist else 0.0

        projections: List[CycleProjection] = []
        std = math.sqrt(variance)
        for i in range(1, cycles + 1):
            # Linear trend with minor pull towards simulated ROI
            expected_roi = base_roi + slope * i + (sim_roi - base_roi) * (i / cycles)
            risk = max(0.0, min(1.0, (1.0 - resilience) + std + entropy * 0.1))
            confidence = max(0.0, min(1.0, base_conf * stability * (1.0 - risk)))
            decay = base_decay + entropy * 0.05 * i
            projections.append(CycleProjection(i, expected_roi, risk, confidence, decay))

        # Persist the forecast for external inspection
        out_path = self.records_base / f"{workflow_id}.json"
        try:
            out_path.write_text(json.dumps([p.__dict__ for p in projections], indent=2))
        except Exception:
            pass
        return projections
