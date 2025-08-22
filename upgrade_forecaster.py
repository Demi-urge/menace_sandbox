"""Project patch impact over forthcoming improvement cycles."""

from __future__ import annotations

from typing import Iterable, List, Dict, Tuple

import numpy as np

from foresight_tracker import ForesightTracker
from sandbox_runner.environment import simulate_temporal_trajectory


class UpgradeForecaster:
    """Forecast ROI and stability for potential upgrades.

    Parameters
    ----------
    tracker:
        Instance providing historic metrics.
    horizon:
        Number of cycles to forecast. Clamped to the range 3–5.
    """

    def __init__(self, tracker: ForesightTracker, horizon: int = 5) -> None:
        self.tracker = tracker
        self.horizon = max(3, min(5, int(horizon)))

    def forecast(
        self,
        workflow_id: str,
        patch: Iterable[str] | str,
        cycles: int | None = None,
    ) -> Tuple[List[Dict[str, float]], float]:
        """Return cycle projections and overall forecast confidence."""

        cycles = self.horizon if cycles is None else max(3, min(5, int(cycles)))

        # Simulate the patched workflow to obtain prospective metrics
        roi_tracker = simulate_temporal_trajectory(
            str(workflow_id), patch, foresight_tracker=self.tracker
        )

        roi_hist = getattr(roi_tracker, "roi_history", []) or []
        conf_hist = getattr(roi_tracker, "confidence_history", []) or []
        metrics_hist = getattr(roi_tracker, "metrics_history", {}) or {}
        entropy_hist = metrics_hist.get("synergy_shannon_entropy", []) or []
        risk_hist = metrics_hist.get("synergy_risk_index", []) or []

        slope, intercept, stability = self.tracker.get_trend_curve(str(workflow_id))

        projections: List[Dict[str, float]] = []
        for i in range(1, cycles + 1):
            sim_roi = roi_hist[i - 1] if i - 1 < len(roi_hist) else (
                roi_hist[-1] if roi_hist else 0.0
            )
            trend_roi = intercept + slope * i
            roi = trend_roi + sim_roi

            sim_conf = conf_hist[i - 1] if i - 1 < len(conf_hist) else (
                conf_hist[-1] if conf_hist else 0.0
            )
            confidence = max(0.0, min(1.0, sim_conf * stability))

            entropy = entropy_hist[i - 1] if i - 1 < len(entropy_hist) else (
                entropy_hist[-1] if entropy_hist else 0.0
            )
            sim_risk = risk_hist[i - 1] if i - 1 < len(risk_hist) else (
                risk_hist[-1] if risk_hist else 0.0
            )
            risk = max(0.0, min(1.0, sim_risk + (1.0 - stability)))

            decay = entropy * 0.1 * i

            projections.append(
                {
                    "cycle": float(i),
                    "roi": float(roi),
                    "risk": float(risk),
                    "confidence": float(confidence),
                    "decay": float(decay),
                }
            )

        variance = float(np.var([p["roi"] for p in projections])) if projections else 0.0
        samples = len(self.tracker.history.get(str(workflow_id), []))
        forecast_confidence = (samples / (samples + 1.0)) * (1.0 / (1.0 + variance))

        return projections, float(forecast_confidence)
