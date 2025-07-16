from __future__ import annotations

"""Predict synergy metrics using PredictionManager."""

from typing import Dict, Optional

from menace.prediction_manager_bot import PredictionManager
from menace.roi_tracker import ROITracker

_manager: PredictionManager | None = None
_tracker: ROITracker | None = None
_last_predictions: Dict[str, float] = {}


def register(pm: PredictionManager, tracker: ROITracker | None = None) -> None:
    """Store manager and optional tracker for later use."""
    global _manager, _tracker
    _manager = pm
    _tracker = tracker


def collect_metrics(
    prev_roi: float, roi: float, resources: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """Return predicted synergy metrics."""
    result: Dict[str, float] = {}
    global _last_predictions
    if _manager and _tracker:
        # record accuracy for previous predictions
        for name, pred in list(_last_predictions.items()):
            hist = (
                _tracker.metrics_history.get(name)
                or getattr(_tracker, "synergy_metrics_history", {}).get(name)
            )
            if hist:
                try:
                    _tracker.record_metric_prediction(name, pred, hist[-1])
                except Exception:
                    pass
        syn_names = [n for n in _tracker.metrics_history if n.startswith("synergy_")]
        syn_names.extend(
            n
            for n in getattr(_tracker, "synergy_metrics_history", {})
            if n.startswith("synergy_") and n not in syn_names
        )
        new_preds: Dict[str, float] = {}
        for name in syn_names:
            history = (
                _tracker.metrics_history.get(name)
                or getattr(_tracker, "synergy_metrics_history", {}).get(name)
            )
            actual = history[-1] if history else None
            try:
                pred = _tracker.predict_metric_with_manager(
                    _manager, name, [], actual=actual
                )
            except Exception:
                pred = 0.0
            result[f"pred_{name}"] = pred
            new_preds[name] = pred
        _last_predictions = new_preds
    return result
