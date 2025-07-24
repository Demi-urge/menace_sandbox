from __future__ import annotations

"""Predict synergy metrics using PredictionManager."""

from typing import Dict, Optional
import asyncio
import uuid

from menace.prediction_manager_bot import PredictionManager
from menace.roi_tracker import ROITracker
from menace.resilience import (
    retry_with_backoff,
    CircuitBreaker,
    CircuitOpenError,
    ResilienceError,
)
from menace.logging_utils import set_correlation_id, get_logger

_manager: PredictionManager | None = None
_tracker: ROITracker | None = None
_last_predictions: Dict[str, float] = {}
_circuit = CircuitBreaker()
logger = get_logger(__name__)


class PredictionError(ResilienceError):
    """Raised when metric prediction fails after retries."""


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
        cid = str(uuid.uuid4())
        set_correlation_id(cid)
        logger.info("collect_metrics start")
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
                    logger.warning("record_metric_prediction failed", exc_info=True)
        syn_names = [n for n in _tracker.metrics_history if n.startswith("synergy_")]
        syn_names.extend(
            n
            for n in getattr(_tracker, "synergy_metrics_history", {})
            if n.startswith("synergy_") and n not in syn_names
        )
        new_preds: Dict[str, float] = {}

        def _predict(name: str, actual: float | None) -> float:
            if name == "synergy_roi" and hasattr(_tracker, "forecast_synergy"):
                pred, _ = _tracker.forecast_synergy()
            else:
                pred = _tracker.predict_metric_with_manager(
                    _manager, name, [], actual=actual
                )
            return float(pred)

        for name in syn_names:
            history = (
                _tracker.metrics_history.get(name)
                or getattr(_tracker, "synergy_metrics_history", {}).get(name)
            )
            actual = history[-1] if history else None
            try:
                pred = retry_with_backoff(
                    lambda: _circuit.call(lambda: _predict(name, actual)),
                    attempts=3,
                    logger=logger,
                )
            except CircuitOpenError as exc:
                logger.error("prediction circuit open for %s: %s", name, exc)
                pred = 0.0
            except Exception as exc:
                logger.error("prediction failed for %s: %s", name, exc)
                pred = 0.0
            result[f"pred_{name}"] = float(pred)
            new_preds[name] = float(pred)
        _last_predictions = new_preds
        set_correlation_id(None)
    return result


async def collect_metrics_async(
    prev_roi: float, roi: float, resources: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """Asynchronous wrapper for :func:`collect_metrics`."""

    return await asyncio.to_thread(collect_metrics, prev_roi, roi, resources)

