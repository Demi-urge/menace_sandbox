from __future__ import annotations

"""Predict key metrics using PredictionManager."""

from typing import Dict, Optional, Iterable

try:
    import numpy as np  # type: ignore
    from sklearn.linear_model import LinearRegression  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    LinearRegression = None  # type: ignore

from menace.prediction_manager_bot import PredictionManager
from menace.roi_tracker import ROITracker
from menace.coding_bot_interface import self_coding_managed

_manager: PredictionManager | None = None
_tracker: ROITracker | None = None
_lt_bot: "LongTermLucrativityBot" | None = None


class _StubRegistry:
    def register_bot(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        return None

    def update_bot(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        return None


class _StubDataBot:
    def reload_thresholds(self, _name: str):  # pragma: no cover - stub
        return type("_T", (), {})()


_REGISTRY_STUB = _StubRegistry()
_DATA_BOT_STUB = _StubDataBot()

# metrics collected in ``_sandbox_cycle_runner``
_METRICS = (
    "security_score",
    "safety_rating",
    "adaptability",
    "antifragility",
    "shannon_entropy",
    "efficiency",
    "flexibility",
    "projected_lucrativity",
    "profitability",
    "patch_complexity",
    "energy_consumption",
    "resilience",
    "network_latency",
    "throughput",
    "risk_index",
    "maintainability",
    "code_quality",
    "recovery_time",
    "discrepancy_count",
    "gpu_usage",
    "cpu_usage",
    "memory_usage",
    "long_term_lucrativity",
)


@self_coding_managed(bot_registry=_REGISTRY_STUB, data_bot=_DATA_BOT_STUB)
class LongTermLucrativityBot:
    """Predict long-term lucrativity from ROI and lucrativity history."""

    prediction_profile = {
        "metric": ["long_term_lucrativity"],
        "scope": ["lucrativity"],
        "risk": ["medium"],
    }

    def __init__(self, tracker: ROITracker) -> None:
        self.tracker = tracker

    def predict_metric(
        self, name: str, _features: Iterable[float] | None = None
    ) -> float:
        if name != "long_term_lucrativity":
            return 0.0
        roi_hist = self.tracker.roi_history
        lucr_hist = self.tracker.metrics_history.get("projected_lucrativity", [])
        n = min(len(roi_hist), len(lucr_hist))
        if n < 2:
            return float(lucr_hist[-1]) if lucr_hist else 0.0
        if np is not None and LinearRegression is not None:
            try:
                X = np.column_stack(
                    [np.arange(n), np.array(roi_hist[-n:], dtype=float)]
                )
                y = np.array(lucr_hist[-n:], dtype=float)
                model = LinearRegression().fit(X, y)
                pred = model.predict([[n + 10, roi_hist[-1]]])[0]
                return float(pred)
            except Exception:
                pass
        return float(sum(lucr_hist[-n:]) / n)


_last_predictions: Dict[str, float] = {}


def register(pm: PredictionManager, tracker: ROITracker | None = None) -> None:
    """Store manager and optional tracker for later use."""
    global _manager, _tracker, _lt_bot
    _manager = pm
    _tracker = tracker
    if pm and tracker:
        try:
            _lt_bot = LongTermLucrativityBot(tracker)
            pm.register_bot(
                _lt_bot,
                {
                    "metric": ["long_term_lucrativity"],
                    "scope": ["lucrativity"],
                    "risk": ["medium"],
                },
            )
        except Exception:
            _lt_bot = None


def collect_metrics(
    prev_roi: float, roi: float, resources: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """Return predicted metrics and record accuracy if possible."""
    result: Dict[str, float] = {}
    global _last_predictions
    if _manager and _tracker:
        # Record accuracy of previous predictions when actual values exist
        for name, pred in list(_last_predictions.items()):
            hist = _tracker.metrics_history.get(name, [])
            if hist:
                try:
                    _tracker.record_metric_prediction(name, pred, hist[-1])
                except Exception:
                    pass

        features = [prev_roi, roi]
        new_preds: Dict[str, float] = {}
        for metric in _METRICS:
            try:
                pred = _tracker.predict_metric_with_manager(
                    _manager,
                    metric,
                    features,
                )
            except Exception:
                pred = 0.0
            result[f"pred_{metric}"] = pred
            new_preds[metric] = pred

        # Predict synergy metrics using tracker-only methods
        syn_names = [n for n in _tracker.metrics_history if n.startswith("synergy_")]
        syn_names.extend(
            n
            for n in getattr(_tracker, "synergy_metrics_history", {})
            if n.startswith("synergy_") and n not in syn_names
        )
        for name in syn_names:
            try:
                if name == "synergy_roi":
                    pred = _tracker.predict_synergy()
                else:
                    base = name[len("synergy_"):]
                    pred = _tracker.predict_synergy_metric(base, manager=_manager)
            except Exception:
                pred = 0.0
            result[f"pred_{name}"] = pred
            new_preds[name] = pred
            hist = _tracker.metrics_history.get(name, [])
            if hist:
                try:
                    _tracker.record_metric_prediction(name, pred, hist[-1])
                except Exception:
                    pass
        if _lt_bot:
            try:
                lt_val = _lt_bot.predict_metric("long_term_lucrativity")
            except Exception:
                lt_val = 0.0
            result["long_term_lucrativity"] = lt_val
        _last_predictions = new_preds
    return result
