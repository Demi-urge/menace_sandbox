import sys
import importlib.util
import types
from pathlib import Path

class DummyPM:
    registry = {}
    def get_prediction_bots_for(self, name: str):
        return []
    def assign_prediction_bots(self, tracker):
        return []

pm_mod = types.ModuleType("menace.prediction_manager_bot")
pm_mod.PredictionManager = DummyPM
sys.modules.setdefault("menace.prediction_manager_bot", pm_mod)

stub = types.ModuleType("stub")
stub.stats = types.SimpleNamespace()
stub.isscalar = lambda x: isinstance(x, (int, float))
stub.bool_ = bool
sys.modules.setdefault("numpy", stub)
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
lm_stub = types.ModuleType("sklearn.linear_model")
lm_stub.LinearRegression = lambda *a, **k: types.SimpleNamespace(fit=lambda *a, **k: None, predict=lambda X: [0.0])
pf_stub = types.ModuleType("sklearn.preprocessing")
pf_stub.PolynomialFeatures = lambda *a, **k: types.SimpleNamespace(fit_transform=lambda X: X)
sys.modules.setdefault("sklearn.linear_model", lm_stub)
sys.modules.setdefault("sklearn.preprocessing", pf_stub)

spec = importlib.util.spec_from_file_location("menace.roi_tracker", Path("roi_tracker.py"))  # path-ignore
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
sys.modules.setdefault("menace.roi_tracker", rt)

import plugins.synergy_predict as sp
from menace.roi_tracker import ROITracker


def test_forecast_synergy_used(monkeypatch):
    tracker = ROITracker()
    tracker.metrics_history["synergy_roi"] = [0.1, 0.2]
    tracker.synergy_metrics_history["synergy_roi"] = [0.1, 0.2]

    calls = []
    def pred(manager, name, features, actual=None, bot_name=None):
        calls.append(name)
        return 0.0
    monkeypatch.setattr(tracker, "predict_metric_with_manager", pred)
    monkeypatch.setattr(tracker, "forecast_synergy", lambda: (0.5, (0.0, 1.0)))

    sp.register(DummyPM(), tracker)
    result = sp.collect_metrics(0.0, 0.2, None)
    assert result["pred_synergy_roi"] == 0.5
    assert "synergy_roi" not in calls


def test_synergy_fallback_to_manager(monkeypatch):
    class DummyTracker:
        def __init__(self):
            self.metrics_history = {"synergy_roi": [0.1, 0.2]}
            self.synergy_metrics_history = {"synergy_roi": [0.1, 0.2]}
        def predict_metric_with_manager(self, manager, name, features, actual=None, bot_name=None):
            calls.append(name)
            return 0.7

    tracker = DummyTracker()

    calls = []

    sp.register(DummyPM(), tracker)
    result = sp.collect_metrics(0.0, 0.2, None)
    assert result["pred_synergy_roi"] == 0.7
    assert calls == ["synergy_roi"]
