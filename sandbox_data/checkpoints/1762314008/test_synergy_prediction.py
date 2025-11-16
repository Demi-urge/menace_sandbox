import os
import importlib.util
import sys
import types
from pathlib import Path
import pytest

from tests.test_sandbox_section_simulations import _stub_module

if "menace.synergy_predictor" in sys.modules:
    from menace.synergy_predictor import ARIMASynergyPredictor, LSTMSynergyPredictor
    from menace.roi_tracker import ROITracker
else:  # pragma: no cover - allow running file directly
    spec = importlib.util.spec_from_file_location(
        "menace.synergy_predictor",
        Path(__file__).resolve().parents[1] / "menace" / "synergy_predictor.py",  # path-ignore
    )
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    sys.modules["menace.synergy_predictor"] = sp
    pkg.synergy_predictor = sp
    log_mod = types.ModuleType("menace.logging_utils")
    log_mod.get_logger = lambda *a, **k: None
    sys.modules["menace.logging_utils"] = log_mod
    pkg.logging_utils = log_mod
    rt_spec = importlib.util.spec_from_file_location(
        "menace.roi_tracker",
        Path(__file__).resolve().parents[1] / "roi_tracker.py",  # path-ignore
    )
    rt = importlib.util.module_from_spec(rt_spec)
    rt_spec.loader.exec_module(rt)
    pkg.roi_tracker = rt
    sys.modules["menace.roi_tracker"] = rt
    ARIMASynergyPredictor = sp.ARIMASynergyPredictor
    LSTMSynergyPredictor = sp.LSTMSynergyPredictor
    ROITracker = rt.ROITracker


def test_arima_synergy_predictor_basic():
    pred = ARIMASynergyPredictor()
    vals = [0.1 * i for i in range(12)]
    out = pred.predict(vals)
    assert isinstance(out, float)


def test_arima_synergy_predictor_auto_order(monkeypatch):
    import menace.synergy_predictor as sp

    monkeypatch.setattr(sp, "_pick_best_order", lambda vals: (2, 1, 0))

    calls = {}

    class DummyModel:
        def __init__(self, series, order):
            calls["order"] = order

        def fit(self):
            return self

        def get_forecast(self, steps=1):
            return types.SimpleNamespace(predicted_mean=[1.5])

    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima.model", types.SimpleNamespace(ARIMA=DummyModel))

    pred = ARIMASynergyPredictor(order=None)
    vals = [float(i) for i in range(11)]
    out = pred.predict(vals)
    assert out == pytest.approx(1.5)
    assert calls.get("order") == (2, 1, 0)


def test_lstm_synergy_predictor_basic():
    torch = pytest.importorskip("torch")
    pred = LSTMSynergyPredictor(seq_len=3, epochs=1)
    vals = [float(i) for i in range(12)]
    out = pred.predict(vals)
    assert isinstance(out, float)


def test_tracker_delegates_synergy_predictor(monkeypatch):
    tracker = ROITracker()
    tracker.metrics_history["synergy_roi"] = [float(i) for i in range(11)]

    class DummyPred:
        def predict(self, hist):
            return 9.9

    monkeypatch.setenv("SANDBOX_SYNERGY_MODEL", "arima")
    import menace.synergy_predictor as sp

    monkeypatch.setattr(sp, "ARIMASynergyPredictor", lambda: DummyPred())
    val = tracker.predict_synergy()
    assert val == pytest.approx(9.9)
    monkeypatch.delenv("SANDBOX_SYNERGY_MODEL", raising=False)


def test_tracker_delegates_lstm_synergy_predictor(monkeypatch):
    tracker = ROITracker()
    tracker.metrics_history["synergy_roi"] = [float(i) for i in range(11)]

    class DummyPred:
        def predict(self, hist):
            return 8.8

    monkeypatch.setenv("SANDBOX_SYNERGY_MODEL", "lstm")
    import menace.synergy_predictor as sp

    monkeypatch.setattr(sp, "torch", object())
    monkeypatch.setattr(sp, "LSTMSynergyPredictor", lambda: DummyPred())
    val = tracker.predict_synergy()
    assert val == pytest.approx(8.8)
    monkeypatch.delenv("SANDBOX_SYNERGY_MODEL", raising=False)


def test_tracker_lstm_falls_back_without_torch(monkeypatch):
    tracker = ROITracker()
    tracker.metrics_history["synergy_roi"] = [float(i) for i in range(11)]

    class DummyLSTM:
        def predict(self, hist):
            raise AssertionError("should not be called")

    class DummyARIMA:
        def predict(self, hist):
            return 3.3

    monkeypatch.setenv("SANDBOX_SYNERGY_MODEL", "lstm")
    import menace.synergy_predictor as sp

    monkeypatch.setattr(sp, "torch", None)
    monkeypatch.setattr(sp, "LSTMSynergyPredictor", lambda: DummyLSTM())
    monkeypatch.setattr(sp, "ARIMASynergyPredictor", lambda: DummyARIMA())
    val = tracker.predict_synergy()
    assert val == pytest.approx(3.3)
    monkeypatch.delenv("SANDBOX_SYNERGY_MODEL", raising=False)
