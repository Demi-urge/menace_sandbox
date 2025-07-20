import os
import sys
import types
import pytest

os.environ["MENACE_LIGHT_IMPORTS"] = "1"

stub = types.ModuleType("stub")
stats_stub = types.SimpleNamespace(
    pearsonr=lambda *a, **k: (0.0, 0.0),
    t=lambda *a, **k: 0.0,
)
stub.stats = stats_stub
stub.isscalar = lambda x: isinstance(x, (int, float))
stub.bool_ = bool
sys.modules.setdefault("scipy", stub)
sys.modules.setdefault("scipy.stats", stats_stub)
sys.modules.setdefault("yaml", stub)
sys.modules.setdefault("numpy", stub)
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
lm_stub = types.ModuleType("sklearn.linear_model")
lm_stub.LinearRegression = lambda *a, **k: types.SimpleNamespace(
    fit=lambda *a, **k: None,
    predict=lambda X: [0.0],
)
pf_stub = types.ModuleType("sklearn.preprocessing")
pf_stub.PolynomialFeatures = lambda *a, **k: types.SimpleNamespace(
    fit_transform=lambda X: X
)
sys.modules.setdefault("sklearn.linear_model", lm_stub)
sys.modules.setdefault("sklearn.preprocessing", pf_stub)
import importlib.util
from pathlib import Path

class _StubPM:
    registry: dict = {}

    def get_prediction_bots_for(self, name: str) -> list:
        return []

    def assign_prediction_bots(self, tracker):
        return []

pm_mod = types.ModuleType("menace.prediction_manager_bot")
pm_mod.PredictionManager = _StubPM
sys.modules.setdefault("menace.prediction_manager_bot", pm_mod)

spec = importlib.util.spec_from_file_location("menace.roi_tracker", Path("roi_tracker.py"))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
sys.modules.setdefault("menace.roi_tracker", rt)


# Provide a minimal Flask stub to satisfy metrics_dashboard import
flask_stub = types.ModuleType("flask")
class _DummyFlask:
    def add_url_rule(self, *a, **k):
        pass
    def run(self, host="0.0.0.0", port=0):
        pass
flask_stub.Flask = lambda *a, **k: _DummyFlask()
flask_stub.jsonify = lambda *a, **k: {}
sys.modules.setdefault("flask", flask_stub)

# Minimal stub for MetricsDashboard import
dash_stub = types.ModuleType("menace.metrics_dashboard")

class _DummyDash:
    def __init__(self, *a, **k):
        pass

    def run(self, host="0.0.0.0", port=0):
        pass

dash_stub.MetricsDashboard = _DummyDash
sys.modules.setdefault("menace.metrics_dashboard", dash_stub)

import argparse
import sandbox_runner.cli as cli


class DummyTracker:
    def __init__(self):
        self.module_deltas = {}
        self.metrics_history = {}
        self.predicted_roi = []
        self.actual_roi = []
        self.predicted_metrics = {}
        self.actual_metrics = {}

    def update(self, delta, synergy):
        self.module_deltas.setdefault("mod", []).append(delta)
        self.metrics_history.setdefault("synergy_roi", []).append(synergy)

    def diminishing(self):
        return 0.01

    def rankings(self):
        total = sum(self.module_deltas.get("mod", []))
        return [("mod", total)]


def test_autonomous_synergy_loop(monkeypatch):
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"env": "A"}, {"env": "B"}])

    tracker = DummyTracker()
    calls = []
    values = iter([
        (0.05, 0.05),
        (0.05, 0.05),
        (0.005, 0.005),
        (0.005, 0.005),
    ])

    def fake_capture(preset, args):
        calls.append(preset)
        delta, syn = next(values)
        tracker.update(delta, syn)
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    args = argparse.Namespace(
        sandbox_data_dir=None,
        preset_count=2,
        max_iterations=10,
        dashboard_port=None,
        roi_cycles=2,
        synergy_cycles=2,
    )

    cli.full_autonomous_run(args)

    assert len(calls) == 4
    assert tracker.module_deltas["mod"] == [0.05, 0.05, 0.005, 0.005]
    assert tracker.metrics_history["synergy_roi"] == [0.05, 0.05, 0.005, 0.005]


def test_autonomous_synergy_converges(monkeypatch):
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"env": "A"}, {"env": "B"}])

    tracker = DummyTracker()
    calls = []
    values = iter([
        (0.1, 0.1),
        (0.1, 0.1),
        (0.02, 0.02),
        (0.02, 0.02),
        (0.005, 0.005),
        (0.005, 0.005),
    ])

    def fake_capture(preset, args):
        calls.append(preset)
        delta, syn = next(values)
        tracker.update(delta, syn)
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    log_calls = []

    def fake_info(msg, *a, **k):
        log_calls.append((msg, k.get("extra")))

    monkeypatch.setattr(cli.logger, "info", fake_info)

    args = argparse.Namespace(
        sandbox_data_dir=None,
        preset_count=2,
        max_iterations=10,
        dashboard_port=None,
        roi_cycles=2,
        synergy_cycles=2,
    )

    cli.full_autonomous_run(args)

    assert len(calls) == 6
    assert tracker.module_deltas["mod"] == [0.1, 0.1, 0.02, 0.02, 0.005, 0.005]
    assert tracker.metrics_history["synergy_roi"] == [0.1, 0.1, 0.02, 0.02, 0.005, 0.005]
    msg = ("synergy convergence reached", {"iteration": 3, "ema": pytest.approx(0.005)})
    assert msg in log_calls


def test_async_synergy_prediction(monkeypatch):
    """Ensure async synergy prediction stores pred_synergy values."""

    from plugins import synergy_predict as sp
    from menace.roi_tracker import ROITracker
    pm = _StubPM()
    import asyncio

    tracker = ROITracker()
    sp.register(pm, tracker)

    tracker.update(0.0, 0.1, metrics={"synergy_roi": 0.05})
    metrics = asyncio.run(sp.collect_metrics_async(0.0, 0.1, None))
    tracker.register_metrics(*metrics.keys())
    tracker.update(0.1, 0.2, metrics=metrics)

    assert "pred_synergy_roi" in tracker.metrics_history
