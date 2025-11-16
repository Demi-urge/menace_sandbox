import importlib
import types
import sys
import argparse
from pathlib import Path

from tests.test_run_autonomous_env_vars import _load_module


def test_forecast_failure_counters_increment(monkeypatch, tmp_path):
    # stub metrics exporter with simple gauges before importing module
    class Gauge:
        def __init__(self):
            self.value = 0.0
        def set(self, v):
            self.value = v
        def inc(self, amount=1.0):
            self.value += amount
        def labels(self, *a, **k):
            return self
        def get(self):
            return self.value

    me = types.ModuleType("metrics_exporter")
    me.roi_forecast_gauge = Gauge()
    me.synergy_forecast_gauge = Gauge()
    me.roi_threshold_gauge = Gauge()
    me.synergy_threshold_gauge = Gauge()
    me.roi_forecast_failures_total = Gauge()
    me.synergy_forecast_failures_total = Gauge()
    me.start_metrics_server = lambda *a, **k: None
    me.synergy_adaptation_actions_total = Gauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", me)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", me)

    stats_mod = types.ModuleType("scipy.stats")
    stats_mod.t = types.SimpleNamespace(cdf=lambda *a, **k: 0.0)
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(stats=stats_mod))
    monkeypatch.setitem(sys.modules, "scipy.stats", stats_mod)

    mod = _load_module(monkeypatch)
    cli_mod = mod.sandbox_runner.cli
    cli_mod._diminishing_modules = lambda *a, **k: (set(), None)
    cli_mod._adaptive_threshold = lambda *a, **k: 0.0
    cli_mod._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_mod.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, 1.0)
    cli_mod._ema = lambda seq: (0.0, 0.0)

    class BadTracker:
        module_deltas = {}
        metrics_history = {}
        roi_history = []
        def forecast(self):
            raise RuntimeError("boom")
        def predict_synergy(self):
            raise RuntimeError("boom")
        def diminishing(self):
            return 0.0

    args = argparse.Namespace(
        auto_thresholds=False,
        roi_cycles=1,
        synergy_cycles=1,
        synergy_threshold=None,
        roi_threshold=None,
        roi_confidence=None,
        synergy_threshold_window=1,
        synergy_threshold_weight=1.0,
        synergy_confidence=None,
        save_synergy_history=False,
    )

    module_history = {}
    flagged = set()
    synergy_history = []
    synergy_ma_history = []
    roi_ma_history = []
    thresh_log = types.SimpleNamespace(log=lambda *a, **k: None)

    mod.update_metrics(
        BadTracker(),
        args,
        0,
        module_history,
        flagged,
        synergy_history,
        synergy_ma_history,
        roi_ma_history,
        None,
        None,
        None,
        1,
        1.0,
        None,
        thresh_log,
    )

    assert me.roi_forecast_failures_total.value == 1.0
    assert me.synergy_forecast_failures_total.value == 1.0
