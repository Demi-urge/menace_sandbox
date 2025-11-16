import sys
import types
import pytest

# minimal stubs for optional modules
menace_stub = types.ModuleType("menace")
metrics_stub = types.ModuleType("menace.metrics_dashboard")
metrics_stub.MetricsDashboard = lambda *a, **k: object()
menace_stub.metrics_dashboard = metrics_stub
sys.modules.setdefault("menace", menace_stub)
sys.modules.setdefault("menace.metrics_dashboard", metrics_stub)

import sandbox_runner.cli as cli


def test_adaptive_threshold_empty():
    assert cli._adaptive_threshold([], 3) == 0.0


def test_adaptive_threshold_single_value():
    assert cli._adaptive_threshold([5.0], 3) == 0.0


def test_adaptive_threshold_scaling_with_variance():
    low_var = [1.0, 1.1, 0.9, 1.0]
    high_var = [2.0, -2.0, 2.0, -2.0]
    thr_low = cli._adaptive_threshold(low_var, 4, factor=1.0)
    thr_high = cli._adaptive_threshold(high_var, 4, factor=1.0)
    assert thr_high > thr_low


def test_adaptive_synergy_threshold_empty():
    assert cli._adaptive_synergy_threshold([], 3) == 0.0


def test_adaptive_synergy_threshold_single_value():
    hist = [{"synergy_roi": 0.5}]
    assert cli._adaptive_synergy_threshold(hist, 1) == 0.0


def test_adaptive_synergy_threshold_scaling_with_variance():
    high_hist = [
        {"synergy_roi": 1.5},
        {"synergy_roi": -1.5},
        {"synergy_roi": 1.5},
        {"synergy_roi": -1.5},
    ]
    low_hist = [
        {"synergy_roi": 0.2},
        {"synergy_roi": -0.2},
        {"synergy_roi": 0.2},
        {"synergy_roi": -0.2},
    ]
    thr_high = cli._adaptive_synergy_threshold(high_hist, 4, factor=1.0)
    thr_low = cli._adaptive_synergy_threshold(low_hist, 4, factor=1.0)
    assert thr_high > thr_low
