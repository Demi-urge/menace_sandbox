import sys
import types
import pytest

# provide minimal stubs for optional heavy modules
menace_stub = types.ModuleType("menace")
metrics_stub = types.ModuleType("menace.metrics_dashboard")
metrics_stub.MetricsDashboard = lambda *a, **k: object()
menace_stub.metrics_dashboard = metrics_stub
sys.modules.setdefault("menace", menace_stub)
sys.modules.setdefault("menace.metrics_dashboard", metrics_stub)

import sandbox_runner.cli as cli

def test_adaptive_threshold():
    values = [1.0, 2.0, 3.0]
    thr = cli._adaptive_threshold(values, 3, factor=1.0)
    assert thr > 0


def test_adaptive_synergy_threshold():
    hist = [
        {"synergy_a": 1.0},
        {"synergy_a": 2.0},
        {"synergy_a": 3.0},
    ]
    preds = [
        {"synergy_a": 0.9},
        {"synergy_a": 2.1},
        {"synergy_a": 2.9},
    ]
    thr = cli._adaptive_synergy_threshold(
        hist, 3, factor=1.0, weight=1.0, predictions=preds
    )
    diffs = [1.0 - 0.9, 2.0 - 2.1, 3.0 - 2.9]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    expected = (var ** 0.5) * 1.0
    assert abs(thr - expected) < 1e-6


def test_synergy_threshold_weighting():
    hist = [
        {"synergy_a": 1.0},
        {"synergy_a": 2.0},
        {"synergy_a": 3.0},
        {"synergy_a": 4.0},
    ]
    preds = [
        {"synergy_a": 0.9},
        {"synergy_a": 1.9},
        {"synergy_a": 3.1},
        {"synergy_a": 4.1},
    ]
    thr1 = cli._adaptive_synergy_threshold(hist, 4, weight=1.0, predictions=preds)
    thr2 = cli._adaptive_synergy_threshold(hist, 4, weight=0.5, predictions=preds)
    assert thr2 < thr1


def test_synergy_threshold_weighted_expected():
    hist = [
        {"synergy_a": 1.0},
        {"synergy_a": 2.0},
        {"synergy_a": 3.0},
        {"synergy_a": 4.0},
    ]
    preds = [
        {"synergy_a": 0.0},
        {"synergy_a": 0.0},
        {"synergy_a": 2.0},
        {"synergy_a": 4.0},
    ]
    thr = cli._adaptive_synergy_threshold(hist, 4, weight=0.5, factor=1.0, predictions=preds)

    diffs = [1.0, 2.0, 1.0, 0.0]
    w = [0.5 ** i for i in range(len(diffs) - 1, -1, -1)]
    ema = sum(d * w_i for d, w_i in zip(diffs, w)) / sum(w)
    var = sum(w_i * (d - ema) ** 2 for d, w_i in zip(diffs, w)) / sum(w)
    expected = (var ** 0.5) * 1.0

    assert thr == pytest.approx(expected)
