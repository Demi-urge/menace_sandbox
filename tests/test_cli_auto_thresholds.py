import sys
import types

# provide minimal stubs for optional heavy modules
menace_stub = types.ModuleType("menace")
metrics_stub = types.ModuleType("menace.metrics_dashboard")
metrics_stub.MetricsDashboard = lambda *a, **k: object()
menace_stub.metrics_dashboard = metrics_stub
sys.modules.setdefault("menace", menace_stub)
sys.modules.setdefault("menace.metrics_dashboard", metrics_stub)

import sandbox_runner.cli as cli
from scipy.stats import t

def test_adaptive_threshold():
    values = [1.0, 2.0, 3.0]
    thr = cli._adaptive_threshold(values, 3, factor=1.0)
    assert thr > 0


def test_adaptive_synergy_threshold():
    hist = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    thr = cli._adaptive_synergy_threshold(hist, 3, factor=1.0, weight=1.0)
    vals = [1.0, 2.0, 3.0]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    expected = t.ppf(0.975, len(vals) - 1) * std / len(vals) ** 0.5
    assert abs(thr - expected) < 1e-6


def test_synergy_threshold_weighting():
    hist = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}, {"a": 4.0}]
    thr1 = cli._adaptive_synergy_threshold(hist, 4, weight=1.0)
    thr2 = cli._adaptive_synergy_threshold(hist, 4, weight=0.5)
    assert thr2 < thr1
