import importlib
import sys
import types
import pytest


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg, _, sub = name.partition('.')
    pkg_mod = sys.modules.get(pkg)
    if pkg_mod and sub:
        setattr(pkg_mod, sub, mod)
    return mod


def _add_optional_deps(monkeypatch, adf_p=0.01, lev_p=0.8):
    stats = types.ModuleType("statsmodels.tsa.stattools")
    stats.adfuller = lambda vals, *a, **k: (0.0, adf_p, 0, len(vals), {}, 0.0)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", stats)
    tsa = types.ModuleType("statsmodels.tsa")
    tsa.stattools = stats
    root = types.ModuleType("statsmodels")
    root.tsa = tsa
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa)
    monkeypatch.setitem(sys.modules, "statsmodels", root)

    sstats = types.ModuleType("scipy.stats")
    sstats.levene = lambda *a, **k: types.SimpleNamespace(pvalue=lev_p)
    sstats.pearsonr = lambda *a, **k: (0.0, 0.0)
    sstats.t = types.SimpleNamespace(cdf=lambda *a, **k: 0.5)
    sroot = types.ModuleType("scipy")
    sroot.stats = sstats
    monkeypatch.setitem(sys.modules, "scipy.stats", sstats)
    monkeypatch.setitem(sys.modules, "scipy", sroot)


def _remove_optional_deps(monkeypatch):
    for m in ["statsmodels.tsa.stattools", "statsmodels.tsa", "statsmodels"]:
        if m in sys.modules:
            monkeypatch.delitem(sys.modules, m, raising=False)
    # provide minimal scipy.stats so import in cli succeeds
    sstats = types.ModuleType("scipy.stats")
    sstats.pearsonr = lambda *a, **k: (0.0, 0.0)
    sstats.t = types.SimpleNamespace(cdf=lambda *a, **k: 0.5)
    sroot = types.ModuleType("scipy")
    sroot.stats = sstats
    monkeypatch.setitem(sys.modules, "scipy.stats", sstats)
    monkeypatch.setitem(sys.modules, "scipy", sroot)


def _import_cli(monkeypatch, with_deps: bool):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(monkeypatch, "menace.metrics_dashboard", MetricsDashboard=lambda *a, **k: None)
    _remove_optional_deps(monkeypatch)
    if with_deps:
        _add_optional_deps(monkeypatch)
    sys.modules.pop("sandbox_runner.cli", None)
    return importlib.import_module("sandbox_runner.cli")


@pytest.fixture(params=[False, True], ids=["no_deps", "with_deps"])
def cli_mod(monkeypatch, request):
    """Return sandbox_runner.cli with or without optional deps."""
    return _import_cli(monkeypatch, with_deps=request.param)


def test_adaptive_threshold_variance(monkeypatch):
    cli = _import_cli(monkeypatch, with_deps=False)
    low = [1.0, 1.1, 0.9, 1.0]
    high = [2.0, -2.0, 2.0, -2.0]
    t_low = cli._adaptive_threshold(low, 4, factor=1.0)
    t_high = cli._adaptive_threshold(high, 4, factor=1.0)
    assert t_high > t_low


def test_adaptive_synergy_threshold_predictions(monkeypatch):
    cli = _import_cli(monkeypatch, with_deps=False)
    hist = [{"synergy_roi": 0.2}, {"synergy_roi": 0.1}, {"synergy_roi": 0.3}]
    preds = [{"synergy_roi": 0.2}, {"synergy_roi": 0.1}, {"synergy_roi": 0.3}]
    thr = cli._adaptive_synergy_threshold(hist, 3, predictions=preds)
    assert thr == 0.0


def test_synergy_converged_constant(cli_mod):
    hist = [{"synergy_roi": 0.001}] * 5
    ok, _, _ = cli_mod._synergy_converged(hist, 5, 0.01)
    assert ok is True


def test_synergy_converged_increasing(cli_mod):
    hist_inc = [{"synergy_roi": 0.001 * i} for i in range(5)]
    ok, _, _ = cli_mod._synergy_converged(hist_inc, 5, 0.01)
    assert ok is False


def test_slow_convergence(cli_mod):
    hist = [
        {"synergy_roi": 0.1},
        {"synergy_roi": 0.08},
        {"synergy_roi": 0.06},
        {"synergy_roi": 0.05},
        {"synergy_roi": 0.04},
        {"synergy_roi": 0.035},
        {"synergy_roi": 0.033},
        {"synergy_roi": 0.031},
        {"synergy_roi": 0.029},
        {"synergy_roi": 0.03},
        {"synergy_roi": 0.029},
        {"synergy_roi": 0.03},
    ]
    ok, _, conf = cli_mod._synergy_converged(
        hist, 6, 0.03, ma_window=1, confidence=0.8
    )
    assert ok is True
    assert conf > 0.8


def test_synergy_converged_insufficient_history(cli_mod):
    hist = [{"synergy_roi": 0.01}, {"synergy_roi": 0.02}]
    ok, _, _ = cli_mod._synergy_converged(hist, 3, 0.05)
    assert ok is False


def test_synergy_converged_prediction_metric(cli_mod):
    hist = [
        {"synergy_roi": 0.001, "pred_synergy_roi": 0.05},
        {"synergy_roi": 0.001, "pred_synergy_roi": 0.05},
        {"synergy_roi": 0.001, "pred_synergy_roi": 0.05},
    ]
    ok, _, _ = cli_mod._synergy_converged(hist, 3, 0.01)
    assert ok is False
