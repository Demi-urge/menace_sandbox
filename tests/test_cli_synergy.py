import sandbox_runner.cli as cli

def test_synergy_converged():
    hist = [
        {"synergy_roi": 0.005},
        {"synergy_roi": 0.005},
        {"synergy_roi": 0.005},
    ]
    ok, ema, conf = cli._synergy_converged(hist, 3, 0.03)
    assert ok is True
    assert conf >= 0.95

def test_synergy_not_converged():
    hist = [
        {"synergy_roi": 0.05},
        {"synergy_roi": -0.04},
        {"synergy_roi": 0.04},
    ]
    ok, ema, conf = cli._synergy_converged(hist, 3, 0.03)
    assert ok is False
    assert conf < 0.95


def test_synergy_trending_not_converged():
    hist = [
        {"synergy_roi": 0.01},
        {"synergy_roi": 0.015},
        {"synergy_roi": 0.02},
    ]
    ok, ema, conf = cli._synergy_converged(hist, 3, 0.03)
    assert ok is False


def _patch_adfuller(monkeypatch, pvalue: float) -> None:
    import types, sys

    mod = types.ModuleType("statsmodels.tsa.stattools")

    def adfuller(vals, *a, **k):
        return (0.0, pvalue, 0, len(vals), {}, 0.0)

    mod.adfuller = adfuller
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", mod)
    tsa = types.ModuleType("tsa")
    tsa.stattools = mod
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa)
    root = types.ModuleType("statsmodels")
    root.tsa = tsa
    monkeypatch.setitem(sys.modules, "statsmodels", root)


def _patch_adfuller_dynamic(monkeypatch) -> None:
    import types, sys

    mod = types.ModuleType("statsmodels.tsa.stattools")

    def adfuller(vals, *a, **k):
        if abs(vals[-1] - vals[0]) > 0.01:
            p = 0.9
        else:
            p = 0.01
        return (0.0, p, 0, len(vals), {}, 0.0)

    mod.adfuller = adfuller
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", mod)
    tsa = types.ModuleType("tsa")
    tsa.stattools = mod
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa)
    root = types.ModuleType("statsmodels")
    root.tsa = tsa
    monkeypatch.setitem(sys.modules, "statsmodels", root)


def test_synergy_stationary_via_adf(monkeypatch):
    _patch_adfuller(monkeypatch, 0.01)
    hist = [
        {"synergy_roi": 0.0001},
        {"synergy_roi": 0.0001},
        {"synergy_roi": 0.0001},
    ]
    ok, ema, conf = cli._synergy_converged(
        hist, 3, 0.05, ma_window=2, stationarity_confidence=0.95
    )
    assert ok is True
    assert conf >= 0.95


def test_synergy_non_stationary_via_adf(monkeypatch):
    _patch_adfuller(monkeypatch, 0.9)
    hist = [
        {"synergy_roi": 0.01},
        {"synergy_roi": 0.02},
        {"synergy_roi": 0.03},
    ]
    ok, ema, conf = cli._synergy_converged(
        hist, 3, 0.05, ma_window=2, stationarity_confidence=0.95
    )
    assert ok is False
    assert conf < 0.95


def test_synergy_rolling_correlation_not_converged():
    hist = [
        {"synergy_roi": 0.0},
        {"synergy_roi": 0.02},
        {"synergy_roi": 0.0},
        {"synergy_roi": 0.02},
        {"synergy_roi": 0.0},
    ]
    ok, _, conf = cli._synergy_converged(hist, 5, 0.03, ma_window=2)
    assert ok is False
    assert conf < 0.95


def test_synergy_high_variance_not_converged():
    hist = [
        {"synergy_roi": 0.02},
        {"synergy_roi": -0.02},
        {"synergy_roi": 0.02},
        {"synergy_roi": -0.02},
    ]
    ok, _, conf = cli._synergy_converged(hist, 4, 0.03)
    assert ok is False
    assert conf < 0.95


def test_synergy_slow_convergence(monkeypatch):
    _patch_adfuller_dynamic(monkeypatch)
    hist = [
        {"synergy_roi": 0.08},
        {"synergy_roi": 0.05},
        {"synergy_roi": 0.03},
        {"synergy_roi": 0.02},
        {"synergy_roi": 0.015},
        {"synergy_roi": 0.012},
        {"synergy_roi": 0.011},
        {"synergy_roi": 0.0111},
        {"synergy_roi": 0.0109},
        {"synergy_roi": 0.011},
        {"synergy_roi": 0.0112},
        {"synergy_roi": 0.011},
    ]
    ok, _, conf = cli._synergy_converged(
        hist,
        6,
        0.02,
        ma_window=1,
        confidence=0.8,
        stationarity_confidence=0.95,
    )
    assert ok is True
    assert conf > 0.8
