import builtins
import sys
import types
import menace.roi_tracker as rt


def _setup_failing_arima(monkeypatch):
    class FailingARIMA:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")

    mod = types.ModuleType("statsmodels.tsa.arima.model")
    mod.ARIMA = FailingARIMA
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima.model", mod)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima", types.ModuleType("arima"))
    sys.modules["statsmodels.tsa.arima"].model = mod
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", types.ModuleType("tsa"))
    sys.modules["statsmodels.tsa"].arima = sys.modules["statsmodels.tsa.arima"]
    sys.modules.setdefault("statsmodels", types.ModuleType("statsmodels")).tsa = sys.modules["statsmodels.tsa"]


def test_arima_forecast(monkeypatch):
    calls = []

    class DummyARIMA:
        def __init__(self, data, order, exog=None):
            self.data = list(data)
            calls.append(order)

        def fit(self):
            return self

        def get_forecast(self, steps=1, exog=None):
            class Res:
                predicted_mean = [float(len(self.data) + 1)]

                @staticmethod
                def conf_int(alpha=0.05):
                    return [[0.0, 1.0]]

            return Res()

    mod = types.ModuleType("statsmodels.tsa.arima.model")
    mod.ARIMA = DummyARIMA
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima.model", mod)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima", types.ModuleType("arima"))
    sys.modules["statsmodels.tsa.arima"].model = mod
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", types.ModuleType("tsa"))
    sys.modules["statsmodels.tsa"].arima = sys.modules["statsmodels.tsa.arima"]
    sys.modules.setdefault("statsmodels", types.ModuleType("statsmodels")).tsa = sys.modules["statsmodels.tsa"]

    tracker = rt.ROITracker()
    for i in range(1, 5):
        tracker.update(0.0, float(i))
    pred, _ = tracker.forecast()
    assert pred == 5.0
    assert calls


def test_regression_forecast(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("statsmodels"):
            raise ImportError("no statsmodels")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tracker = rt.ROITracker()
    for i in range(1, 5):
        tracker.update(0.0, float(i))
    pred, _ = tracker.forecast()
    assert round(pred, 1) == 5.0


def _setup_arima(monkeypatch, calls):
    class DummyARIMA:
        def __init__(self, data, order, exog=None):
            self.data = list(data)
            calls.append(order)

        def fit(self):
            return self

        def get_forecast(self, steps=1, exog=None):
            class Res:
                predicted_mean = [float(len(self.data) + 1)]

                @staticmethod
                def conf_int(alpha=0.05):
                    return [[0.0, 1.0]]

            return Res()

    mod = types.ModuleType("statsmodels.tsa.arima.model")
    mod.ARIMA = DummyARIMA
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima.model", mod)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.arima", types.ModuleType("arima"))
    sys.modules["statsmodels.tsa.arima"].model = mod
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", types.ModuleType("tsa"))
    sys.modules["statsmodels.tsa"].arima = sys.modules["statsmodels.tsa.arima"]
    sys.modules.setdefault("statsmodels", types.ModuleType("statsmodels")).tsa = sys.modules["statsmodels.tsa"]


def test_arima_synergy_forecast(monkeypatch):
    calls = []
    _setup_arima(monkeypatch, calls)

    tracker = rt.ROITracker()
    for i in range(1, 5):
        roi = float(i)
        tracker.update(0.0, roi, metrics={"m": roi, "synergy_roi": roi, "synergy_m": roi})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    assert tracker.predict_synergy() == 5.0
    assert tracker.predict_synergy_metric("m") == 5.0
    assert calls


def test_regression_synergy_forecast(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("statsmodels"):
            raise ImportError("no statsmodels")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tracker = rt.ROITracker()
    for i in range(1, 5):
        roi = float(i)
        tracker.update(0.0, roi, metrics={"m": roi, "synergy_roi": roi, "synergy_m": roi})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    assert round(tracker.predict_synergy(), 1) == 5.0
    assert round(tracker.predict_synergy_metric("m"), 1) == 5.0


def test_arima_failure_logs(monkeypatch, caplog):
    _setup_failing_arima(monkeypatch)
    tracker = rt.ROITracker()
    for i in range(1, 5):
        tracker.update(0.0, float(i))
    caplog.set_level("ERROR")
    pred, _ = tracker.forecast()
    assert round(pred, 1) == 5.0
    assert "ARIMA forecast failed" in caplog.text


def test_synergy_arima_failure_logs(monkeypatch, caplog):
    _setup_failing_arima(monkeypatch)
    tracker = rt.ROITracker()
    for i in range(1, 5):
        val = float(i)
        tracker.update(0.0, val, metrics={"m": val, "synergy_roi": val, "synergy_m": val})
    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))
    caplog.set_level("ERROR")
    result = tracker.predict_synergy_metric("m")
    assert round(result, 1) == 5.0
    assert "ARIMA synergy forecast failed" in caplog.text
