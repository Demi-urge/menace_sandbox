import json
import pytest
import numpy as np
import logging
import json
import menace_sandbox.roi_tracker as rt
import menace_sandbox.self_test_service as sts
import types
import sys
import pytest
from menace_sandbox.telemetry_backend import TelemetryBackend
from menace_sandbox.readiness_index import compute_readiness


def test_roi_tracker_basic():
    tracker = rt.ROITracker(window=3, tolerance=0.01)
    vertex, preds, stop, _ = tracker.update(0.0, 0.1)
    assert vertex is None
    assert not stop

    data = [0.2, 0.3, 0.2, 0.1, -0.05]
    for d in data:
        vertex, preds, stop, _ = tracker.update(0.0, d)
    assert vertex is not None
    assert stop
    assert len(preds) == len(tracker.roi_history)


def test_new_synergy_metrics_present():
    tracker = rt.ROITracker()
    expected = {
        "synergy_shannon_entropy",
        "synergy_flexibility",
        "synergy_energy_consumption",
    }
    for name in expected:
        assert name in tracker.metrics_history
        assert name in tracker.synergy_metrics_history


def test_register_metrics_pads_synergy_series():
    tracker = rt.ROITracker()
    tracker.update(0.0, 0.0, metrics={})
    tracker.register_metrics("profitability")
    assert tracker.synergy_metrics_history["synergy_profitability"] == [0.0]


def test_register_scenario_metrics():
    tracker = rt.ROITracker()
    tracker.register_metrics(
        "latency_error_rate",
        "hostile_failures",
        "misuse_failures",
        "concurrency_throughput",
    )
    for name in (
        "synergy_latency_error_rate",
        "synergy_hostile_failures",
        "synergy_misuse_failures",
        "synergy_concurrency_throughput",
    ):
        assert name in tracker.synergy_metrics_history


def test_roi_forecast_linear_sequence():
    tracker = rt.ROITracker()
    for i in range(1, 6):
        tracker.update(0.0, float(i))
    pred, interval = tracker.forecast()
    assert round(pred, 1) == 6.0
    assert interval[0] <= pred <= interval[1]


def test_forecast_metric():
    tracker = rt.ROITracker()
    for i in range(5):
        tracker.update(0.0, 0.0, metrics={"loss": float(i)})
    pred, interval = tracker.forecast_metric("loss")
    assert round(pred, 1) == 5.0
    assert interval[0] <= pred <= interval[1]


def test_save_and_load_json(tmp_path):
    tracker = rt.ROITracker()
    for i in range(5):
        tracker.update(0.0, float(i), [f"m{i}.py"], metrics={"metric": float(i)})  # path-ignore
    tracker.record_prediction(0.5, 0.6)
    tracker.record_prediction(0.7, 0.8)
    tracker.record_metric_prediction("metric", 1.0, 0.5)
    tracker.synergy_history.append({"synergy_roi": 0.5})
    tracker.record_scenario_delta(
        "normal", 1.0, {"metric": 0.0}, {"synergy_metric": 0.0}, 1.0, 0.0
    )
    tracker.record_scenario_delta(
        "concurrency_spike",
        -1.0,
        {"metric": -1.0},
        {"synergy_metric": -1.0},
        0.0,
        -1.0,
    )
    path = tmp_path / "hist.json"
    tracker.save_history(str(path))

    new_tracker = rt.ROITracker()
    new_tracker.load_history(str(path))
    assert new_tracker.roi_history == tracker.roi_history
    assert new_tracker.module_deltas == tracker.module_deltas
    assert new_tracker.module_entropy_deltas == tracker.module_entropy_deltas
    assert new_tracker.predicted_roi == tracker.predicted_roi
    assert new_tracker.actual_roi == tracker.actual_roi
    assert new_tracker.metrics_history == tracker.metrics_history
    assert new_tracker.predicted_metrics.get("metric") == tracker.predicted_metrics.get("metric")
    assert new_tracker.actual_metrics.get("metric") == tracker.actual_metrics.get("metric")
    assert new_tracker.synergy_history == tracker.synergy_history
    assert new_tracker.scenario_roi_deltas == tracker.scenario_roi_deltas
    assert new_tracker.scenario_raroi == tracker.scenario_raroi
    assert new_tracker.scenario_raroi_delta == tracker.scenario_raroi_delta
    assert new_tracker.scenario_metrics_delta == tracker.scenario_metrics_delta
    assert new_tracker.scenario_synergy_delta == tracker.scenario_synergy_delta
    assert new_tracker.worst_scenario() == tracker.worst_scenario()


def test_save_and_load_sqlite(tmp_path):
    tracker = rt.ROITracker()
    for i in range(3):
        tracker.update(0.0, float(i), [f"x{i}.py"], metrics={"m": float(i)})  # path-ignore
    tracker.record_prediction(1.0, 0.9)
    tracker.synergy_history.append({"synergy_roi": 0.2})
    tracker.record_scenario_delta(
        "normal", 0.5, {"m": 0.0}, {"synergy_m": 0.0}, 0.5, 0.0
    )
    tracker.record_scenario_delta(
        "concurrency_spike",
        -1.0,
        {"m": -1.0},
        {"synergy_m": -1.0},
        -0.5,
        -1.0,
    )
    tracker.metrics_history.setdefault("synergy_roi", [0.0] * len(tracker.roi_history))
    path = tmp_path / "hist.db"
    tracker.save_history(str(path))

    other = rt.ROITracker()
    other.load_history(str(path))
    assert other.roi_history == tracker.roi_history
    assert other.module_deltas == tracker.module_deltas
    assert other.module_entropy_deltas == tracker.module_entropy_deltas
    assert other.predicted_roi == tracker.predicted_roi
    assert other.actual_roi == tracker.actual_roi
    assert other.metrics_history == tracker.metrics_history
    assert other.synergy_history == tracker.synergy_history
    assert other.scenario_roi_deltas == tracker.scenario_roi_deltas
    assert other.scenario_raroi == tracker.scenario_raroi
    assert other.scenario_raroi_delta == tracker.scenario_raroi_delta
    assert other.scenario_metrics_delta == tracker.scenario_metrics_delta
    assert other.scenario_synergy_delta == tracker.scenario_synergy_delta
    assert other.worst_scenario() == tracker.worst_scenario()


def test_entropy_plateau_detection():
    tracker = rt.ROITracker()
    tracker.module_entropy_deltas = {
        "a.py": [0.005, 0.004, 0.003],  # path-ignore
        "b.py": [0.02, 0.01, 0.02],  # path-ignore
    }
    assert tracker.entropy_plateau(0.01, 3) == ["a.py"]  # path-ignore


def test_roi_tracker_filters_outliers():
    tracker = rt.ROITracker()
    normal = [0.1, 0.2, 0.15, 0.18]
    for v in normal:
        tracker.update(0.0, v)
    baseline = len(tracker.roi_history)
    tracker.update(0.0, 5.0)
    tracker.update(0.0, -3.0)
    assert len(tracker.roi_history) == baseline


def test_roi_tracker_filtering_disabled():
    tracker = rt.ROITracker(filter_outliers=False)
    values = [0.1, 0.2, 0.15, 0.18, 5.0]
    for v in values:
        tracker.update(0.0, v)
    assert tracker.roi_history[-1] == 5.0


def test_plot_history_creates_file(tmp_path):
    tracker = rt.ROITracker()
    for i in range(5):
        tracker.update(0.0, float(i))
    out = tmp_path / "plot.png"
    tracker.plot_history(str(out))
    assert out.exists()


def test_cli_plot(tmp_path):
    tracker = rt.ROITracker()
    for i in range(3):
        tracker.update(0.0, float(i))
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))

    out = tmp_path / "out.png"
    rt.cli(["plot", str(hist), str(out)])
    assert out.exists()


def test_cli_forecast(tmp_path, capsys):
    tracker = rt.ROITracker()
    for i in range(1, 6):
        tracker.update(0.0, float(i))
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))

    rt.cli(["forecast", str(hist)])
    out = capsys.readouterr().out
    assert "Predicted ROI" in out
    assert "6.000" in out


def test_cli_rank(tmp_path, capsys):
    tracker = rt.ROITracker()
    tracker.update(0.0, 0.5, ["a.py", "b.py"])  # path-ignore
    tracker.update(0.5, 1.0, ["b.py"])  # path-ignore
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))

    rt.cli(["rank", str(hist)])
    out = capsys.readouterr().out.strip().splitlines()
    assert any(line.startswith("a.py") for line in out)  # path-ignore
    assert any(line.startswith("b.py") for line in out)  # path-ignore
    assert all("(roi" in line for line in out)


def test_module_deltas_tracked():
    tracker = rt.ROITracker()
    tracker.update(0.0, 0.5, ["a.py", "b.py"])  # path-ignore
    tracker.update(0.5, 1.0, ["b.py"])  # path-ignore
    assert tracker.module_deltas["a.py"] == [0.5]  # path-ignore
    assert tracker.module_deltas["b.py"] == [0.5, 0.5]  # path-ignore


def test_update_without_modules():
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0)
    assert tracker.module_deltas == {}


def test_entropy_delta_history(tmp_path, monkeypatch):
    from menace.code_database import PatchHistoryDB, PatchRecord

    db_path = tmp_path / "p.db"
    monkeypatch.setenv("PATCH_HISTORY_DB_PATH", str(db_path))
    patch_db = PatchHistoryDB(str(db_path))
    rec = PatchRecord(
        filename="a.py",  # path-ignore
        description="",
        roi_before=0.0,
        roi_after=0.0,
        complexity_delta=2.0,
    )
    patch_db.add(rec)
    tracker = rt.ROITracker()
    tracker.metrics_history["synergy_shannon_entropy"] = [0.1, 0.3]
    tracker.record_prediction(0.0, 0.0)
    assert rt.ROITracker.entropy_delta_history(tracker, "a.py") == [pytest.approx(0.1)]  # path-ignore


def test_arima_order_selection_cached(monkeypatch):
    import sys, types

    calls = []

    class DummyARIMA:
        def __init__(self, data, order, exog=None):
            self.data = data
            self.order = order
            self.exog = exog
            calls.append(order)

        def fit(self):
            self.aic = sum(self.order)
            self.bic = self.aic + 0.5
            return self

        def get_forecast(self):
            class _Res:
                predicted_mean = [float(len(self.data) + 1)]

                @staticmethod
                def conf_int(alpha=0.05):
                    return [[0.0, 1.0]]

            return _Res()

    mod = types.ModuleType("statsmodels.tsa.arima.model")
    mod.ARIMA = DummyARIMA
    sys.modules["statsmodels.tsa.arima.model"] = mod
    sys.modules.setdefault("statsmodels.tsa.arima", types.ModuleType("arima")).model = (
        mod
    )
    sys.modules.setdefault("statsmodels.tsa", types.ModuleType("tsa")).arima = (
        sys.modules["statsmodels.tsa.arima"]
    )
    sys.modules.setdefault("statsmodels", types.ModuleType("statsmodels")).tsa = (
        sys.modules["statsmodels.tsa"]
    )

    tracker = rt.ROITracker()
    for i in range(4):
        tracker.update(0.0, float(i + 1))

    pred1, _ = tracker.forecast()
    assert tracker._best_order is not None
    pred2, _ = tracker.forecast()
    assert tracker._best_order is not None
    assert len(calls) == 5


def test_weighted_averaging_triggers_stop():
    tracker = rt.ROITracker(window=3, tolerance=0.05, weights=[0.1, 0.2, 0.7])
    tracker.update(0.0, 0.1)
    tracker.update(0.0, 0.1)
    _, _, stop, _ = tracker.update(0.0, -0.04)
    assert stop


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        rt.ROITracker(window=3, weights=[1.0, 2.0])


def test_update_with_resource_metrics():
    tracker = rt.ROITracker()
    tracker.update(
        0.0,
        1.0,
        resources={"cpu": 1.0, "memory": 2.0, "disk": 3.0, "time": 4.0, "gpu": 5.0},
    )
    assert tracker.resource_metrics and tracker.resource_metrics[0][-1] == 5.0


def test_forecast_with_resource_data(monkeypatch):
    class DummyDB:
        def history(self):
            import pandas as pd

            return pd.DataFrame(
                {
                    "cpu": [1, 2, 3, 4, 5],
                    "memory": [2, 3, 4, 5, 6],
                    "disk": [1, 1, 1, 1, 1],
                    "time": [1, 1, 1, 1, 1],
                    "gpu": [0, 0, 1, 1, 2],
                }
            )

    tracker = rt.ROITracker(resource_db=DummyDB())
    for i in range(1, 6):
        tracker.update(0.0, float(i))
    pred, _ = tracker.forecast()
    assert isinstance(pred, float)


def test_prediction_reliability(tmp_path, capsys):
    tracker = rt.ROITracker()
    tracker.record_prediction(1.0, 1.5)
    tracker.record_prediction(2.0, 1.0)
    mae = tracker.rolling_mae()
    assert mae == pytest.approx(0.75)
    assert tracker.rolling_mae(window=1) == pytest.approx(1.0)
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))
    rt.cli(["reliability", str(hist)])
    out = capsys.readouterr().out
    assert "ROI MAE" in out and "0.75" in out


def test_metric_prediction_reliability(tmp_path):
    tracker = rt.ROITracker()
    tracker.record_metric_prediction("profit", 1.0, 1.5)
    tracker.record_metric_prediction("profit", 2.0, 1.0)
    mae = tracker.rolling_mae_metric("profit")
    assert mae == pytest.approx(0.75)
    assert tracker.rolling_mae_metric("profit", window=1) == pytest.approx(1.0)
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))
    other = rt.ROITracker()
    other.load_history(str(hist))
    assert other.predicted_metrics == tracker.predicted_metrics
    assert other.actual_metrics == tracker.actual_metrics


def test_metric_prediction_pairs_persist(tmp_path):
    tracker = rt.ROITracker()
    tracker.record_metric_prediction("profit", 1.0, 1.5)
    tracker.record_metric_prediction("profit", 2.0, 1.0)
    path = tmp_path / "hist.json"
    tracker.save_history(str(path))
    data = json.loads(path.read_text())
    assert data["metric_predictions"]["profit"] == [[1.0, 1.5], [2.0, 1.0]]
    other = rt.ROITracker()
    other.load_history(str(path))
    assert other.predicted_metrics == tracker.predicted_metrics
    assert other.actual_metrics == tracker.actual_metrics


def test_cli_metric_reliability(tmp_path, capsys):
    tracker = rt.ROITracker()
    tracker.record_metric_prediction("profit", 1.0, 0.5)
    tracker.record_metric_prediction("profit", 2.0, 2.5)
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))
    rt.cli(["reliability", str(hist), "--metric", "profit"])
    out = capsys.readouterr().out
    assert "profit MAE" in out


def test_cli_synergy_metric_reliability(tmp_path, capsys):
    tracker = rt.ROITracker()
    tracker.record_metric_prediction("synergy_projected_lucrativity", 1.0, 0.5)
    tracker.record_metric_prediction("synergy_projected_lucrativity", 2.0, 2.5)
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))
    rt.cli(["reliability", str(hist), "--metric", "synergy_projected_lucrativity"])
    out = capsys.readouterr().out
    assert "synergy_projected_lucrativity MAE" in out


def test_cv_reliability_roi():
    tracker = rt.ROITracker()
    for i in range(5):
        tracker.record_prediction(float(i), float(i) + 0.1)
    score = tracker.cv_reliability()
    assert 0.9 <= score <= 1.0


def test_cv_reliability_metric():
    tracker = rt.ROITracker()
    for i in range(5):
        tracker.record_metric_prediction("profit", float(i), float(i) + 0.1)
    score = tracker.cv_reliability(metric="profit")
    assert 0.9 <= score <= 1.0


class _DummyPredBot:
    def __init__(self):
        self.calls = []

    def predict_metric(self, name, feats):
        self.calls.append((name, feats))
        return 3.0


class _StubManager:
    def __init__(self, bot):
        self.registry = {"b": type("E", (), {"bot": bot})()}

    def get_prediction_bots_for(self, _name):
        return ["b"]

    def assign_prediction_bots(self, _):
        return ["b"]


def test_predict_metric_with_manager_records():
    bot = _DummyPredBot()
    mgr = _StubManager(bot)
    tracker = rt.ROITracker()
    val = tracker.predict_metric_with_manager(mgr, "profit", [1.0], actual=2.0)
    assert val == 3.0
    assert bot.calls[0][0] == "profit"
    assert tracker.predicted_metrics["profit"] == [3.0]
    assert tracker.actual_metrics["profit"] == [2.0]


def test_predict_all_metrics_records():
    bot = _DummyPredBot()
    mgr = _StubManager(bot)
    tracker = rt.ROITracker()
    tracker.update(0.0, 0.0, metrics={"a": 1.0, "b": 2.0})
    tracker.predict_all_metrics(mgr, [0.5])
    assert tracker.predicted_metrics["a"] == [3.0]
    assert tracker.actual_metrics["a"] == [1.0]
    assert tracker.predicted_metrics["b"] == [3.0]
    assert tracker.actual_metrics["b"] == [2.0]
    assert len(bot.calls) == 2


def test_predict_all_metrics_without_history():
    bot = _DummyPredBot()
    mgr = _StubManager(bot)
    tracker = rt.ROITracker()
    tracker.register_metrics("x")
    result = tracker.predict_all_metrics(mgr, [0.1])
    assert result == {"x": 3.0}
    assert tracker.predicted_metrics["x"] == []
    assert tracker.actual_metrics["x"] == []


def test_cli_predict_metric(tmp_path, capsys, monkeypatch):
    bot = _DummyPredBot()
    mgr = _StubManager(bot)
    mod = types.SimpleNamespace(PredictionManager=lambda: mgr)
    monkeypatch.setitem(sys.modules, "menace.prediction_manager_bot", mod)
    tracker = rt.ROITracker()
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))
    rt.cli(["predict-metric", str(hist), "profit", "--actual", "1.0"])
    out = capsys.readouterr().out
    assert "Predicted profit" in out


def test_cli_synergy_predict_metric(tmp_path, capsys, monkeypatch):
    tracker = rt.ROITracker()
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))
    monkeypatch.setattr(
        rt.ROITracker, "predict_synergy_profitability", lambda self: 4.2
    )
    rt.cli(["predict-metric", str(hist), "synergy_profitability"])
    out = capsys.readouterr().out
    assert "Predicted synergy_profitability" in out


def test_cli_synergy_revenue_metric(tmp_path, capsys, monkeypatch):
    tracker = rt.ROITracker()
    hist = tmp_path / "hist.json"
    tracker.save_history(str(hist))
    monkeypatch.setattr(rt.ROITracker, "predict_synergy_revenue", lambda self: 3.3)
    rt.cli(["predict-metric", str(hist), "synergy_revenue"])
    out = capsys.readouterr().out
    assert "Predicted synergy_revenue" in out


def test_rolling_mae_metric_window():
    tracker = rt.ROITracker()
    vals = [(0.0, 5.0), (0.0, 4.0), (0.0, 3.0), (0.0, 2.0), (0.0, 1.0)]
    for p, a in vals:
        tracker.record_metric_prediction("loss", p, a)
    assert tracker.rolling_mae_metric("loss") == pytest.approx(3.0)
    assert tracker.rolling_mae_metric("loss", window=2) == pytest.approx(1.5)


def test_reliability_roi_scores():
    tracker = rt.ROITracker()
    tracker.record_prediction(1.0, 1.5)
    tracker.record_prediction(2.0, 1.0)
    err = 1.0 / (1.0 + rt.ROITracker._ema([0.5, 1.0]))
    assert tracker.reliability() == pytest.approx(err)


def test_reliability_metric_scores():
    tracker = rt.ROITracker()
    tracker.record_metric_prediction("profit", 1.0, 2.0)
    tracker.record_metric_prediction("profit", 2.0, 1.0)
    err = 1.0 / (1.0 + rt.ROITracker._ema([1.0, 1.0]))
    assert tracker.reliability(metric="profit") == pytest.approx(err)


def test_prediction_metrics_exported():
    tracker = rt.ROITracker()
    tracker.record_prediction(1.0, 1.5)
    from menace_sandbox import metrics_exporter as me

    def _get_value(gauge):
        try:
            return gauge._value.get()
        except Exception:
            try:
                wrappers = getattr(gauge, "_values", None)
                if wrappers:
                    return float(sum(w.get() for w in wrappers.values()))
                wrappers = getattr(gauge, "_metrics", None)
                if wrappers:
                    return float(
                        sum(getattr(w, "_value", 0).get() for w in wrappers.values())
                    )
            except Exception:
                return 0.0
        return 0.0

    assert _get_value(me.prediction_error) == pytest.approx(0.5)
    assert _get_value(me.prediction_mae) == pytest.approx(0.5)


def test_synergy_reliability():
    tracker = rt.ROITracker()
    tracker.record_metric_prediction("synergy_roi", 1.0, 1.5)
    tracker.record_metric_prediction("synergy_roi", 2.0, 1.0)
    assert tracker.synergy_reliability() == pytest.approx(0.75)
    assert tracker.synergy_reliability(window=1) == pytest.approx(1.0)


def test_reliability_cross_validation():
    tracker = rt.ROITracker()
    for i in range(4):
        tracker.record_prediction(float(i), float(i))
    score = tracker.reliability(cv=3)
    assert 0.9 <= score <= 1.0


class _NamedManager(_StubManager):
    def __init__(self, bot):
        super().__init__(bot)
        self.names = []

    def get_prediction_bots_for(self, name):
        self.names.append(name)
        return super().get_prediction_bots_for(name)


def test_predict_all_metrics_features_and_name():
    bot = _DummyPredBot()
    mgr = _NamedManager(bot)
    tracker = rt.ROITracker()
    tracker.update(0.0, 0.0, metrics={"x": 1.0})
    tracker.predict_all_metrics(mgr, [0.2], bot_name="spec")
    assert mgr.names == ["spec"]
    assert bot.calls[0] == ("x", [0.2])


def test_predict_synergy_metric_uses_manager():
    bot = _DummyPredBot()
    mgr = _StubManager(bot)
    tracker = rt.ROITracker()
    val = tracker.predict_synergy_metric("profitability", manager=mgr)
    assert val == 3.0
    assert bot.calls == [("synergy_profitability", [])]


def test_forecast_uses_advanced_predictor(monkeypatch):
    monkeypatch.setenv("ENABLE_ADVANCED_ROI_PREDICTOR", "1")
    import menace_sandbox.roi_predictor as rp

    calls = []

    def fake_forecast(self, history, exog=None):
        calls.append(True)
        return 9.0, (8.0, 10.0)

    monkeypatch.setattr(rp.ROIPredictor, "forecast", fake_forecast)
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0)
    tracker.update(0.0, 2.0)
    pred, interval = tracker.forecast()
    assert pred == 9.0
    assert interval == (8.0, 10.0)
    assert calls


def test_tracker_init_has_recovery_metric():
    tracker = rt.ROITracker()
    assert "recovery_time" in tracker.metrics_history
    assert tracker.metrics_history["recovery_time"] == []


def test_recovery_time_saved(tmp_path):
    tracker = rt.ROITracker()
    tracker.update(1.0, 0.5, metrics={"recovery_time": 0.0})
    tracker.update(0.5, 1.0, metrics={"recovery_time": 2.5})
    path = tmp_path / "hist.json"
    tracker.save_history(str(path))

    other = rt.ROITracker()
    other.load_history(str(path))
    assert other.metrics_history["recovery_time"][-1] == 2.5

def test_synergy_history_multi_metric(tmp_path):
    tracker = rt.ROITracker()
    tracker.synergy_history.append({"synergy_roi": 0.1, "synergy_efficiency": 0.2})
    tracker.synergy_history.append({"synergy_roi": 0.2, "synergy_efficiency": 0.3})
    path = tmp_path / "hist.json"
    tracker.save_history(str(path))

    other = rt.ROITracker()
    other.load_history(str(path))
    assert other.synergy_history == tracker.synergy_history
    path2 = tmp_path / "hist.db"
    tracker.save_history(str(path2))
    another = rt.ROITracker()
    another.load_history(str(path2))
    assert another.synergy_history == tracker.synergy_history


def test_raroi_high_risk():
    tracker = rt.ROITracker()
    tracker.roi_history = [0.1, 0.1, 0.1]
    base, raroi, _ = tracker.calculate_raroi(
        1.0,
        workflow_type="critical",
        rollback_prob=0.8,
    )
    expected = 1.0 * (1 - 0.8 * 0.9) * (1 - np.std([0.1, 0.1, 0.1]))
    assert base == pytest.approx(1.0)
    assert raroi == pytest.approx(expected)


def test_raroi_unstable_roi():
    tracker = rt.ROITracker()
    deltas = [0.5, -0.5, 0.5]
    tracker.roi_history = deltas
    base, raroi, _ = tracker.calculate_raroi(
        1.0,
        workflow_type="standard",
        rollback_prob=0.0,
    )
    expected = 1.0 * (1 - 0.0 * 0.5) * (1 - np.std(deltas))
    assert raroi == pytest.approx(expected)
    assert raroi < base


def test_raroi_failing_tests():
    tracker = rt.ROITracker()
    tracker.roi_history = [0.1, 0.1, 0.1]
    sts.set_failed_critical_tests(["security"])
    base, raroi, _ = tracker.calculate_raroi(
        1.0,
        workflow_type="standard",
        rollback_prob=0.0,
    )
    penalty = rt.CRITICAL_TEST_PENALTIES.get("security", 1.0)
    expected = (
        1.0
        * (1 - 0.0 * 0.5)
        * (1 - np.std([0.1, 0.1, 0.1]))
        * penalty
    )
    assert raroi == pytest.approx(expected)


def test_workflow_metrics():
    tracker = rt.ROITracker(workflow_window=3)
    tracker.record_prediction(1.0, 1.2, workflow_id="wf1")
    tracker.record_prediction(1.0, 0.8, workflow_id="wf1")
    tracker.record_prediction(1.0, 1.0, workflow_id="wf1")
    assert tracker.workflow_predicted_roi["wf1"] == [1.0, 1.0, 1.0]
    assert tracker.workflow_actual_roi["wf1"] == [1.2, 0.8, 1.0]
    errors = [0.2, 0.2, 0.0]
    assert tracker.workflow_mae("wf1") == pytest.approx(np.mean(errors))
    assert tracker.workflow_variance("wf1") == pytest.approx(np.var([1.2, 0.8, 1.0]))
    tracker.record_prediction(1.0, 2.0, workflow_id="wf1")
    assert tracker.workflow_predicted_roi["wf1"] == [1.0, 1.0, 1.0]
    assert tracker.workflow_actual_roi["wf1"] == [0.8, 1.0, 2.0]
    assert tracker.workflow_mae("wf1") == pytest.approx(np.mean([0.2, 0.0, 1.0]))
    assert tracker.workflow_variance("wf1") == pytest.approx(np.var([0.8, 1.0, 2.0]))


def test_workflow_confidence_formula():
    tracker = rt.ROITracker()
    tracker.record_prediction(0.0, 0.0, workflow_id="wf")
    tracker.record_prediction(1.0, 2.0, workflow_id="wf")
    assert tracker.workflow_mae("wf") == pytest.approx(0.5)
    assert tracker.workflow_variance("wf") == pytest.approx(1.0)
    expected = 1.0 / (1.0 + 0.5 + 1.0)
    assert tracker.workflow_confidence("wf") == pytest.approx(expected)


def test_update_logs_metrics(caplog, monkeypatch):
    tracker = rt.ROITracker()
    monkeypatch.setattr(tracker, "calculate_raroi", lambda roi, **kw: (roi, roi, []))
    with caplog.at_level(logging.INFO):
        tracker.update(0.0, 0.5, confidence=0.4)
    rec = next(r for r in caplog.records if r.msg == "roi update")
    assert rec.confidence == pytest.approx(0.4)
    assert rec.mae == pytest.approx(0.0)
    assert rec.roi_variance == pytest.approx(0.0)
    assert rec.final_score == pytest.approx(rec.adjusted)


def test_generate_scorecards():
    tracker = rt.ROITracker()
    tracker.record_scenario_delta("normal", 1.0, {}, {}, 1.0, 1.0)
    tracker.record_scenario_delta(
        "concurrency_spike", -1.0, {}, {}, 0.0, -1.0
    )
    cards = tracker.generate_scorecards()
    assert tracker.workflow_label == "situationally weak"
    cs = {c.scenario: c for c in cards}
    assert cs["concurrency_spike"].raroi_delta == pytest.approx(-1.0)
    assert cs["concurrency_spike"].recommendation == "add locking or queueing"
    assert cs["concurrency_spike"].status == "situationally weak"
    assert cs["normal"].recommendation is None


def test_generate_scenario_scorecard(tmp_path):
    tracker = rt.ROITracker()
    tracker.record_scenario_delta(
        "concurrency_spike", -1.0, {"errors": 1.0}, {"cpu": 2.0}, 0.0, -1.0
    )
    tracker.record_scenario_delta(
        "schema_drift", 0.5, {"schema_mismatches": 2.0}, {"cpu": 1.0}, 0.0, 0.5
    )
    card = tracker.generate_scenario_scorecard(
        "wf1", ["concurrency_spike", "schema_drift"]
    )
    assert card["workflow_id"] == "wf1"
    assert set(card["scenarios"]) == {"concurrency_spike", "schema_drift"}
    assert card["scenarios"]["concurrency_spike"]["roi_delta"] == pytest.approx(-1.0)
    assert card["scenarios"]["concurrency_spike"]["score"] == pytest.approx(-1.0)
    assert (
        card["scenarios"]["schema_drift"]["metrics_delta"]["schema_mismatches"]
        == pytest.approx(2.0)
    )
    assert card["status"] == "situationally weak"
    assert (
        card["scenarios"]["concurrency_spike"]["recommendation"]
        == "add locking or queueing"
    )
    assert (
        card["scenarios"]["schema_drift"]["recommendation"]
        == "tighten schema validation"
    )


def test_hardening_recommendations():
    tips = {
        "concurrency_spike": "add locking or queueing",
        "schema_drift": "tighten schema validation",
        "hostile_input": "sanitize inputs",
        "flaky_upstream": "add retries or fallback logic",
    }
    for scen, tip in tips.items():
        tracker = rt.ROITracker()
        tracker.record_scenario_delta(scen, -0.5, {}, {}, 0.0, -0.5)
        cards = tracker.generate_scorecards()
        cs = {c.scenario: c for c in cards}
        assert cs[scen].recommendation == tip


def test_situationally_weak_from_metrics():
    tracker = rt.ROITracker()
    tracker.record_scenario_delta("normal", 0.5, {}, {}, 0.0, 0.0)
    tracker.record_scenario_delta(
        "hostile_input", 0.2, {"error_rate": 1.0}, {}, 0.0, 0.0
    )
    tracker.record_scenario_delta("schema_drift", 0.3, {}, {}, 0.0, 0.0)
    tracker.generate_scorecards()
    assert tracker.workflow_label == "situationally weak"


def test_build_governance_scorecard():
    tracker = rt.ROITracker()
    tracker.record_scenario_delta("concurrency_spike", -1.0, {}, {}, 0.0, -1.0)
    base, raroi, _ = tracker.calculate_raroi(0.5)
    tracker.score_workflow("wf1", raroi)
    tracker.record_roi_prediction([0.4], [0.3], workflow_id="wf1")
    card, metrics = tracker.build_governance_scorecard("wf1", ["concurrency_spike"])
    assert card["raroi"] == pytest.approx(raroi)
    assert metrics["confidence"] == pytest.approx(tracker.last_confidence)
    assert metrics["predicted_roi"] == pytest.approx(0.4)
    assert metrics["scenario_scores"]["concurrency_spike"] == pytest.approx(
        card["scenarios"]["concurrency_spike"]["score"]
    )


def test_scorecard_cli(tmp_path, capsys):
    tracker = rt.ROITracker()
    tracker.record_scenario_delta("concurrency_spike", -1.0, {}, {}, 0.0, -1.0)
    history = tmp_path / "history.json"
    tracker.save_history(history.as_posix())

    out = tmp_path / "score.json"
    from menace_sandbox import adaptive_roi_cli

    adaptive_roi_cli.main(
        [
            "scorecard",
            "wf1",
            "--scenarios",
            "concurrency_spike",
            "--history",
            history.as_posix(),
            "--output",
            out.as_posix(),
        ]
    )
    data = json.loads(out.read_text())
    assert data["workflow_id"] == "wf1"
    assert "concurrency_spike" in data["scenarios"]


def test_prediction_drift_adjusts_confidence_and_logs_readiness(tmp_path):
    backend = TelemetryBackend(str(tmp_path / "tel.db"))
    tracker = rt.ROITracker(
        evaluation_window=3, mae_threshold=0.1, telemetry_backend=backend
    )
    tracker.calculate_raroi(0.5)
    tracker.score_workflow("wf1", 0.5)
    for _ in range(3):
        tracker.record_prediction([0.0], [1.0], workflow_id="wf1")
    assert tracker.last_confidence == pytest.approx(0.9)
    readiness = compute_readiness(
        tracker.last_raroi or 0.0,
        tracker.reliability(),
        1.0,
        1.0,
    )
    hist = backend.fetch_history()
    assert hist[-1]["confidence"] == pytest.approx(tracker.last_confidence)
    assert hist[-1]["readiness"] == pytest.approx(readiness)

