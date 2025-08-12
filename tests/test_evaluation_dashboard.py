import sys
import types
import json
import pytest

# Stub heavy dependencies and engine modules before importing EvaluationManager
sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")


class DummyEngineMod:
    pass


engine_mod.Engine = DummyEngineMod
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))

le_mod = types.ModuleType("learning_engine")
le_mod.LearningEngine = object
ue_mod = types.ModuleType("unified_learning_engine")
ue_mod.UnifiedLearningEngine = object
ae_mod = types.ModuleType("action_learning_engine")
ae_mod.ActionLearningEngine = object
sys.modules.setdefault("menace.learning_engine", le_mod)
sys.modules.setdefault("menace.unified_learning_engine", ue_mod)
sys.modules.setdefault("menace.action_learning_engine", ae_mod)

import menace.evaluation_manager as em
import menace.evaluation_dashboard as ed
import menace.roi_tracker as rt

# Clear stubs for optional libs after import
for mod in [
    "networkx",
    "pulp",
    "pandas",
    "prometheus_client",
]:
    sys.modules.pop(mod, None)


class DummyEngine:
    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(self):
        return {"cv_score": self.score}

    def persist_evaluation(self, res):
        pass


def _make_manager() -> em.EvaluationManager:
    e1 = DummyEngine(0.2)
    e2 = DummyEngine(0.8)
    mgr = em.EvaluationManager(learning_engine=e1, unified_engine=e2)
    mgr.history["learning_engine"] = [
        {"cv_score": 0.2},
        {"cv_score": 0.4},
    ]
    mgr.history["unified_engine"] = [
        {"cv_score": 0.8},
        {"cv_score": 0.6},
    ]
    return mgr


def test_dataframe_and_weights(monkeypatch):
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)

    class DummyDF(list):
        def __init__(self, records):
            super().__init__(records)
            self.columns = list(records[0].keys()) if records else []

    dummy_pd = types.SimpleNamespace(DataFrame=DummyDF)
    monkeypatch.setattr(ed, "pd", dummy_pd)

    df = dash.dataframe()
    assert isinstance(df, DummyDF)
    assert set(df.columns) == {"cv_score", "engine"}

    weights = dash.deployment_weights()
    assert weights["unified_engine"] == 1.0
    expected = (0.2 + 0.4) / 2 / ((0.8 + 0.6) / 2)
    assert abs(weights["learning_engine"] - expected) < 1e-6


def test_to_json_roundtrip(tmp_path):
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    path = tmp_path / "hist.json"
    dash.to_json(path)
    data = json.loads(path.read_text())
    assert data["learning_engine"][0]["cv_score"] == 0.2
    assert data["unified_engine"][1]["cv_score"] == 0.6


def test_roi_prediction_panel():
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    tracker = rt.ROITracker()
    tracker.record_prediction(0.5, 0.7)
    tracker.record_prediction(1.0, 0.9)
    tracker.record_class_prediction("up", "up")
    tracker.record_class_prediction("down", "up")
    panel = dash.roi_prediction_panel(tracker)
    assert panel["mae"] == pytest.approx(0.15)
    assert panel["accuracy"] == pytest.approx(0.5)
    assert panel["class_counts"]["predicted"]["up"] == 1
    assert panel["confusion_matrix"]["up"]["down"] == 1
    assert panel["mae_trend"][0] == pytest.approx(0.2)
    assert panel["mae_trend"][1] == pytest.approx(0.15)
    assert panel["accuracy_trend"][0] == pytest.approx(1.0)
    assert panel["accuracy_trend"][1] == pytest.approx(0.5)


def test_roi_prediction_chart():
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    tracker = rt.ROITracker()
    tracker.record_prediction(0.5, 0.7)
    tracker.record_prediction(1.0, 0.9)
    chart = dash.roi_prediction_chart(tracker)
    assert chart["predicted"] == [pytest.approx(0.5), pytest.approx(1.0)]
    assert chart["actual"] == [pytest.approx(0.7), pytest.approx(0.9)]
    assert chart["labels"] == [0, 1]
    chart_w = dash.roi_prediction_chart(tracker, window=1)
    assert chart_w["predicted"] == [pytest.approx(1.0)]
    assert chart_w["actual"] == [pytest.approx(0.9)]
    assert chart_w["labels"] == [1]


def test_roi_prediction_events_panel():
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    tracker = rt.ROITracker()
    tracker.horizon_mae_history.append({1: 0.1, 2: 0.2})
    tracker.record_class_prediction("up", "up")
    tracker.record_class_prediction("down", "up")
    tracker.drift_flags.extend([False, True])
    tracker._adaptive_predictor = types.SimpleNamespace(
        drift_metrics={"accuracy": 0.8, "mae": 0.05}
    )
    panel = dash.roi_prediction_events_panel(tracker)
    assert panel["mae_by_horizon"][1] == pytest.approx(0.1)
    assert panel["growth_class_accuracy"] == pytest.approx(0.5)
    assert panel["drift_flags"] == [False, True]
    assert panel["growth_type_accuracy"] == pytest.approx(0.8)
    assert panel["drift_metrics"]["mae"] == pytest.approx(0.05)
    panel_w = dash.roi_prediction_events_panel(tracker, window=1)
    assert panel_w["growth_class_accuracy"] == pytest.approx(0.0)
    assert panel_w["drift_flags"] == [True]

