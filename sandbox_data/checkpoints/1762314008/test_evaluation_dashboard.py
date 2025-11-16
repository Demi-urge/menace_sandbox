import sys
import importlib
import logging
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

import sys
from pathlib import Path
import dynamic_path_router

sys.modules["menace"].RAISE_ERRORS = False

dynamic_path_router.resolve_path = lambda p: Path(p).resolve()
dynamic_path_router.path_for_prompt = lambda p: Path(p).resolve().as_posix()

for p in [
    Path("workflow_roi_history.json"),
    Path("menace_roi_tracker_local.db"),
    Path("shared/global.db"),
]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()

import menace.evaluation_manager as em
import menace.evaluation_dashboard as ed
import menace.roi_tracker as rt
import menace.telemetry_backend as tb

rt.ROITracker.load_prediction_history = lambda self, path=None: None
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


def test_roi_prediction_panel_workflows():
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    tracker = rt.ROITracker()
    tracker.record_prediction(0.5, 0.6, workflow_id="wf1")
    tracker.update(0.0, 0.1, modules=["wf1"], confidence=0.1)
    panel = dash.roi_prediction_panel(tracker)
    wf = panel["workflows"]["wf1"]
    assert wf["mae"][-1] == pytest.approx(tracker.workflow_mae("wf1"))
    assert wf["variance"][-1] == pytest.approx(tracker.workflow_variance("wf1"))
    assert wf["confidence"][-1] == pytest.approx(tracker.workflow_confidence("wf1"))
    assert wf["needs_review"] is True


def test_roi_prediction_chart():
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    tracker = rt.ROITracker()
    tracker.record_prediction(0.5, 0.7)
    tracker.record_prediction(1.0, 0.9)
    chart = dash.roi_prediction_chart(tracker)
    assert chart["predicted"][-2:] == [pytest.approx(0.5), pytest.approx(1.0)]
    assert chart["actual"][-2:] == [pytest.approx(0.7), pytest.approx(0.9)]
    assert chart["labels"][-2:] == [len(chart["labels"]) - 2, len(chart["labels"]) - 1]
    chart_w = dash.roi_prediction_chart(tracker, window=1)
    assert chart_w["predicted"] == [pytest.approx(1.0)]
    assert chart_w["actual"] == [pytest.approx(0.9)]
    assert chart_w["labels"] == [chart["labels"][-1]]


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


def test_roi_prediction_events_panel_workflows():
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    tracker = rt.ROITracker()
    tracker.record_prediction(0.5, 0.6, workflow_id="wf1")
    tracker.update(0.0, 0.1, modules=["wf1"], confidence=0.1)
    panel = dash.roi_prediction_events_panel(tracker)
    wf = panel["workflows"]["wf1"]
    assert wf["mae"][-1] == pytest.approx(tracker.workflow_mae("wf1"))
    assert wf["variance"][-1] == pytest.approx(tracker.workflow_variance("wf1"))
    assert wf["confidence"][-1] == pytest.approx(tracker.workflow_confidence("wf1"))
    assert wf["needs_review"] is True


def test_alignment_warning_panel(monkeypatch):
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    monkeypatch.setattr(
        ed, "load_persisted_alignment_warnings", lambda **kw: [{"entry_id": "w"}]
    )
    warnings = dash.alignment_warning_panel()
    assert warnings and warnings[0]["entry_id"] == "w"


def test_readiness_panels_and_refresh(tmp_path):
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    tel = tb.TelemetryBackend(str(tmp_path / "tel.db"))
    tel.log_prediction("wf1", 1.0, 0.8, 0.9, {}, False, 0.6, ts="2021-01-01")
    tel.log_prediction("wf1", 0.5, 0.4, 0.8, {}, True, 0.3, ts="2021-01-02")
    chart = dash.readiness_chart(tel)
    assert chart["readiness"] == [0.6, 0.3]
    dist = dash.readiness_distribution_panel(tel)
    assert dist["readiness"] == [0.6, 0.3]
    assert dist["prediction_errors"] == [pytest.approx(0.2), pytest.approx(0.1)]
    tracker = rt.ROITracker()
    tracker.drift_flags.extend([False, True])
    tracker.metrics_history["instability"] = [0.1, 0.2]
    drift = dash.drift_instability_panel(tracker)
    assert drift["drift_flags"] == [False, True]
    out = tmp_path / "dash.json"
    hist = tmp_path / "roi_history.json"
    hist.write_text("[]")
    ed.refresh_dashboard(out, history=str(hist), telemetry_db=str(tmp_path / "tel.db"))
    data = json.loads(out.read_text())
    assert "readiness_over_time" in data


def test_governance_panel(tmp_path, monkeypatch):
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)
    log_path = tmp_path / "gov.log"
    monkeypatch.setattr(ed, "GOVERNANCE_LOG", log_path)
    ed.append_governance_result(
        {
            "decision": "ship",
            "alignment": {"status": "pass", "rationale": ""},
            "raroi_increase": 0,
        },
        [],
        {"confidence": 0.9},
        ["auto"],
    )
    ed.append_governance_result(
        {
            "decision": "rollback",
            "alignment": {"status": "fail", "rationale": ""},
            "raroi_increase": 1,
        },
        ["rule"],
        {"confidence": 0.1},
        ["rule"],
    )
    panel = dash.governance_panel()
    assert len(panel) == 2
    assert panel[-1]["vetoes"] == ["rule"]
    assert panel[-1]["forecast"] == {"confidence": 0.1}
    assert panel[-1]["reasons"] == ["rule"]


def test_relevancy_radar_panel_includes_impact_and_flag(tmp_path, monkeypatch):
    mgr = _make_manager()
    dash = ed.EvaluationDashboard(mgr)

    metrics_dir = tmp_path / "sandbox_data"
    metrics_dir.mkdir()
    metrics_file = metrics_dir / "relevancy_metrics.json"
    metrics_file.write_text(
        json.dumps({"alpha": {"imports": 10, "executions": 0, "impact": 2.5}})
    )

    monkeypatch.setattr(ed, "__file__", str(tmp_path / "evaluation_dashboard.py"))  # path-ignore
    monkeypatch.setattr(ed, "flagged_modules", lambda: {"alpha": "retire"})

    panel = dash.relevancy_radar_panel()
    assert panel and panel[0]["impact"] == pytest.approx(2.5)
    assert panel[0]["flag"] == "retire"


def test_dashboard_uses_fallbacks_when_dependencies_missing(monkeypatch, caplog, tmp_path):
    target = "menace_sandbox.evaluation_dashboard"
    for dependency in ("evaluation_manager", "roi_tracker", "telemetry_backend"):
        sentinel = types.ModuleType(f"menace_sandbox.{dependency}")
        monkeypatch.setitem(sys.modules, f"menace_sandbox.{dependency}", sentinel)

    sys.modules.pop(target, None)

    caplog.set_level(logging.WARNING)
    module = importlib.import_module(target)

    manager = module.EvaluationManager()
    assert manager.history == {}

    tracker = module.ROITracker()
    summary = tracker.prediction_summary()
    assert summary["workflow_confidence"] == {}
    assert summary["workflow_mae"] == {}
    assert summary["workflow_variance"] == {}

    telemetry = module.TelemetryBackend("ignored.db")
    assert telemetry.fetch_history() == []

    output_path = tmp_path / "dash.json"
    result = module.refresh_dashboard(output=output_path, telemetry_db="ignored.db")
    assert result == output_path
    assert output_path.exists()
    assert any("using fallback stub" in rec.message for rec in caplog.records)


def test_roi_panels_recover_when_tracker_helper_times_out(monkeypatch):
    module = importlib.reload(ed)
    module._get_roi_tracker_cls.cache_clear()

    def raising_loader():
        raise TimeoutError("import timed out")

    monkeypatch.setattr(module, "_get_roi_tracker_cls", raising_loader)
    tracker_instance = module._instantiate_dependency(
        "ROITracker", module._get_roi_tracker_cls, module._FallbackROITracker
    )
    assert isinstance(tracker_instance, module._FallbackROITracker)

    dash = module.EvaluationDashboard(module._FallbackEvaluationManager())
    panel = dash.roi_prediction_panel()
    assert isinstance(panel, dict)
    assert panel.get("workflow_confidence") == {}
    assert panel.get("workflows") == {}

    chart = dash.roi_prediction_chart()
    assert chart == {"labels": [], "predicted": [], "actual": []}

    events = dash.roi_prediction_events_panel()
    assert events["drift_flags"] == []
    assert events["workflows"] == {}

    instability = dash.drift_instability_panel()
    assert instability["drift_flags"] == []


def test_refresh_dashboard_uses_fallbacks_when_helpers_fail(monkeypatch, tmp_path):
    module = importlib.reload(ed)
    module._get_evaluation_manager_cls.cache_clear()
    module._get_roi_tracker_cls.cache_clear()
    module._get_telemetry_backend_cls.cache_clear()

    monkeypatch.setattr(
        module,
        "_get_evaluation_manager_cls",
        lambda: (_ for _ in ()).throw(TimeoutError("eval timeout")),
    )
    monkeypatch.setattr(
        module,
        "_get_roi_tracker_cls",
        lambda: (_ for _ in ()).throw(RuntimeError("roi failure")),
    )
    monkeypatch.setattr(
        module,
        "_get_telemetry_backend_cls",
        lambda: (_ for _ in ()).throw(ImportError("no telemetry")),
    )
    manager_instance = module._instantiate_dependency(
        "EvaluationManager", module._get_evaluation_manager_cls, module._FallbackEvaluationManager
    )
    tracker_instance = module._instantiate_dependency(
        "ROITracker", module._get_roi_tracker_cls, module._FallbackROITracker
    )
    telemetry_instance = module._instantiate_dependency(
        "TelemetryBackend", module._get_telemetry_backend_cls, module._FallbackTelemetryBackend
    )
    assert isinstance(manager_instance, module._FallbackEvaluationManager)
    assert isinstance(tracker_instance, module._FallbackROITracker)
    assert isinstance(telemetry_instance, module._FallbackTelemetryBackend)

    output = tmp_path / "dash.json"
    result = module.refresh_dashboard(output=output)
    assert result == output
    data = json.loads(output.read_text())
    assert set(data.keys()) == {
        "readiness_over_time",
        "readiness_distribution",
        "drift_instability",
    }

