from menace.deployment_governance import evaluate
import menace.deployment_governance as dg
from menace.upgrade_forecaster import ForecastResult


class DummyTracker:
    def predict_roi_collapse(self, workflow_id):
        return {"risk": "Stable", "brittle": False}


class DummyGraph:
    pass


class DummyLogger:
    def log(self, data):
        pass

    def close(self):
        pass


class DummyBucket:
    def add_candidate(self, *a, **k):
        pass

    def process(self, *a, **k):
        pass


def _patch_gate(monkeypatch, ok, reasons):
    forecast = ForecastResult([], 0.9, "u0")

    def fake_gate(
        workflow_id,
        patch,
        tracker,
        workflow_graph,
        roi_threshold=0.0,
        confidence_threshold=0.6,
        allow_negative_dag=False,
    ):
        return ok, list(reasons), forecast

    monkeypatch.setattr(dg, "WorkflowGraph", DummyGraph)
    monkeypatch.setattr(dg, "ForecastLogger", lambda *a, **k: DummyLogger())
    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg.audit_logger, "log_event", lambda *a, **k: None)
    return DummyTracker(), forecast


def test_deployment_evaluate_promote(monkeypatch):
    tracker, _ = _patch_gate(monkeypatch, True, [])
    res = evaluate(
        {"alignment": "pass", "security": "pass"},
        {"raroi": 1.0, "confidence": 0.9},
        patch=[],
        foresight_tracker=tracker,
        workflow_id="wf",
    )
    assert res["verdict"] == "promote"
    assert "meets_promotion_criteria" in res["reasons"]


def test_deployment_evaluate_borderline(monkeypatch):
    tracker, _ = _patch_gate(monkeypatch, False, ["low_confidence"])
    bucket = DummyBucket()
    res = evaluate(
        {"alignment": "pass", "security": "pass"},
        {"raroi": 1.0, "confidence": 0.9},
        patch=[],
        foresight_tracker=tracker,
        workflow_id="wf",
        borderline_bucket=bucket,
    )
    assert res["verdict"] == "borderline"
    assert "low_confidence" in res["reasons"]
    assert res["foresight"]["reason_codes"] == ["low_confidence"]


def test_deployment_evaluate_pilot(monkeypatch):
    tracker, _ = _patch_gate(monkeypatch, False, ["negative_impact_wave"])
    res = evaluate(
        {"alignment": "pass", "security": "pass"},
        {"raroi": 1.0, "confidence": 0.9},
        patch=[],
        foresight_tracker=tracker,
        workflow_id="wf",
    )
    assert res["verdict"] == "pilot"
    assert "negative_impact_wave" in res["reasons"]
