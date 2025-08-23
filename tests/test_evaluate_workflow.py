import json

import menace.deployment_governance as dg
from menace.override_validator import generate_signature


def test_basic_evaluation(monkeypatch):
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }
    res = dg.evaluate_workflow(scorecard, {})
    assert res["verdict"] == "promote"
    assert res["reason_codes"] == []


def test_honours_override_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)

    key = tmp_path / "key"
    key.write_text("secret")

    data = {"verdict": "promote"}
    sig = generate_signature(data, str(key))
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"data": data, "signature": sig}))

    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 0.2,
        "confidence": 0.8,
    }

    res = dg.evaluate_workflow(
        scorecard,
        {"override_path": str(override), "public_key_path": str(key)},
    )

    assert res["verdict"] == "promote"
    assert "manual_override" in res["reason_codes"]
    assert res["overrides"]["override_path"] == str(override)


def _reset(monkeypatch):
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)


def test_no_go_when_raroi_low(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 0.2,
        "confidence": 0.9,
    }
    res = dg.evaluate_workflow(scorecard, {})
    assert res["verdict"] == "no_go"
    assert "raroi_below_threshold" in res["reason_codes"]


def test_no_go_when_confidence_low(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.2,
    }
    res = dg.evaluate_workflow(scorecard, {})
    assert res["verdict"] == "no_go"
    assert "confidence_below_threshold" in res["reason_codes"]


def test_no_go_when_scenario_low(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.2},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }
    res = dg.evaluate_workflow(scorecard, {})
    assert res["verdict"] == "no_go"
    assert "scenario_below_min" in res["reason_codes"]


def test_micro_pilot_trigger(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.5,
        "confidence": 0.9,
        "sandbox_roi": 0.05,
        "adapter_roi": 1.2,
    }
    res = dg.evaluate_workflow(scorecard, {})
    assert res["verdict"] == "pilot"
    assert "micro_pilot" in res["reason_codes"]
    assert res["overrides"].get("mode") == "micro-pilot"


def test_override_path_nested_bypasses_micro_pilot(tmp_path, monkeypatch):
    _reset(monkeypatch)
    key = tmp_path / "key"
    key.write_text("secret")
    data = {"bypass_micro_pilot": True}
    sig = generate_signature(data, str(key))
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"data": data, "signature": sig}))

    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.5,
        "confidence": 0.9,
        "sandbox_roi": 0.05,
        "adapter_roi": 1.2,
    }

    policy = {
        "overrides": {"override_path": str(override), "public_key_path": str(key)}
    }
    res = dg.evaluate_workflow(scorecard, policy)
    assert res["verdict"] == "promote"
    assert "micro_pilot" not in res["reason_codes"]
    assert res["overrides"]["override_path"] == str(override)
    assert res["overrides"]["bypass_micro_pilot"] is True


class _DummyTracker:
    def predict_roi_collapse(self, _wf_id):
        return {"risk": "Stable", "brittle": False}


class _DummyGraph:
    def simulate_impact_wave(self, *args, **kwargs):
        return {}


def test_foresight_gate_pass(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }

    def fake_gate(
        workflow_id,
        patch,
        forecaster,
        workflow_graph,
        *,
        roi_threshold=dg.DeploymentGovernor.raroi_threshold,
        confidence_threshold=0.6,
    ):
        return (
            True,
            {"upgrade_id": "fid1", "projections": [], "confidence": None, "recommendation": "promote"},
            [],
        )

    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: _DummyGraph())
    monkeypatch.setattr(dg.audit_logger, "log_event", lambda *a, **k: None)

    res = dg.evaluate_workflow(
        scorecard,
        {},
        foresight_tracker=_DummyTracker(),
        workflow_id="wf1",
        patch=[],
    )
    assert res["verdict"] == "promote"
    assert res.get("foresight", {}).get("reason_codes") == []


def test_foresight_gate_failure(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }

    def fake_gate(
        workflow_id,
        patch,
        forecaster,
        workflow_graph,
        *,
        roi_threshold=dg.DeploymentGovernor.raroi_threshold,
        confidence_threshold=0.6,
    ):
        return (
            False,
            {"upgrade_id": "fid2", "projections": [], "confidence": None, "recommendation": "pilot"},
            ["low_confidence"],
        )

    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: _DummyGraph())
    monkeypatch.setattr(dg.audit_logger, "log_event", lambda *a, **k: None)

    res = dg.evaluate_workflow(
        scorecard,
        {},
        foresight_tracker=_DummyTracker(),
        workflow_id="wf1",
        patch=[],
    )
    assert res["verdict"] == "pilot"
    assert "low_confidence" in res["reason_codes"]
    assert res.get("foresight", {}).get("reason_codes") == ["low_confidence"]


def test_foresight_gate_failure_borderline_bucket(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.02,
        "confidence": 0.72,
    }

    def fake_gate(
        workflow_id,
        patch,
        forecaster,
        workflow_graph,
        *,
        roi_threshold=dg.DeploymentGovernor.raroi_threshold,
        confidence_threshold=0.6,
    ):
        return (
            False,
            {"upgrade_id": "fid3", "projections": [], "confidence": None, "recommendation": "borderline"},
            ["borderline"],
        )

    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: _DummyGraph())
    monkeypatch.setattr(dg.audit_logger, "log_event", lambda *a, **k: None)

    class Bucket:
        def __init__(self):
            self.called = False

        def enqueue(self, workflow_id, raroi, confidence, context=None):
            self.called = True
            self.args = (workflow_id, raroi, confidence, context)

    bucket = Bucket()
    res = dg.evaluate_workflow(
        scorecard,
        {},
        foresight_tracker=_DummyTracker(),
        workflow_id="wf1",
        patch=[],
        borderline_bucket=bucket,
    )
    assert res["verdict"] == "borderline"
    assert "borderline" in res["reason_codes"]
    assert res.get("foresight", {}).get("reason_codes") == ["borderline"]
    assert bucket.called


def test_policy_thresholds_passed(monkeypatch):
    _reset(monkeypatch)
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }

    called = {}

    def fake_gate(
        workflow_id,
        patch,
        forecaster,
        workflow_graph,
        *,
        roi_threshold=dg.DeploymentGovernor.raroi_threshold,
        confidence_threshold=0.6,
    ):
        called["roi_threshold"] = roi_threshold
        return (
            True,
            {"upgrade_id": "fid4", "projections": [], "confidence": None, "recommendation": "promote"},
            [],
        )

    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: _DummyGraph())

    policy = {"roi_forecast_min": 0.5}

    dg.evaluate_workflow(
        scorecard,
        policy,
        foresight_tracker=_DummyTracker(),
        workflow_id="wf1",
        patch=[],
    )

    assert called["roi_threshold"] == 0.5
