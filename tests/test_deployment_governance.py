import json
import yaml
import pytest

from menace.override_validator import generate_signature
import menace.deployment_governance as dg
from menace.foresight_tracker import ForesightTracker
from menace.workflow_graph import WorkflowGraph
from menace.upgrade_forecaster import CycleProjection, ForecastResult


@pytest.fixture(autouse=True)
def reset_rules(monkeypatch):
    """Ensure rule cache is cleared before each test."""
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)
    monkeypatch.setattr(dg, "_POLICY_CACHE", None)


@pytest.fixture
def promotion_case():
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }
    return scorecard, {}


@pytest.fixture
def demotion_case(tmp_path):
    rules = [
        {
            "decision": "demote",
            "condition": "confidence < 0.5",
            "reason_code": "policy_demote",
        }
    ]
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules))
    dg._load_rules(str(path))
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.4,
        "sandbox_roi": 1.0,
        "adapter_roi": 1.0,
    }
    return scorecard, {}


@pytest.fixture
def micro_pilot_case():
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.5,
        "confidence": 0.9,
        "sandbox_roi": 0.05,
        "adapter_roi": 1.2,
    }
    return scorecard, {}


@pytest.fixture
def veto_case():
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "fail",
        "raroi": 1.2,
        "confidence": 0.8,
    }
    return scorecard, {}


@pytest.fixture
def manual_override_case(tmp_path):
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
    policy = {"override_path": str(override), "public_key_path": str(key)}
    return scorecard, policy


def test_promotion(promotion_case):
    scorecard, policy = promotion_case
    res = dg.evaluate_workflow(scorecard, policy)
    assert res["verdict"] == "promote"
    assert res["reason_codes"] == []


def test_demotion(demotion_case):
    scorecard, policy = demotion_case
    res = dg.evaluate_workflow(scorecard, policy)
    assert res["verdict"] == "demote"
    assert "policy_demote" in res["reason_codes"]


def test_micro_pilot(micro_pilot_case):
    scorecard, policy = micro_pilot_case
    res = dg.evaluate_workflow(scorecard, policy)
    assert res["verdict"] == "pilot"
    assert res["overrides"].get("mode") == "micro-pilot"


def test_veto(veto_case):
    scorecard, policy = veto_case
    res = dg.evaluate_workflow(scorecard, policy)
    assert res["verdict"] == "demote"
    assert "alignment_veto" in res["reason_codes"]


def test_manual_override(manual_override_case):
    scorecard, policy = manual_override_case
    res = dg.evaluate_workflow(scorecard, policy)
    assert res["verdict"] == "promote"
    assert "manual_override" in res["reason_codes"]
    assert res["overrides"]["override_path"] == policy["override_path"]


def test_invalid_rule_config(tmp_path):
    bad = tmp_path / "rules.yaml"
    bad.write_text(yaml.safe_dump({"decision": "promote"}))  # not a list
    with pytest.raises(ValueError):
        dg._load_rules(str(bad))


def test_invalid_policy_config(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "_POLICY_CACHE", None)
    bad = tmp_path / "policy.yaml"
    bad.write_text("[]")
    with pytest.raises(ValueError):
        dg._load_policy(str(bad))


def test_unsafe_expression_rejected():
    with pytest.raises(ValueError):
        dg._safe_eval("__import__('os').system('echo hi')", {})


def test_unsafe_rule_ignored(tmp_path):
    rules = [
        {
            "decision": "demote",
            "condition": "__import__('os').system('echo hi')",
            "reason_code": "bad",
        }
    ]
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules))
    dg._load_rules(str(path))
    scorecard = {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }
    res = dg.evaluate_workflow(scorecard, {})
    assert res["verdict"] != "demote"
    assert "bad" not in res["reason_codes"]


# ---------------------------------------------------------------------------
# Foresight promotion gate tests
# ---------------------------------------------------------------------------


class DummyTracker(ForesightTracker):
    def __init__(self, collapse=None):
        self.collapse = collapse or {"risk": "Stable"}

    def predict_roi_collapse(self, workflow_id):
        return self.collapse


@pytest.fixture
def foresight_tracker():
    return DummyTracker()


@pytest.fixture
def workflow_graph(tmp_path, monkeypatch):
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    graph.add_workflow("wf", roi=0.0)
    graph.add_workflow("dep", roi=0.0)
    graph.add_dependency("wf", "dep", impact_weight=1.0)
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: graph)
    return graph


@pytest.fixture
def dummy_patch():
    return ["patch"]


@pytest.fixture
def decision_logs(monkeypatch):
    logs = []

    class DummyLogger:
        def log(self, data):
            logs.append(data)

    monkeypatch.setattr(dg, "_decision_logger", DummyLogger())
    return logs


def _set_forecast(monkeypatch, rois, confidence=0.9):
    result = ForecastResult(
        projections=[CycleProjection(i + 1, r, 0.0, 1.0, 0.0) for i, r in enumerate(rois)],
        confidence=confidence,
        upgrade_id="u1",
    )

    class DummyForecaster:
        def __init__(self, tracker):
            self.tracker = tracker

        def forecast(self, workflow_id, patch, cycles=None, simulations=None):
            return result

    monkeypatch.setattr(dg, "UpgradeForecaster", lambda tracker: DummyForecaster(tracker))
    return result


def _base_scorecard():
    return {
        "scenario_scores": {"s": 0.8},
        "alignment_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
    }


def test_promotion_proceeds_when_all_criteria_pass(
    foresight_tracker, workflow_graph, dummy_patch, decision_logs, monkeypatch
):
    _set_forecast(monkeypatch, [1.0, 1.0], confidence=0.9)
    res = dg.evaluate_workflow(
        _base_scorecard(),
        {},
        foresight_tracker=foresight_tracker,
        workflow_id="wf",
        patch=dummy_patch,
    )
    assert res["verdict"] == "promote"
    assert res["reason_codes"] == []
    assert decision_logs and decision_logs[0]["reason_codes"] == []


def test_roi_below_threshold_downgrades_verdict(
    foresight_tracker, workflow_graph, dummy_patch, decision_logs, monkeypatch
):
    _set_forecast(monkeypatch, [0.1], confidence=0.9)
    res = dg.evaluate_workflow(
        _base_scorecard(),
        {"roi_forecast_min": 0.5},
        foresight_tracker=foresight_tracker,
        workflow_id="wf",
        patch=dummy_patch,
    )
    assert res["verdict"] == "pilot"
    assert "projected_roi_below_threshold" in res["reason_codes"]
    assert decision_logs[0]["reason_codes"] == ["projected_roi_below_threshold"]


def test_low_confidence_downgrades_verdict(
    foresight_tracker, workflow_graph, dummy_patch, decision_logs, monkeypatch
):
    _set_forecast(monkeypatch, [1.0], confidence=0.5)
    res = dg.evaluate_workflow(
        _base_scorecard(),
        {},
        foresight_tracker=foresight_tracker,
        workflow_id="wf",
        patch=dummy_patch,
    )
    assert res["verdict"] == "pilot"
    assert "low_confidence" in res["reason_codes"]
    assert decision_logs[0]["reason_codes"] == ["low_confidence"]


def test_early_collapse_downgrades_verdict(
    foresight_tracker, workflow_graph, dummy_patch, decision_logs, monkeypatch
):
    _set_forecast(monkeypatch, [1.0], confidence=0.9)
    foresight_tracker.collapse = {"risk": "Immediate collapse risk"}
    res = dg.evaluate_workflow(
        _base_scorecard(),
        {},
        foresight_tracker=foresight_tracker,
        workflow_id="wf",
        patch=dummy_patch,
    )
    assert res["verdict"] == "pilot"
    assert "roi_collapse_risk" in res["reason_codes"]
    assert decision_logs[0]["reason_codes"] == ["roi_collapse_risk"]


def test_negative_dag_impact_downgrades_verdict(
    foresight_tracker, workflow_graph, dummy_patch, decision_logs, monkeypatch
):
    _set_forecast(monkeypatch, [1.0], confidence=0.9)

    def bad_wave(wf_id, roi_delta, _):
        return {"wf": {"roi": roi_delta}, "dep": {"roi": -0.1}}

    monkeypatch.setattr(workflow_graph, "simulate_impact_wave", bad_wave)
    res = dg.evaluate_workflow(
        _base_scorecard(),
        {},
        foresight_tracker=foresight_tracker,
        workflow_id="wf",
        patch=dummy_patch,
    )
    assert res["verdict"] == "pilot"
    assert "negative_impact_wave" in res["reason_codes"]
    assert decision_logs[0]["reason_codes"] == ["negative_impact_wave"]
