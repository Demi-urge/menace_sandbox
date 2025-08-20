import json
import yaml
import pytest

from menace.override_validator import generate_signature
import menace.deployment_governance as dg


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
