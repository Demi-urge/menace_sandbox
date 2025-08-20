import os
import json
import yaml
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.deployment_governance as dg


def _load_rules(tmp_path, rules, *, ext="yaml"):
    path = tmp_path / f"rules.{ext}"
    if ext == "json":
        path.write_text(json.dumps(rules))
    else:
        path.write_text(yaml.safe_dump(rules))
    dg._RULES_CACHE = None
    dg._RULES_PATH = None
    dg._load_rules(str(path))
    return path


def test_promotion_when_thresholds_met(monkeypatch):
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)
    gov = dg.DeploymentGovernor()
    scorecard = {"scenario_scores": {"s": 0.8}}
    res = gov.evaluate(scorecard, "pass", 1.2, 0.8, sandbox_roi=None, adapter_roi=None)
    assert res["verdict"] == "promote"
    assert res["reasons"] == []
    assert res["override"] == {}


@pytest.mark.parametrize(
    "confidence,scorecard",
    [
        (0.4, {"scenario_scores": {"s": 0.8}}),
        (0.9, {"scenario_scores": {"s": 0.2}}),
    ],
)
def test_demote_when_low_confidence_or_failing_scenario(tmp_path, monkeypatch, confidence, scorecard):
    rules = [
        {
            "decision": "demote",
            "condition": "confidence < 0.5 or min_scenario < 0.3",
            "reason_code": "policy_demote",
        }
    ]
    _load_rules(tmp_path, rules, ext="json")
    gov = dg.DeploymentGovernor()
    res = gov.evaluate(scorecard, "pass", 1.2, confidence, sandbox_roi=1.0, adapter_roi=1.0)
    assert res["verdict"] == "demote"
    assert "policy_demote" in res["reasons"]


def test_pilot_when_sandbox_roi_low_adapter_high(monkeypatch):
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)
    gov = dg.DeploymentGovernor()
    scorecard = {"scenario_scores": {"s": 0.8}}
    res = gov.evaluate(scorecard, "pass", 1.5, 0.9, sandbox_roi=0.05, adapter_roi=1.2)
    assert res["verdict"] == "pilot"
    assert "micro_pilot" in res["reasons"]
    assert res["override"].get("mode") == "micro-pilot"


def test_can_bypass_micro_pilot(monkeypatch):
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)
    gov = dg.DeploymentGovernor()
    scorecard = {"scenario_scores": {"s": 0.8}}
    res = gov.evaluate(
        scorecard,
        "pass",
        1.5,
        0.9,
        sandbox_roi=0.05,
        adapter_roi=1.2,
        overrides={"bypass_micro_pilot": True},
    )
    assert res["verdict"] == "promote"
    assert "micro_pilot" not in res["reasons"]


def test_veto_on_alignment_failure(monkeypatch):
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)
    gov = dg.DeploymentGovernor()
    res = gov.evaluate({}, "fail", None, None, sandbox_roi=None, adapter_roi=None)
    assert res["verdict"] == "demote"
    assert "alignment_veto" in res["reasons"]


def test_custom_rule_precedence_over_defaults(tmp_path, monkeypatch):
    rules = [
        {"decision": "pilot", "condition": "raroi > -1", "reason_code": "always_pilot"}
    ]
    _load_rules(tmp_path, rules, ext="yaml")
    gov = dg.DeploymentGovernor()
    scorecard = {"scenario_scores": {"s": 0.9}}
    res = gov.evaluate(scorecard, "pass", 2.0, 0.9, sandbox_roi=1.0, adapter_roi=1.0)
    assert res["verdict"] == "pilot"
    assert res["reasons"] == ["always_pilot"]
