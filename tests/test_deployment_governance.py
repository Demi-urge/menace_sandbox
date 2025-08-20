import os
import json
import yaml
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.deployment_governance as dg


def _load_policy(tmp_path, policy, *, ext="yaml"):
    path = tmp_path / f"policy.{ext}"
    if ext == "json":
        path.write_text(json.dumps(policy))
    else:
        path.write_text(yaml.safe_dump(policy))
    dg._POLICY_CACHE = None
    dg._POLICY_PATH = None
    dg._load_policy(str(path))
    return path


def test_promotion_when_thresholds_met(monkeypatch):
    monkeypatch.setattr(dg, "_POLICY_CACHE", {})
    monkeypatch.setattr(dg, "_POLICY_PATH", None)
    gov = dg.DeploymentGovernor()
    scorecard = {"raroi": 1.2, "confidence": 0.8, "scenario_scores": {"s": 0.8}}
    res = gov.evaluate(scorecard, "pass", "pass", sandbox_roi=None, adapter_roi=None)
    assert res["verdict"] == "promote"
    assert res["reasons"] == []
    assert res["overrides"] == {}


@pytest.mark.parametrize(
    "scorecard",
    [
        {"raroi": 1.2, "confidence": 0.4, "scenario_scores": {"s": 0.8}},
        {"raroi": 1.2, "confidence": 0.9, "scenario_scores": {"s": 0.2}},
    ],
)
def test_demote_when_low_confidence_or_failing_scenario(tmp_path, monkeypatch, scorecard):
    policy = {"demote": {"condition": "confidence < 0.5 or min_scenario < 0.3"}}
    _load_policy(tmp_path, policy, ext="json")
    gov = dg.DeploymentGovernor()
    res = gov.evaluate(scorecard, "pass", "pass", sandbox_roi=1.0, adapter_roi=1.0)
    assert res["verdict"] == "demote"
    assert "demote" in res["reasons"]


def test_pilot_when_sandbox_roi_low_adapter_high(monkeypatch):
    monkeypatch.setattr(dg, "_POLICY_CACHE", {})
    monkeypatch.setattr(dg, "_POLICY_PATH", None)
    gov = dg.DeploymentGovernor()
    scorecard = {"raroi": 1.5, "confidence": 0.9, "scenario_scores": {"s": 0.8}}
    res = gov.evaluate(scorecard, "pass", "pass", sandbox_roi=0.05, adapter_roi=1.2)
    assert res["verdict"] == "pilot"
    assert "micro_pilot" in res["reasons"]
    assert res["overrides"].get("mode") == "micro-pilot"


@pytest.mark.parametrize(
    "alignment,security,reason",
    [("fail", "pass", "alignment_veto"), ("pass", "fail", "security_veto")],
)
def test_veto_on_alignment_or_security_failure(monkeypatch, alignment, security, reason):
    monkeypatch.setattr(dg, "_POLICY_CACHE", {})
    monkeypatch.setattr(dg, "_POLICY_PATH", None)
    gov = dg.DeploymentGovernor()
    res = gov.evaluate({}, alignment, security, sandbox_roi=None, adapter_roi=None)
    assert res["verdict"] == "demote"
    assert reason in res["reasons"]


def test_override_handling_from_policy(tmp_path, monkeypatch):
    policy = {
        "promote": {"condition": "raroi > 1.5"},
        "overrides": {"promote": {"priority": "fast"}},
    }
    _load_policy(tmp_path, policy, ext="yaml")
    gov = dg.DeploymentGovernor()
    scorecard = {"raroi": 2.0, "confidence": 0.9, "scenario_scores": {"s": 0.9}}
    res = gov.evaluate(scorecard, "pass", "pass", sandbox_roi=1.0, adapter_roi=1.0)
    assert res["verdict"] == "promote"
    assert res["overrides"] == {"promote": {"priority": "fast"}}
