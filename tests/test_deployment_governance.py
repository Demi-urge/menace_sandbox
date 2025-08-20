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
    policy = {"demote": {"condition": "confidence < 0.5 or min_scenario < 0.3"}}
    _load_policy(tmp_path, policy, ext="json")
    gov = dg.DeploymentGovernor()
    res = gov.evaluate(scorecard, "pass", 1.2, confidence, sandbox_roi=1.0, adapter_roi=1.0)
    assert res["verdict"] == "demote"
    assert "demote" in res["reasons"]


def test_pilot_when_sandbox_roi_low_adapter_high(monkeypatch):
    monkeypatch.setattr(dg, "_POLICY_CACHE", {})
    monkeypatch.setattr(dg, "_POLICY_PATH", None)
    gov = dg.DeploymentGovernor()
    scorecard = {"scenario_scores": {"s": 0.8}}
    res = gov.evaluate(scorecard, "pass", 1.5, 0.9, sandbox_roi=0.05, adapter_roi=1.2)
    assert res["verdict"] == "pilot"
    assert "micro_pilot" in res["reasons"]
    assert res["override"].get("mode") == "micro-pilot"


def test_veto_on_alignment_failure(monkeypatch):
    monkeypatch.setattr(dg, "_POLICY_CACHE", {})
    monkeypatch.setattr(dg, "_POLICY_PATH", None)
    gov = dg.DeploymentGovernor()
    res = gov.evaluate({}, "fail", None, None, sandbox_roi=None, adapter_roi=None)
    assert res["verdict"] == "demote"
    assert "alignment_veto" in res["reasons"]


def test_override_handling_from_policy(tmp_path, monkeypatch):
    policy = {
        "promote": {"condition": "raroi > 1.5"},
        "overrides": {"promote": {"priority": "fast"}},
    }
    _load_policy(tmp_path, policy, ext="yaml")
    gov = dg.DeploymentGovernor()
    scorecard = {"scenario_scores": {"s": 0.9}}
    res = gov.evaluate(scorecard, "pass", 2.0, 0.9, sandbox_roi=1.0, adapter_roi=1.0)
    assert res["verdict"] == "promote"
    assert res["override"] == {"priority": "fast"}
