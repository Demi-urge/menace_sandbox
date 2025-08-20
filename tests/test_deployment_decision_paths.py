import pytest

import menace.deployment_governance as dg


@pytest.fixture(autouse=True)
def reset_rules(monkeypatch):
    """Ensure rule cache is cleared before each test."""
    monkeypatch.setattr(dg, "_RULES_CACHE", None)
    monkeypatch.setattr(dg, "_RULES_PATH", None)


def test_promote_high_raroi_confidence():
    scorecard = {
        "alignment_status": "pass",
        "raroi": 1.3,
        "confidence": 0.9,
        "scenario_scores": {"s": 0.9},
    }
    result = dg.evaluate_workflow(scorecard, {})
    assert result["verdict"] == "promote"
    assert result["reason_codes"] == []
    assert result["overrides"] == {}


def test_pilot_low_sandbox_high_predicted():
    scorecard = {
        "alignment_status": "pass",
        "raroi": 1.5,
        "confidence": 0.9,
        "sandbox_roi": 0.05,
        "adapter_roi": 1.5,
        "scenario_scores": {"s": 0.8},
    }
    result = dg.evaluate_workflow(scorecard, {})
    assert result["verdict"] == "pilot"
    assert "micro_pilot" in result["reason_codes"]
    assert result["overrides"].get("mode") == "micro-pilot"


def test_demote_scenario_inconsistency():
    scorecard = {
        "alignment_status": "pass",
        "security_status": "pass",
        "raroi": 1.2,
        "confidence": 0.8,
        "scenario_scores": {"good": 0.9, "bad": 0.4},
    }
    result = dg.evaluate_scorecard(scorecard)
    assert result["decision"] == "demote"
    assert "scenario_below_min" in result["reason_codes"]
    assert result["override_allowed"] is True
