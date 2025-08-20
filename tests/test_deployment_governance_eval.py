import menace.deployment_governance as dg


def test_alignment_veto():
    res = dg.evaluate({"alignment": "fail"}, {})
    assert res["verdict"] == "demote"
    assert "alignment_veto" in res["reasons"]
    assert res["overridable"] is False


def test_promote_path():
    scorecard = {"alignment": "pass", "security": "pass"}
    metrics = {
        "raroi": 0.9,
        "confidence": 0.7,
        "scenario_scores": {"normal": 0.6, "stress": 0.55},
        "sandbox_roi": 0.2,
        "adapter_roi": 0.25,
    }
    res = dg.evaluate(scorecard, metrics)
    assert res["verdict"] == "promote"
    assert "meets_promotion_criteria" in res["reasons"]


def test_micro_pilot_path():
    scorecard = {"alignment": "pass", "security": "pass"}
    metrics = {
        "raroi": 0.55,
        "confidence": 0.52,
        "scenario_scores": {"normal": 0.4, "stress": 0.35},
        "sandbox_roi": 0.1,
        "adapter_roi": 0.2,
        "predicted_roi": 0.3,
    }
    res = dg.evaluate(scorecard, metrics)
    assert res["verdict"] == "micro_pilot"
    assert "evaluate_via_micro_pilot" in res["reasons"]
    assert "predicted_roi_exceeds_sandbox_roi" in res["reasons"]
