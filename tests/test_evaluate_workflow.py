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
