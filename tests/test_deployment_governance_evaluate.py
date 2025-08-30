import json
import menace_sandbox.deployment_governance as dg
from menace_sandbox.override_validator import generate_signature


def _reset(monkeypatch):
    monkeypatch.setattr(dg, "_POLICY_CACHE", None)


def test_evaluate_honours_override(tmp_path, monkeypatch):
    _reset(monkeypatch)
    key = tmp_path / "key"
    key.write_text("secret")
    data = {"verdict": "promote"}
    sig = generate_signature(data, str(key))
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"data": data, "signature": sig}))
    scorecard = {
        "alignment": {"status": "pass", "rationale": ""},
        "security": "pass",
    }
    metrics = {"raroi": 0.1, "confidence": 0.2}
    res = dg.evaluate(
        scorecard,
        metrics,
        policy={"override_path": str(override), "public_key_path": str(key)},
    )
    assert res["verdict"] == "promote"
    assert "manual_override" in res["reasons"]
