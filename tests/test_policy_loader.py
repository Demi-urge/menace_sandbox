from menace_sandbox.policy_loader import load_policies


def test_load_policies_and_overrides():
    policies = load_policies()
    assert "deployment_policy" in policies
    rules = {r.decision: r for r in policies["deployment_policy"]}

    # micro pilot rule should trigger with provided metrics
    micro = rules["micro_pilot"]
    metrics = {
        "raroi": 0.6,
        "confidence": 0.6,
        "scenario_min": 0.4,
        "predicted_roi": 0.8,
        "sandbox_roi": 0.5,
    }
    assert micro.condition(metrics)

    # override condition should also evaluate to True
    override = micro.overrides["force_micro_pilot"]
    assert override.condition(metrics)
