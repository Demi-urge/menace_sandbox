import importlib
import sys


def _reload_policy(monkeypatch, state_path):
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(state_path))
    sys.modules.pop("bootstrap_timeout_policy", None)
    sys.modules.pop("menace_sandbox.bootstrap_timeout_policy", None)
    return importlib.import_module("bootstrap_timeout_policy")


def test_budget_violation_persisted(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    policy = _reload_policy(monkeypatch, state_path)

    policy.record_component_budget_violation(
        policy._PREPARE_PIPELINE_COMPONENT,
        floor=policy._COMPONENT_TIMEOUT_MINIMUMS.get("orchestrator_state", 0.0),
        shortfall=45.0,
        host_load=1.5,
        requested=90.0,
        context={"source": "test"},
    )

    violations = policy.load_component_budget_violations()
    assert policy._PREPARE_PIPELINE_COMPONENT in violations
    violation = violations[policy._PREPARE_PIPELINE_COMPONENT]
    assert violation.get("floor_shortfall") >= 45.0
    assert violation.get("recommended_budget") >= violation.get("floor", 0.0)
    assert "host_load" in violation


def test_budget_violation_feeds_budget_pools(monkeypatch, tmp_path):
    state_path = tmp_path / "timeout_state.json"
    policy = _reload_policy(monkeypatch, state_path)
    policy.record_component_budget_violation(
        "vectorizers",
        floor=120.0,
        shortfall=180.0,
        requested=90.0,
    )

    policy = _reload_policy(monkeypatch, state_path)
    pools = policy.load_component_budget_pools()
    assert pools.get("vectorizers", 0.0) >= 300.0
