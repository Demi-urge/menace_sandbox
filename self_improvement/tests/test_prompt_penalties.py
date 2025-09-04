import json
import sys
from types import SimpleNamespace, ModuleType

sys.modules.setdefault("dynamic_path_router", ModuleType("dynamic_path_router"))

from self_improvement import prompt_memory
from self_improvement.prompt_memory import record_regression
from self_improvement_policy import SelfImprovementPolicy, PolicyConfig


def test_record_regression(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(prompt_memory._settings, "prompt_penalty_path", "penalties.json")
    assert record_regression("p1") == 1
    assert record_regression("p1") == 2
    data = json.loads((tmp_path / "penalties.json").read_text())
    assert data["p1"] == 2


def test_policy_respects_penalties(monkeypatch):
    monkeypatch.setattr(
        prompt_memory, "load_prompt_penalties", lambda: {"1": 5}
    )

    class DummySettings(SimpleNamespace):
        prompt_failure_threshold = 3
        prompt_penalty_multiplier = 0.1

    monkeypatch.setattr("self_improvement_policy.SandboxSettings", lambda: DummySettings())

    policy = SelfImprovementPolicy(config=PolicyConfig(epsilon=0.0))
    state = (0,)
    policy.values[state] = {0: 2.0, 1: 10.0}
    action = policy.select_action(state)
    assert action == 0
