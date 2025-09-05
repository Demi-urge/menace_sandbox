import json
import os
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace, ModuleType

root = Path(__file__).resolve().parents[2]
root_pkg = ModuleType("menace_sandbox")
root_pkg.__path__ = [str(root)]
sys.modules["menace_sandbox"] = root_pkg
pkg = ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(root / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

dyn = ModuleType("dynamic_path_router")
dyn.resolve_path = lambda p: p
dyn.repo_root = lambda: ""
sys.modules.setdefault("dynamic_path_router", dyn)
boot = ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)

settings_mod = ModuleType("sandbox_settings")

from dynamic_path_router import resolve_path

class DummySettings:
    def __init__(self):
        self.prompt_penalty_path = resolve_path("penalties.json")
        self.prompt_success_log_path = resolve_path("success.log")
        self.prompt_failure_log_path = resolve_path("failure.log")
        self.sandbox_repo_path = "."
        self.sandbox_data_dir = os.getenv("SANDBOX_DATA_DIR", ".")
        self.prompt_failure_threshold = 3
        self.prompt_penalty_multiplier = 1.0


settings_mod.SandboxSettings = DummySettings
settings_mod.load_sandbox_settings = lambda: DummySettings()
sys.modules.setdefault("sandbox_settings", settings_mod)
SandboxSettings = DummySettings

prompt_memory = importlib.import_module("menace_sandbox.self_improvement.prompt_memory")
record_regression = prompt_memory.record_regression
reset_penalty = prompt_memory.reset_penalty
load_prompt_penalties = prompt_memory.load_prompt_penalties

stub = ModuleType("self_improvement.prompt_memory")
stub.load_prompt_penalties = load_prompt_penalties
sys.modules.setdefault("self_improvement.prompt_memory", stub)

policy_mod = ModuleType("self_improvement_policy")


class PolicyConfig:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon


def policy_load_prompt_penalties() -> dict:
    return {}


class SelfImprovementPolicy:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.values: dict = {}

    def select_action(self, state):
        penalties = policy_mod.load_prompt_penalties()
        settings = policy_mod.SandboxSettings()
        actions = self.values[state].copy()
        for action, value in list(actions.items()):
            if penalties.get(str(action), 0) >= settings.prompt_failure_threshold:
                actions[action] = float("-inf")
        return max(actions, key=actions.get)


policy_mod.PolicyConfig = PolicyConfig
policy_mod.SelfImprovementPolicy = SelfImprovementPolicy
policy_mod.load_prompt_penalties = policy_load_prompt_penalties
policy_mod.SandboxSettings = SandboxSettings
sys.modules.setdefault("self_improvement_policy", policy_mod)

from self_improvement_policy import SelfImprovementPolicy, PolicyConfig


def test_record_regression(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    assert record_regression("p1") == 1
    assert record_regression("p1") == 2
    assert load_prompt_penalties()["p1"] == 2


def test_reset_penalty(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    record_regression("p1")
    reset_penalty("p1")
    assert load_prompt_penalties().get("p1", 0) == 0


def test_policy_respects_penalties(monkeypatch):
    monkeypatch.setattr("self_improvement_policy.load_prompt_penalties", lambda: {"1": 5})

    class DummySettings(SimpleNamespace):
        prompt_failure_threshold = 3
        prompt_penalty_multiplier = 0.1
        policy_alpha = 0.5
        policy_gamma = 0.9
        policy_epsilon = 0.1
        policy_temperature = 1.0
        policy_exploration = "epsilon_greedy"

    monkeypatch.setattr("self_improvement_policy.SandboxSettings", lambda: DummySettings())

    policy = SelfImprovementPolicy(config=PolicyConfig(epsilon=0.0))
    state = (0,)
    policy.values[state] = {0: 2.0, 1: 10.0}
    action = policy.select_action(state)
    assert action == 0
