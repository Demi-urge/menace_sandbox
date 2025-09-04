import json
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

prompt_memory = importlib.import_module("menace_sandbox.self_improvement.prompt_memory")
record_regression = prompt_memory.record_regression
reset_penalty = prompt_memory.reset_penalty
load_prompt_penalties = prompt_memory.load_prompt_penalties

stub = ModuleType("self_improvement.prompt_memory")
stub.load_prompt_penalties = load_prompt_penalties
sys.modules.setdefault("self_improvement.prompt_memory", stub)
from self_improvement_policy import SelfImprovementPolicy, PolicyConfig


def test_record_regression(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(prompt_memory._settings, "prompt_penalty_path", "penalties.json")
    monkeypatch.setattr(prompt_memory, "_penalty_path", tmp_path / "penalties.json")
    assert record_regression("p1") == 1
    assert record_regression("p1") == 2
    data = json.loads((tmp_path / "penalties.json").read_text())
    assert data["p1"] == 2


def test_reset_penalty(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(prompt_memory._settings, "prompt_penalty_path", "penalties.json")
    monkeypatch.setattr(prompt_memory, "_penalty_path", tmp_path / "penalties.json")
    record_regression("p1")
    reset_penalty("p1")
    assert load_prompt_penalties()["p1"] == 0


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
