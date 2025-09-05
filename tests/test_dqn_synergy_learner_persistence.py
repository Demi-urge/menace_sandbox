import json
import pytest

import importlib.util
import os
import sys
import types

from dynamic_path_router import resolve_path

import menace

menace.RAISE_ERRORS = False

ROOT = os.path.dirname(os.path.dirname(__file__))

# set up minimal package hierarchy to load learners without heavy deps
menace_pkg = sys.modules.setdefault("menace", menace)
si_pkg = types.ModuleType("menace.self_improvement")
si_pkg.__path__ = [os.path.join(ROOT, "self_improvement")]
sys.modules.setdefault("menace.self_improvement", si_pkg)

policy_mod = types.ModuleType("menace.self_improvement_policy")
# minimal strategy stub
class DummyStrategy:
    def __init__(self, **kw):
        self.memory = []

    def update(self, table, state, action, reward, next_state, *args):
        self.memory.append((state, action, reward, next_state, False))
        return reward

    def predict(self, state):
        return state

policy_mod.ActorCriticStrategy = DummyStrategy
policy_mod.DQNStrategy = DummyStrategy
policy_mod.DoubleDQNStrategy = DummyStrategy
policy_mod.SelfImprovementPolicy = object
policy_mod.torch = None
sys.modules.setdefault("menace.self_improvement_policy", policy_mod)

bootstrap_mod = types.ModuleType("sandbox_runner.bootstrap")
bootstrap_mod.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", bootstrap_mod)

spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.learners",
    resolve_path("self_improvement/learners.py"),  # path-ignore
)
sie = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sie
spec.loader.exec_module(sie)


def test_dqn_weight_persistence_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(sie, "sip_torch", None)
    path = tmp_path / "w.json"
    learner = sie.DQNSynergyLearner(path=path, lr=0.5)
    start = learner.weights.copy()
    deltas = {
        "synergy_roi": 0.5,
        "synergy_efficiency": 0.2,
        "synergy_resilience": -0.3,
        "synergy_antifragility": 0.1,
    }
    learner.update(1.0, deltas)
    changed = learner.weights.copy()
    assert changed != start
    learner2 = sie.DQNSynergyLearner(path=path, lr=0.5)
    assert learner2.weights == pytest.approx(changed)

