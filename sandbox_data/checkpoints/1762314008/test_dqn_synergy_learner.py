import pytest
import importlib.util
import sys
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

pytest.importorskip("torch")
import menace.self_improvement as sie


def test_dqn_synergy_learner_updates(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.DQNSynergyLearner(path=path, lr=0.1)
    start = learner.weights["roi"]
    deltas = {"synergy_roi": 1.0, "synergy_efficiency": 0.0, "synergy_resilience": 0.0, "synergy_antifragility": 0.0}
    learner.update(1.0, deltas)
    after_inc = learner.weights["roi"]
    learner.update(-1.0, deltas)
    after_dec = learner.weights["roi"]
    assert after_inc > start
    assert after_dec < after_inc
    learner2 = sie.DQNSynergyLearner(path=path)
    assert learner2.weights["roi"] == pytest.approx(after_dec)


def test_double_dqn_synergy_learner(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.DQNSynergyLearner(path=path, lr=0.1, strategy="double_dqn")
    deltas = {"synergy_roi": 1.0, "synergy_efficiency": 0.0, "synergy_resilience": 0.0, "synergy_antifragility": 0.0}
    learner.update(1.0, deltas)
    base = os.path.splitext(path)[0]
    assert Path(base + ".policy.pkl").exists()
    assert Path(base + ".target.pt").exists()


def test_policy_gradient_synergy_learner(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.DQNSynergyLearner(path=path, lr=0.1, strategy="policy_gradient")
    deltas = {"synergy_roi": 1.0, "synergy_efficiency": 0.0, "synergy_resilience": 0.0, "synergy_antifragility": 0.0}
    learner.update(1.0, deltas)
    base = os.path.splitext(path)[0]
    assert Path(base + ".policy.pkl").exists()

