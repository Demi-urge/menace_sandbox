import pytest
import importlib.util
import sys
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

pytest.importorskip("torch")
import menace.self_improvement as sie


def test_td3_synergy_learner(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.TD3SynergyLearner(path=path, lr=0.1)
    deltas = {"synergy_roi": 1.0, "synergy_efficiency": 0.0, "synergy_resilience": 0.0, "synergy_antifragility": 0.0}
    learner.update(1.0, deltas)
    base = os.path.splitext(path)[0]
    assert Path(base + ".policy.pkl").exists()
    assert Path(base + ".target.pt").exists()


def test_sac_synergy_learner(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.SACSynergyLearner(path=path, lr=0.1)
    deltas = {"synergy_roi": 1.0, "synergy_efficiency": 0.0, "synergy_resilience": 0.0, "synergy_antifragility": 0.0}
    learner.update(1.0, deltas)
    base = os.path.splitext(path)[0]
    assert Path(base + ".policy.pkl").exists()
