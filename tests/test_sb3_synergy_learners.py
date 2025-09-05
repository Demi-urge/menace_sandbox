import pytest
import importlib.util
import sys
import os
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import types

sandbox = types.ModuleType("sandbox_runner")
env_mod = types.ModuleType("sandbox_runner.environment")
env_mod.simulate_full_environment = lambda *a, **k: None
env_mod.load_presets = lambda *a, **k: {}
boot_mod = types.ModuleType("sandbox_runner.bootstrap")
boot_mod.initialize_autonomous_sandbox = lambda *a, **k: None
boot_mod.main = lambda *a, **k: None
sandbox.environment = env_mod
sandbox.bootstrap = boot_mod
sys.modules.setdefault("sandbox_runner", sandbox)
sys.modules.setdefault("sandbox_runner.environment", env_mod)
sys.modules.setdefault("sandbox_runner.bootstrap", boot_mod)
spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

pytest.importorskip("torch")
import menace.self_improvement as sie


def _train(learner):
    deltas = {
        "synergy_roi": 1.0,
        "synergy_efficiency": 0.0,
        "synergy_resilience": 0.0,
        "synergy_antifragility": 0.0,
    }
    start = dict(learner.weights)
    for _ in range(3):
        learner.update(1.0, deltas)
    assert learner.weights != start
    base = os.path.splitext(learner.path)[0]
    assert Path(base + ".policy.pkl").exists()
    assert Path(base + ".target.pt").exists()
    reloaded = type(learner)(path=learner.path, lr=0.1)
    assert reloaded.weights == pytest.approx(learner.weights)


def test_td3_synergy_learner(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.TD3SynergyLearner(path=path, lr=0.1)
    _train(learner)


def test_sac_synergy_learner(tmp_path):
    path = tmp_path / "w.json"
    learner = sie.SACSynergyLearner(path=path, lr=0.1)
    _train(learner)
