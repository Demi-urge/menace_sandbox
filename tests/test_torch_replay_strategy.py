import importlib.util
import os
import random
import sys
import types

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

ROOT = os.path.dirname(os.path.dirname(__file__))

# Create minimal package structure for "menace.self_improvement"
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [ROOT]
menace_pkg.RAISE_ERRORS = False
sys.modules.setdefault("menace", menace_pkg)

si_pkg = types.ModuleType("menace.self_improvement")
si_pkg.__path__ = [os.path.join(ROOT, "self_improvement")]
sys.modules.setdefault("menace.self_improvement", si_pkg)

policy_mod = types.ModuleType("menace.self_improvement_policy")
policy_mod.torch = torch
policy_mod.ActorCriticStrategy = object
policy_mod.DQNStrategy = object
policy_mod.DoubleDQNStrategy = object
policy_mod.SelfImprovementPolicy = object
sys.modules.setdefault("menace.self_improvement_policy", policy_mod)

bootstrap_mod = types.ModuleType("sandbox_runner.bootstrap")
bootstrap_mod.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", bootstrap_mod)

spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.learners",
    os.path.join(ROOT, "self_improvement", "learners.py"),  # path-ignore
)
learners = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = learners
spec.loader.exec_module(learners)


def test_torch_replay_strategy_converges(monkeypatch):
    random.seed(0)
    torch.manual_seed(0)

    monkeypatch.setattr(
        learners,
        "get_default_synergy_weights",
        lambda: {"metric": 1.0},
    )

    strategy = learners.TorchReplayStrategy(
        net_factory=lambda i, o: torch.nn.Sequential(torch.nn.Linear(i, o)),
        optimizer_cls=torch.optim.Adam,
        lr=0.1,
        train_interval=1,
        replay_size=64,
        gamma=0.9,
        batch_size=4,
    )

    state = [1.0]
    for _ in range(200):
        strategy.update(state, 1.0, state, False)

    q_val = strategy.model(torch.tensor(state, dtype=torch.float32)).item()
    assert q_val == pytest.approx(10.0, rel=0.2)

