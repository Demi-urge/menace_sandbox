import pytest

from self_improvement_policy import (
    PolicyConfig,
    SelfImprovementPolicy,
    torch as policy_torch,
)
from sandbox_settings import SandboxSettings


def test_q_table_persistence(tmp_path):
    path = tmp_path / "policy.pkl"
    policy = SelfImprovementPolicy(path=str(path), epsilon=0.0)
    state = (1, 2)
    policy.update(state, reward=1.0, action=0)
    policy.save()
    assert path.exists()

    loaded = SelfImprovementPolicy(path=str(path))
    assert loaded.values == policy.values


def test_dqn_weight_persistence(tmp_path):
    if policy_torch is None:
        pytest.skip("torch not installed")
    path = tmp_path / "policy.pkl"
    policy = SelfImprovementPolicy(path=str(path), strategy="dqn", epsilon=0.0)
    state = (0, 1)
    policy.update(state, reward=1.0, next_state=(1, 1), action=0)
    policy.save()

    loaded = SelfImprovementPolicy(path=str(path), strategy="dqn", epsilon=0.0)
    q1 = policy.strategy.predict(state)
    q2 = loaded.strategy.predict(state)
    assert policy_torch.allclose(q1, q2)


def test_action_selection_from_settings():
    settings = SandboxSettings(
        exploration_epsilon=0.0,
        exploration_strategy="epsilon_greedy",
        exploration_temperature=1.0,
    )
    policy = SelfImprovementPolicy(
        epsilon=settings.exploration_epsilon,
        exploration=settings.exploration_strategy,
        temperature=settings.exploration_temperature,
    )
    state = (0,)
    policy.values[state] = {0: 1.0, 1: -1.0}
    actions = {policy.select_action(state) for _ in range(5)}
    assert actions == {0}


def test_convergence_simple_task():
    policy = SelfImprovementPolicy(epsilon=0.0)
    state = (0,)
    for _ in range(50):
        policy.update(state, reward=1.0, next_state=state, action=1)
    assert policy.score(state) == pytest.approx(10.0, rel=0.1)


def test_policy_config_serialization(tmp_path):
    cfg = PolicyConfig(
        alpha=0.3,
        gamma=0.8,
        epsilon=0.2,
        temperature=1.5,
        exploration="softmax",
        adaptive=True,
    )
    policy = SelfImprovementPolicy(config=cfg)
    path = tmp_path / "policy.json"
    policy.save_model(str(path))
    loaded = SelfImprovementPolicy.load_model(str(path))
    assert loaded.get_config() == cfg


def test_policy_config_defaults_from_settings():
    settings = SandboxSettings(
        policy_alpha=0.3,
        policy_gamma=0.8,
        policy_epsilon=0.2,
        policy_temperature=1.5,
        policy_exploration="softmax",
    )
    cfg = PolicyConfig(settings=settings)
    assert cfg.alpha == settings.policy_alpha
    assert cfg.gamma == settings.policy_gamma
    assert cfg.epsilon == settings.policy_epsilon
    assert cfg.temperature == settings.policy_temperature
    assert cfg.exploration == settings.policy_exploration


def test_momentum_adjusts_urgency():
    policy = SelfImprovementPolicy(epsilon=0.1)
    policy.adjust_for_momentum(-0.3)
    assert policy.urgency > 0.0
    assert policy.epsilon > policy.base_epsilon
    prev_urgency = policy.urgency
    policy.adjust_for_momentum(0.2)
    assert policy.urgency < prev_urgency
    assert policy.epsilon >= policy.base_epsilon
