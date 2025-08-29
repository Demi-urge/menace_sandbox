import pytest

from self_improvement_policy import SelfImprovementPolicy, torch as policy_torch
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
