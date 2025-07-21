import self_improvement_policy as sip
import pytest


def test_policy_update():
    policy = sip.SelfImprovementPolicy()
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0)
    before = policy.score(state)
    policy.update(state, 1.0)
    after = policy.score(state)
    assert after > before


def test_policy_persistence(tmp_path):
    path = tmp_path / "policy.pkl"
    policy = sip.SelfImprovementPolicy(path=path)
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0)
    policy.update(state, 2.0)
    policy2 = sip.SelfImprovementPolicy(path=path)
    assert policy2.score(state) == policy.score(state)


def test_policy_adaptive_hyperparams():
    policy = sip.SelfImprovementPolicy(alpha=0.5, gamma=0.9, adaptive=True)
    state = (1,) * 15
    for _ in range(5):
        policy.update(state, 1.0)
    assert policy.alpha < 0.5
    assert policy.gamma <= 0.9


def test_policy_action_selection():
    policy = sip.SelfImprovementPolicy(epsilon=0.0)
    state = (2,) * 15
    policy.update(state, 1.0)
    assert policy.select_action(state) == 1
    policy = sip.SelfImprovementPolicy(epsilon=1.0)
    counts = {0: 0, 1: 0}
    for _ in range(50):
        counts[policy.select_action(state)] += 1
    assert counts[0] > 0 and counts[1] > 0


def test_q_lambda_strategy():
    policy = sip.SelfImprovementPolicy(strategy="q_lambda")
    state = (3,) * 15
    before = policy.score(state)
    policy.update(state, 1.0)
    assert policy.score(state) > before


def test_actor_critic_strategy():
    policy = sip.SelfImprovementPolicy(strategy="actor_critic")
    state = (4,) * 15
    before = policy.score(state)
    policy.update(state, 1.0)
    assert policy.score(state) > before


def test_exploration_schedule():
    def decay(ep, value):
        return value * 0.9

    policy = sip.SelfImprovementPolicy(epsilon=1.0, epsilon_schedule=decay)
    state = (5,) * 15
    policy.update(state, 1.0)
    assert policy.epsilon < 1.0


def test_default_schedule_unchanged():
    policy = sip.SelfImprovementPolicy(epsilon=0.5)
    state = (6,) * 15
    policy.update(state, 1.0)
    policy.update(state, 1.0)
    assert policy.epsilon == 0.5


def test_reward_includes_synergy():
    policy = sip.SelfImprovementPolicy()
    state = (0,) * 15 + (5, 5, 0, 0)
    next_state = (0,) * 15 + (6, 7, 0, 0)
    before = policy.score(state)
    policy.update(state, 0.0, next_state)
    after = policy.score(state)
    assert after > before


def test_dqn_strategy_learns():
    pytest.importorskip("torch")
    policy = sip.SelfImprovementPolicy(strategy="dqn", epsilon=0.2)
    state0 = (0, 0, 0)
    for _ in range(100):
        act = policy.select_action(state0)
        reward = 1.0 if act == 1 else 0.0
        next_state = (1, 0, 0) if act == 1 else state0
        policy.update(state0, reward, next_state, action=act)
        policy.update(next_state, 0.0, state0, action=0)
    policy.epsilon = 0.0
    assert policy.select_action(state0) == 1


def test_deep_q_learning_strategy_learns():
    pytest.importorskip("torch")
    policy = sip.SelfImprovementPolicy(strategy="deep_q", epsilon=0.2)
    state0 = (0, 0, 0)
    for _ in range(60):
        act = policy.select_action(state0)
        reward = 1.0 if act == 1 else 0.0
        next_state = (1, 0, 0) if act == 1 else state0
        policy.update(state0, reward, next_state, action=act)
    policy.epsilon = 0.0
    assert policy.select_action(state0) == 1


def test_dqn_value_uses_predict(monkeypatch):
    pytest.importorskip("torch")
    strat = sip.DQNStrategy()
    import torch
    monkeypatch.setattr(strat, "predict", lambda s: torch.tensor([1.0, 2.0]))
    val = strat.value({}, (0, 0, 0))
    assert val == 2.0


def test_configurable_policy_env(monkeypatch):
    monkeypatch.setenv("SELF_IMPROVEMENT_STRATEGY", "sarsa")
    policy = sip.ConfigurableSelfImprovementPolicy()
    assert isinstance(policy.strategy, sip.SarsaStrategy)


def test_configurable_policy_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"strategy": "q_lambda"}')
    monkeypatch.setenv("SELF_IMPROVEMENT_CONFIG", str(cfg))
    policy = sip.ConfigurableSelfImprovementPolicy()
    assert isinstance(policy.strategy, sip.QLambdaStrategy)


def test_available_strategies_exposed():
    assert "q_learning" in sip.ConfigurableSelfImprovementPolicy.available_strategies
