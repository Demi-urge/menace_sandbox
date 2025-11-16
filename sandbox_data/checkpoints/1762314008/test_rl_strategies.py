import pytest
import menace_sandbox.self_improvement_policy as sip


def test_qlearning_update():
    strat = sip.QLearningStrategy()
    table = {}
    strat.update(table, (0,), 1, 1.0, None, 0.5, 0.5)
    assert table[(0,)][1] == pytest.approx(0.5)


def test_qlambda_update():
    strat = sip.QLambdaStrategy()
    table = {}
    strat.update(table, (0,), 1, 1.0, None, 0.5, 0.5)
    assert table[(0,)][1] == pytest.approx(0.5)


def test_sarsa_update():
    strat = sip.SarsaStrategy()
    table = {}
    strat.update(table, (0,), 1, 1.0, (1,), 0.5, 0.5)
    assert table[(0,)][1] == pytest.approx(0.5)


def test_actor_critic_update():
    strat = sip.ActorCriticStrategy()
    table = {}
    q = strat.update(table, (0,), 1, 1.0, (1,), 0.5, 0.5)
    assert isinstance(q, float)
    if sip.torch is None:
        assert table[(0,)][1] == pytest.approx(0.5)


def test_actor_critic_update_without_torch(monkeypatch):
    monkeypatch.setattr(sip, "torch", None)
    monkeypatch.setattr(sip, "nn", None)
    monkeypatch.setattr(sip, "F", None)
    strat = sip.ActorCriticStrategy()
    table = {}
    q = strat.update(table, (0,), 1, 1.0, (1,), 0.5, 0.5)
    assert q == pytest.approx(0.5)
    assert table[(0,)][1] == pytest.approx(0.5)


def test_dimension_validation():
    cfg = sip.QLearningConfig(state_dim=2, action_dim=2)
    strat = sip.QLearningStrategy(cfg)
    with pytest.raises(ValueError):
        strat.update({}, (0,), 1, 1.0, None, 0.5, 0.5)
    with pytest.raises(ValueError):
        strat.update({}, (0, 0), 3, 1.0, None, 0.5, 0.5)


def test_actor_critic_invalid_action():
    cfg = sip.ActorCriticConfig(state_dim=1, action_dim=1)
    strat = sip.ActorCriticStrategy(cfg)
    with pytest.raises(ValueError):
        strat.update({}, (0,), 2, 1.0, None, 0.5, 0.5)


def test_policy_serialization_roundtrip():
    cfg = sip.QLearningConfig(state_dim=1, action_dim=2)
    policy = sip.SelfImprovementPolicy(strategy=sip.QLearningStrategy(cfg))
    policy.update((0,), 1.0)
    data = policy.to_json()
    restored = sip.SelfImprovementPolicy.from_json(data)
    assert restored.score((0,)) == pytest.approx(policy.score((0,)))


def test_policy_save_load(tmp_path):
    cfg = sip.QLearningConfig(state_dim=1, action_dim=2)
    policy = sip.SelfImprovementPolicy(strategy=sip.QLearningStrategy(cfg))
    policy.update((0,), 1.0)
    path = tmp_path / "policy.pkl"
    policy.save(str(path))
    new_policy = sip.SelfImprovementPolicy(strategy=sip.QLearningStrategy(cfg))
    new_policy.load(str(path))
    assert new_policy.score((0,)) == pytest.approx(policy.score((0,)))


def test_dqn_training_example():
    if sip.torch is None:
        pytest.skip("torch not available")
    cfg = sip.DQNConfig(state_dim=2, action_dim=2, hidden_dim=8, batch_size=1, capacity=10)
    strat = sip.DQNStrategy(cfg)
    table: dict = {}
    q = strat.update(table, (0, 0), 1, 1.0, (0, 1), 0.5, 0.9)
    assert isinstance(q, float)
