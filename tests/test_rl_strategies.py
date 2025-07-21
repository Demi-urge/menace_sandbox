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
    strat.update(table, (0,), 1, 1.0, (1,), 0.5, 0.5)
    assert table[(0,)][1] == pytest.approx(0.5)
    assert strat.state_values[(0,)] == pytest.approx(0.5)
