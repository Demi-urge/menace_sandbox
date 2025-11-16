import numpy as np

from actor_critic_agent import ActorCriticAgent
from sandbox_settings import SandboxSettings


def test_actor_critic_learns_and_persists(tmp_path):
    np.random.seed(0)
    cfg = SandboxSettings(
        ac_actor_lr=0.5,
        ac_critic_lr=0.5,
        ac_gamma=0.9,
        ac_epsilon=0.5,
        ac_epsilon_decay=0.9,
        ac_buffer_size=200,
        ac_batch_size=32,
        ac_checkpoint_path=str(tmp_path / "ac_state.json"),
    )
    agent = ActorCriticAgent(2, 2, cfg)
    states = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    optimal = [0, 1]
    for _ in range(300):
        idx = np.random.randint(2)
        s = states[idx]
        a = agent.select_action(s)
        r = 1.0 if a == optimal[idx] else 0.0
        ns = states[np.random.randint(2)]
        agent.store(s, a, r, ns)
        agent.learn()
    agent.epsilon = 0.0
    agent.save()
    for i, s in enumerate(states):
        assert agent.select_action(s) == optimal[i]
    cfg2 = SandboxSettings(
        ac_actor_lr=0.1,
        ac_critic_lr=0.1,
        ac_gamma=0.5,
        ac_epsilon=0.9,
        ac_epsilon_decay=0.5,
        ac_buffer_size=50,
        ac_batch_size=4,
        ac_checkpoint_path=str(tmp_path / "ac_state.json"),
    )
    agent2 = ActorCriticAgent(2, 2, cfg2)
    agent2.epsilon = 0.0
    for i, s in enumerate(states):
        assert agent2.select_action(s) == optimal[i]
    assert agent2.actor_lr == cfg.actor_critic.actor_lr


def test_state_normalisation_and_reward_scaling(tmp_path):
    cfg = SandboxSettings(
        ac_reward_scale=0.5,
        ac_checkpoint_path=str(tmp_path / "ac_state.json"),
    )
    agent = ActorCriticAgent(2, 2, cfg)
    s = np.array([1.0, 0.0])
    ns = np.array([0.0, 1.0])
    agent.store(s, 0, 1.0, ns)
    assert agent.replay.data[0].reward == 0.5
    assert not np.allclose(agent.replay.data[0].state, s)
