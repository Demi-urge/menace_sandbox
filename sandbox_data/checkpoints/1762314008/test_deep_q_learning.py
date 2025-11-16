import pytest

from menace_sandbox.self_improvement_policy import SelfImprovementPolicy
from menace_sandbox.environment_generator import AdaptivePresetAgent


class LearnTracker:
    def __init__(self):
        self.roi_history = [0.0, 0.0]
        self.metrics_history = {
            "synergy_roi": [0.0, 0.0],
            "synergy_efficiency": [0.0, 0.0],
            "synergy_resilience": [0.0, 0.0],
            "cpu_usage": [0.0, 0.0],
            "memory_usage": [0.0, 0.0],
            "threat_intensity": [0.0, 0.0],
        }

    def step(self, action_idx: int):
        inc = 1.0 if action_idx == 0 else 0.0
        self.roi_history.append(self.roi_history[-1] + inc)
        for k in ["synergy_roi", "synergy_efficiency", "synergy_resilience"]:
            self.metrics_history[k].append(self.metrics_history[k][-1] + 0.5 * inc)
        for k in ["cpu_usage", "memory_usage", "threat_intensity"]:
            self.metrics_history[k].append(self.metrics_history[k][-1])


def test_deep_q_learning_strategy_learns():
    pytest.importorskip("torch")
    policy = SelfImprovementPolicy(strategy="deep_q", epsilon=0.2)
    state0 = (0, 0, 0)
    for _ in range(60):
        act = policy.select_action(state0)
        reward = 1.0 if act == 1 else 0.0
        next_state = (1, 0, 0) if act == 1 else state0
        policy.update(state0, reward, next_state, action=act)
    policy.epsilon = 0.0
    assert policy.select_action(state0) == 1


def test_deep_q_agent_learns():
    pytest.importorskip("torch")
    tracker = LearnTracker()
    agent = AdaptivePresetAgent(strategy="deep_q")
    agent.policy.epsilon = 0.2
    for _ in range(80):
        agent.decide(tracker)
        idx = agent.prev_action
        tracker.step(idx)
    agent.policy.epsilon = 0.0
    agent.decide(tracker)
    assert agent.prev_action == 0
