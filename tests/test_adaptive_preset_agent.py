import os
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace_sandbox.environment_generator import AdaptivePresetAgent

class DummyTracker:
    def __init__(self):
        self.roi_history = [0.0, 0.1]
        self.metrics_history = {
            "synergy_roi": [0.0, 0.05],
            "cpu_usage": [0.8, 0.9],
            "memory_usage": [0.9, 0.7],
            "threat_intensity": [0.5, 0.5],
        }


def test_state_includes_resource_metrics():
    agent = AdaptivePresetAgent()
    state = agent._state(DummyTracker())
    assert state == (1, 1, 1, -1, 0)


def test_decide_respects_metrics():
    tracker = DummyTracker()
    agent = AdaptivePresetAgent()
    agent.policy.epsilon = 0.0
    expected_state = agent._state(tracker)
    agent.policy.values[expected_state] = {i: 0.0 for i in range(len(agent.ACTIONS))}
    agent.policy.values[expected_state][2] = 1.0
    action = agent.decide(tracker)
    assert action == {"cpu": 0, "memory": 1, "threat": 0}
    reward = agent._reward(tracker)
    assert reward == pytest.approx(0.25)
