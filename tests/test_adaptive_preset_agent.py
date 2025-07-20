import os
import pytest
import types
import sys
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules["numpy"].isscalar = lambda x: isinstance(x, (int, float, complex))
sys.modules["numpy"].bool_ = bool
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
sys.modules.setdefault("sklearn.linear_model", types.ModuleType("sklearn.linear_model"))
sys.modules.setdefault("sklearn.preprocessing", types.ModuleType("sklearn.preprocessing"))
sys.modules["sklearn.linear_model"].LinearRegression = object
sys.modules["sklearn.preprocessing"].PolynomialFeatures = object

import environment_generator as eg

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace_sandbox.environment_generator import AdaptivePresetAgent

class DummyTracker:
    def __init__(self):
        self.roi_history = [0.0, 0.1]
        self.metrics_history = {
            "synergy_roi": [0.0, 0.05],
            "synergy_efficiency": [0.0, 0.02],
            "synergy_resilience": [0.0, -0.01],
            "cpu_usage": [0.8, 0.9],
            "memory_usage": [0.9, 0.7],
            "threat_intensity": [0.5, 0.5],
        }


def test_state_includes_resource_metrics():
    agent = AdaptivePresetAgent()
    state = agent._state(DummyTracker())
    assert state == (1, 1, 1, -1, 1, -1, 0)


def test_decide_respects_metrics():
    tracker = DummyTracker()
    agent = AdaptivePresetAgent()
    agent.policy.epsilon = 0.0
    expected_state = agent._state(tracker)
    agent.policy.values[expected_state] = {
        i: 0.0 for i in range(len(agent.ACTIONS))
    }
    agent.policy.values[expected_state][2] = 1.0
    action = agent.decide(tracker)
    assert action == {"cpu": 0, "memory": 1, "threat": 0}
    reward = agent._reward(tracker)
    assert reward == pytest.approx(0.26)


def test_state_persistence(tmp_path):
    path = tmp_path / "policy.pkl"
    agent = AdaptivePresetAgent(str(path))
    agent.prev_state = (1, 0, 0, 0, 0, 0, 0)
    agent.prev_action = 3
    agent._save_state()

    loaded = AdaptivePresetAgent(str(path))
    assert loaded.prev_state == agent.prev_state
    assert loaded.prev_action == agent.prev_action


class SimpleTracker:
    def __init__(self, hist, metrics):
        self.roi_history = hist
        self.metrics_history = metrics

    def diminishing(self):
        return 0.01

    def predict_synergy_metric(self, name: str) -> float:
        return 0.0


def _tracker_with_metric(name: str, values: list[float]):
    roi_hist = [0.0, 0.1, 0.2]
    metrics = {"security_score": [70] * len(values), name: list(values)}
    return SimpleTracker(roi_hist, metrics)


def test_adapt_presets_threat_intensity_increase(monkeypatch):
    tracker = _tracker_with_metric("synergy_code_quality", [0.06, 0.07, 0.08])
    monkeypatch.setattr(tracker, "predict_synergy_metric", lambda name: 0.0)
    presets = [{"THREAT_INTENSITY": 30}]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 30


def test_adapt_presets_threat_intensity_decrease(monkeypatch):
    tracker = _tracker_with_metric("synergy_code_quality", [-0.06, -0.07, -0.08])
    monkeypatch.setattr(tracker, "predict_synergy_metric", lambda name: 0.0)
    presets = [{"THREAT_INTENSITY": 70}]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] < 70
