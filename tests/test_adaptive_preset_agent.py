import os
import pytest
import types
import sys
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules["numpy"].isscalar = lambda x: isinstance(x, (int, float, complex))
sys.modules["numpy"].bool_ = bool
sys.modules["numpy"].ndarray = type("ndarray", (), {})
np_random = types.ModuleType("numpy.random")
np_random.seed = lambda *a, **k: None
np_random.get_state = lambda: None
np_random.set_state = lambda state: None
sys.modules.setdefault("numpy.random", np_random)
sys.modules["numpy"].random = np_random
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
sys.modules.setdefault("sklearn.linear_model", types.ModuleType("sklearn.linear_model"))
sys.modules.setdefault("sklearn.preprocessing", types.ModuleType("sklearn.preprocessing"))
sys.modules["sklearn.linear_model"].LinearRegression = object
sys.modules["sklearn.preprocessing"].PolynomialFeatures = object

import environment_generator as eg

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck

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


class PropertyTracker:
    def __init__(
        self,
        roi,
        syn,
        eff,
        res,
        cpu,
        mem,
        threat,
    ):
        self.roi_history = list(roi)
        self.metrics_history = {
            "synergy_roi": list(syn),
            "synergy_efficiency": list(eff),
            "synergy_resilience": list(res),
            "cpu_usage": list(cpu),
            "memory_usage": list(mem),
            "threat_intensity": list(threat),
        }


def _expected_reward(roi, syn, eff, res, cpu, mem, threat):
    roi_delta = roi[-1] - roi[-2] if len(roi) >= 2 else 0.0
    syn_delta = (
        syn[-1] - syn[-2]
        if len(syn) >= 2
        else (syn[-1] if syn else 0.0)
    )
    eff_delta = (
        eff[-1] - eff[-2]
        if len(eff) >= 2
        else (eff[-1] if eff else 0.0)
    )
    res_delta = (
        res[-1] - res[-2]
        if len(res) >= 2
        else (res[-1] if res else 0.0)
    )
    cpu_delta = cpu[-1] - cpu[-2] if len(cpu) >= 2 else 0.0
    mem_delta = mem[-1] - mem[-2] if len(mem) >= 2 else 0.0
    threat_delta = threat[-1] - threat[-2] if len(threat) >= 2 else 0.0
    return float(
        roi_delta
        + syn_delta
        + eff_delta
        + res_delta
        - cpu_delta
        - mem_delta
        - threat_delta
    )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    roi=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    syn=st.lists(st.floats(-1, 1), min_size=0, max_size=5),
    eff=st.lists(st.floats(-1, 1), min_size=0, max_size=5),
    res=st.lists(st.floats(-1, 1), min_size=0, max_size=5),
    cpu=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    mem=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    threat=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
)
def test_reward_matches_formula(roi, syn, eff, res, cpu, mem, threat):
    tracker = PropertyTracker(roi, syn, eff, res, cpu, mem, threat)
    agent = AdaptivePresetAgent()
    reward = agent._reward(tracker)
    expected = _expected_reward(roi, syn, eff, res, cpu, mem, threat)
    assert reward == pytest.approx(expected)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    roi=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    syn=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    eff=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    res=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    cpu=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    mem=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
    threat=st.lists(st.floats(-1, 1), min_size=2, max_size=5),
)
def test_decide_action_within_bounds(roi, syn, eff, res, cpu, mem, threat):
    tracker = PropertyTracker(roi, syn, eff, res, cpu, mem, threat)
    agent = AdaptivePresetAgent()
    state = agent._state(tracker)
    agent.policy.epsilon = 0.0
    agent.policy.values[state] = {i: 0.0 for i in range(len(agent.ACTIONS))}
    action = agent.decide(tracker)
    assert action in [dict(a) for a in agent.ACTIONS]
