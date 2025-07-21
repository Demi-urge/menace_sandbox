import os
import random
import pytest
import menace_sandbox.environment_generator as eg
import menace_sandbox.self_improvement_policy as sp
import menace_sandbox.roi_tracker as rt

class DummyAgent:
    def __init__(self, path):
        self.path = path
    def decide(self, tracker):
        print('dummy decide')
        return {"cpu": 1, "memory": -1, "bandwidth": 1, "threat": -1}
    def save(self):
        pass

def _tracker():
    t = rt.ROITracker()
    vals = [(0.0, 0.1, 0.02), (0.1, 0.4, 0.03), (0.4, 0.5, -0.01)]
    for before, after, syn in vals:
        t.update(before, after, metrics={"security_score": 70, "synergy_roi": syn})
    return t

def test_rl_agent_actions(monkeypatch, tmp_path):
    path = tmp_path / "policy.pkl"
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", str(path))
    import types, sys
    prl = types.ModuleType("menace_sandbox.preset_rl_agent")
    prl.PresetRLAgent = DummyAgent
    sys.modules["menace_sandbox.preset_rl_agent"] = prl
    tracker = _tracker()
    presets = [{
        "CPU_LIMIT": "1",
        "MEMORY_LIMIT": "512Mi",
        "BANDWIDTH_LIMIT": "5Mbps",
        "MAX_BANDWIDTH": "10Mbps",
        "MIN_BANDWIDTH": "1Mbps",
        "THREAT_INTENSITY": 50,
    }]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["CPU_LIMIT"] != "1"
    assert new[0]["BANDWIDTH_LIMIT"] != "5Mbps"


def test_rl_agent_cpu_memory_threat(monkeypatch, tmp_path, tracker_factory):
    path = tmp_path / "policy.pkl"
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", str(path))
    monkeypatch.setattr(
        eg.AdaptivePresetAgent,
        "decide",
        lambda self, t: {"cpu": 1, "memory": -1, "threat": 1},
    )
    monkeypatch.setattr(eg.AdaptivePresetAgent, "save", lambda self: None)
    if hasattr(eg.adapt_presets, "_rl_agent"):
        delattr(eg.adapt_presets, "_rl_agent")

    tracker = tracker_factory(
        metrics={
            "security_score": [70, 70, 70],
            "synergy_roi": [0.1, 0.1, 0.1],
        }
    )
    presets = [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "512Mi", "THREAT_INTENSITY": 50}]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["CPU_LIMIT"] == "1"
    assert new[0]["MEMORY_LIMIT"] == "128Mi"
    assert new[0]["THREAT_INTENSITY"] == 70


class DummyAdaptive:
    def __init__(self, path=None, *, strategy=None):
        self.path = path
        DummyAdaptive.calls.append(path)

    def decide(self, tracker):
        DummyAdaptive.decided = True
        return {"cpu": -1, "memory": 1, "threat": 1}

    def save(self):
        pass

DummyAdaptive.calls = []
DummyAdaptive.decided = False


def _long_tracker():
    t = rt.ROITracker()
    vals = [
        (0.0, 0.05, 0.01),
        (0.05, 0.11, 0.02),
        (0.11, 0.17, 0.03),
        (0.17, 0.24, 0.04),
        (0.24, 0.31, 0.05),
        (0.31, 0.38, 0.06),
    ]
    for before, after, syn in vals:
        t.update(before, after, metrics={"security_score": 70, "synergy_roi": syn})
    return t


def test_adaptive_agent(monkeypatch):
    monkeypatch.delenv("SANDBOX_PRESET_RL_PATH", raising=False)
    monkeypatch.setattr(eg, "AdaptivePresetAgent", DummyAdaptive, raising=False)
    if hasattr(eg.adapt_presets, "_adaptive_agent"):
        delattr(eg.adapt_presets, "_adaptive_agent")
    tracker = _long_tracker()
    presets = [{
        "CPU_LIMIT": "2",
        "MEMORY_LIMIT": "512Mi",
        "THREAT_INTENSITY": 30,
    }]
    new = eg.adapt_presets(tracker, presets)
    assert DummyAdaptive.decided
    assert new[0]["MEMORY_LIMIT"] != "512Mi"
    assert new[0]["THREAT_INTENSITY"] != 30


def test_policy_determinism(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", str(tmp_path / "policy.pkl"))
    monkeypatch.delenv("SANDBOX_ADAPTIVE_AGENT_PATH", raising=False)
    if hasattr(eg.adapt_presets, "_rl_agent"):
        delattr(eg.adapt_presets, "_rl_agent")

    tracker = _long_tracker()
    base = {
        "CPU_LIMIT": "1",
        "MEMORY_LIMIT": "512Mi",
        "BANDWIDTH_LIMIT": "5Mbps",
        "MAX_BANDWIDTH": "10Mbps",
        "MIN_BANDWIDTH": "1Mbps",
        "THREAT_INTENSITY": 30,
    }
    random.seed(0)
    out1 = eg.adapt_presets(tracker, [base.copy()])

    if hasattr(eg.adapt_presets, "_rl_agent"):
        delattr(eg.adapt_presets, "_rl_agent")
    random.seed(0)
    out2 = eg.adapt_presets(tracker, [base.copy()])

    assert out1 == out2


def test_adaptive_agent_uses_synergy_metrics(monkeypatch):
    class CapturePolicy:
        records = []

        def __init__(self, path=None, strategy=None, **k):
            self.path = path
            self.strategy = strategy

        def update(self, state, reward, next_state=None, action=1):
            CapturePolicy.records.append((state, reward, next_state, action))
            return 0.0

        def select_action(self, state):
            return 0

        def save(self):
            pass

    monkeypatch.setattr(sp, "SelfImprovementPolicy", CapturePolicy)

    tracker = rt.ROITracker()
    tracker.update(0.0, 0.1, metrics={
        "security_score": 70,
        "synergy_roi": 0.05,
        "synergy_efficiency": 0.02,
        "synergy_resilience": 0.01,
    })
    agent = eg.AdaptivePresetAgent()
    agent.decide(tracker)
    tracker.update(0.1, 0.2, metrics={
        "security_score": 70,
        "synergy_roi": 0.1,
        "synergy_efficiency": 0.07,
        "synergy_resilience": 0.06,
    })
    agent.decide(tracker)

    state, reward, next_state, _ = CapturePolicy.records[-1]
    assert len(state) == 7
    assert reward == pytest.approx(0.15)


def test_strategy_env_var(monkeypatch, tmp_path):
    captured = {"strategies": []}

    class RecordPolicy:
        def __init__(self, path=None, strategy=None, **k):
            captured["strategies"].append(strategy)

        def update(self, *a, **k):
            return 0.0

        def select_action(self, state):
            return 0

        def save(self):
            pass

    monkeypatch.setattr(sp, "SelfImprovementPolicy", RecordPolicy)
    path = tmp_path / "p.pkl"
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", str(path))
    monkeypatch.setenv("SANDBOX_PRESET_RL_STRATEGY", "actor_critic")
    if hasattr(eg.adapt_presets, "_rl_agent"):
        delattr(eg.adapt_presets, "_rl_agent")
    tracker = _tracker()
    eg.adapt_presets(tracker, [{"CPU_LIMIT": "1"}])
    assert any(
        isinstance(s, sp.ActorCriticStrategy) for s in captured["strategies"]
    )


def test_adaptive_strategy_env_var(monkeypatch, tmp_path):
    captured = {"strategies": []}

    class RecordPolicy:
        def __init__(self, path=None, strategy=None, **k):
            captured["strategies"].append(strategy)

        def update(self, *a, **k):
            return 0.0

        def select_action(self, state):
            return 0

        def save(self):
            pass

    monkeypatch.setattr(sp, "SelfImprovementPolicy", RecordPolicy)
    path = tmp_path / "p.pkl"
    monkeypatch.setenv("SANDBOX_ADAPTIVE_AGENT_PATH", str(path))
    monkeypatch.setenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY", "double_dqn")
    if hasattr(eg.adapt_presets, "_adaptive_agent"):
        delattr(eg.adapt_presets, "_adaptive_agent")
    tracker = _long_tracker()
    eg.adapt_presets(tracker, [{"CPU_LIMIT": "1"}])
    expected = type(sp.strategy_factory("double_dqn"))
    assert any(isinstance(s, expected) for s in captured["strategies"])


def test_adaptive_agent_env_default(monkeypatch):
    captured = {"strategies": []}

    class RecordPolicy:
        def __init__(self, path=None, strategy=None, **k):
            captured["strategies"].append(strategy)

        def update(self, *a, **k):
            return 0.0

        def select_action(self, state):
            return 0

        def save(self):
            pass

    monkeypatch.setattr(sp, "SelfImprovementPolicy", RecordPolicy)
    monkeypatch.setenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY", "double_dqn")
    agent = eg.AdaptivePresetAgent()
    expected = type(sp.strategy_factory("double_dqn"))
    assert any(isinstance(s, expected) for s in captured["strategies"])

