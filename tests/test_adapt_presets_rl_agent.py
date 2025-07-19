import os
import menace_sandbox.environment_generator as eg
import roi_tracker as rt

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

