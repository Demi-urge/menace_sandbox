import pickle
from pathlib import Path

import menace_sandbox.environment_generator as eg
import roi_tracker as rt


def _tracker():
    t = rt.ROITracker()
    vals = [(0.0, 0.1, 0.02), (0.1, 0.4, 0.03), (0.4, 0.5, -0.01)]
    for before, after, syn in vals:
        t.update(before, after, metrics={"security_score": 70, "synergy_roi": syn})
    return t


def test_policy_persists(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SANDBOX_PRESET_RL_PATH", raising=False)
    if hasattr(eg.adapt_presets, "_rl_agent"):
        delattr(eg.adapt_presets, "_rl_agent")

    tracker = _tracker()
    eg.adapt_presets(tracker, [{"CPU_LIMIT": "1"}])
    policy_file = Path("sandbox_data") / "preset_policy.json"
    assert policy_file.exists()

    with open(policy_file, "wb") as fh:
        pickle.dump({(99,): {0: 0.5}}, fh)

    if hasattr(eg.adapt_presets, "_rl_agent"):
        delattr(eg.adapt_presets, "_rl_agent")
    tracker = _tracker()
    eg.adapt_presets(tracker, [{"CPU_LIMIT": "1"}])
    policy = eg.export_preset_policy()
    assert (99,) in policy


class _FailAgent:
    def __init__(self, path=None, *, strategy=None):
        self.policy = type("p", (), {"path": path, "strategy": strategy})()

    def decide(self, tracker):
        raise RuntimeError("boom")

    def save(self):
        pass


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


def test_rl_agent_exception_logged(monkeypatch, tmp_path, caplog):
    path = tmp_path / "p.pkl"
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", str(path))
    monkeypatch.setattr(eg.adapt_presets, "_adaptive_agent", None, raising=False)
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", _FailAgent(str(path)), raising=False)
    caplog.set_level("ERROR")
    tracker = _tracker()
    eg.adapt_presets(tracker, [{"CPU_LIMIT": "1"}])
    assert "preset adaptation failed" in caplog.text


def test_adaptive_agent_exception_logged(monkeypatch, caplog):
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", "")
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", None, raising=False)
    monkeypatch.setattr(eg, "AdaptivePresetAgent", _FailAgent, raising=False)
    if hasattr(eg.adapt_presets, "_adaptive_agent"):
        delattr(eg.adapt_presets, "_adaptive_agent")
    caplog.set_level("ERROR")
    tracker = _long_tracker()
    eg.adapt_presets(tracker, [{"CPU_LIMIT": "1"}])
    assert "preset adaptation failed" in caplog.text
