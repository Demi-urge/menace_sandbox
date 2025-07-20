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
