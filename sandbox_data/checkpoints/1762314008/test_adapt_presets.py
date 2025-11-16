import pickle
import json
from pathlib import Path
import sys
import types

from dynamic_path_router import resolve_path

import menace_sandbox.dynamic_path_router as dpr
import menace_sandbox.environment_generator as eg


def _tracker():
    import menace_sandbox.roi_tracker as rt
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
    policy_file = resolve_path("sandbox_data") / "preset_policy.json"
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
    import menace_sandbox.roi_tracker as rt
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


class _NoopAgent:
    def __init__(self, path=None, *, strategy=None):
        self.policy = type("p", (), {"path": path, "strategy": strategy})()

    def decide(self, tracker):
        return {}

    def save(self):
        pass


class _PredTracker:
    def __init__(self, metrics, preds=None):
        self.metrics_history = metrics
        self.roi_history = []
        self._preds = preds or {}

    def predict_synergy_metric(self, name):
        return self._preds.get(name, 0.0)


def test_synergy_efficiency_downscale(monkeypatch):
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", "")
    monkeypatch.setenv("SANDBOX_ADAPTIVE_AGENT_PATH", "")
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", None, raising=False)
    monkeypatch.setattr(eg.adapt_presets, "_adaptive_agent", None, raising=False)
    monkeypatch.setattr(eg, "AdaptivePresetAgent", _NoopAgent, raising=False)

    metrics = {
        "security_score": [70, 70, 70],
        "synergy_efficiency": [0.02, 0.02, 0.02],
    }
    tracker = _PredTracker(metrics, {"efficiency": 0.2})
    presets = [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "256Mi"}]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["CPU_LIMIT"] == "0.5"
    assert new[0]["MEMORY_LIMIT"] == "128Mi"


def test_synergy_efficiency_upscale(monkeypatch):
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", "")
    monkeypatch.setenv("SANDBOX_ADAPTIVE_AGENT_PATH", "")
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", None, raising=False)
    monkeypatch.setattr(eg.adapt_presets, "_adaptive_agent", None, raising=False)
    monkeypatch.setattr(eg, "AdaptivePresetAgent", _NoopAgent, raising=False)

    metrics = {
        "security_score": [70, 70, 70],
        "synergy_efficiency": [-0.02, -0.02, -0.02],
    }
    tracker = _PredTracker(metrics, {"efficiency": -0.2})
    presets = [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "256Mi"}]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["CPU_LIMIT"] == "2"
    assert new[0]["MEMORY_LIMIT"] == "512Mi"


def test_synergy_resilience_bandwidth_up(monkeypatch):
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", "")
    monkeypatch.setenv("SANDBOX_ADAPTIVE_AGENT_PATH", "")
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", None, raising=False)
    monkeypatch.setattr(eg.adapt_presets, "_adaptive_agent", None, raising=False)
    monkeypatch.setattr(eg, "AdaptivePresetAgent", _NoopAgent, raising=False)

    metrics = {
        "security_score": [70, 70, 70],
        "synergy_resilience": [0.02, 0.02, 0.02],
    }
    tracker = _PredTracker(metrics, {"resilience": 0.2})
    presets = [{
        "BANDWIDTH_LIMIT": "5Mbps",
        "MAX_BANDWIDTH": "5Mbps",
        "MIN_BANDWIDTH": "1Mbps",
    }]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["BANDWIDTH_LIMIT"] == "10Mbps"
    assert new[0]["MAX_BANDWIDTH"] == "10Mbps"
    assert new[0]["MIN_BANDWIDTH"] == "5Mbps"


def test_synergy_resilience_bandwidth_down(monkeypatch):
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", "")
    monkeypatch.setenv("SANDBOX_ADAPTIVE_AGENT_PATH", "")
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", None, raising=False)
    monkeypatch.setattr(eg.adapt_presets, "_adaptive_agent", None, raising=False)
    monkeypatch.setattr(eg, "AdaptivePresetAgent", _NoopAgent, raising=False)

    metrics = {
        "security_score": [70, 70, 70],
        "synergy_resilience": [-0.02, -0.02, -0.02],
    }
    tracker = _PredTracker(metrics, {"resilience": -0.2})
    presets = [{
        "BANDWIDTH_LIMIT": "10Mbps",
        "MAX_BANDWIDTH": "10Mbps",
        "MIN_BANDWIDTH": "5Mbps",
    }]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["BANDWIDTH_LIMIT"] == "5Mbps"
    assert new[0]["MAX_BANDWIDTH"] == "5Mbps"
    assert new[0]["MIN_BANDWIDTH"] == "1Mbps"


def test_generate_presets_from_history_calls_adapt(monkeypatch, tmp_path):
    history = tmp_path / "roi_history.json"
    data = {
        "roi_history": [0.0, 0.1],
        "metrics_history": {"security_score": [85, 85]},
    }
    history.write_text(json.dumps(data))
    called = {}

    def fake_generate(n=None, *, agent=None, tracker=None):
        return [{"CPU_LIMIT": "1", "THREAT_INTENSITY": 30}]

    def fake_adapt(tracker, presets):
        called["roi"] = tracker.roi_history
        presets[0]["ADAPTED"] = True
        return presets

    monkeypatch.setattr(eg, "generate_presets", fake_generate)
    monkeypatch.setattr(eg, "adapt_presets", fake_adapt)

    out = eg.generate_presets_from_history(str(tmp_path), 1)
    assert called["roi"] == [0.0, 0.1]
    assert out[0].get("ADAPTED")


def test_generate_presets_from_history_missing(monkeypatch, tmp_path):
    calls = []

    def fake_generate(n=None, *, agent=None, tracker=None):
        return [{"CPU_LIMIT": "1"}]

    def fake_adapt(tracker, presets):
        calls.append(True)
        return presets

    monkeypatch.setattr(eg, "generate_presets", fake_generate)
    monkeypatch.setattr(eg, "adapt_presets", fake_adapt)

    out = eg.generate_presets_from_history(str(tmp_path), 1)
    assert not calls
    assert out == [{"CPU_LIMIT": "1"}]


def test_generate_presets_from_history_resolves_data_dir(monkeypatch, tmp_path):
    dpr.resolve_path("environment_generator.py")  # path-ignore
    alt_root = tmp_path / "relocated"
    data_dir = alt_root / "nested" / "sandbox_data"
    data_dir.mkdir(parents=True)
    history = data_dir / "roi_history.json"
    history.write_text(
        json.dumps({"roi_history": [0.0], "metrics_history": {"security_score": [85]}})
    )

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(alt_root))
    dpr.clear_cache()

    class DummyTracker:
        metrics_history = {"security_score": [85]}
        roi_history = [0.0]

        def load_history(self, path):
            self.loaded = Path(path)

    module = types.SimpleNamespace(ROITracker=DummyTracker)
    monkeypatch.setitem(sys.modules, "menace_sandbox.roi_tracker", module)

    monkeypatch.setattr(eg, "generate_presets", lambda n=None: [{"CPU_LIMIT": "1"}])
    called = {}

    def fake_adapt(tracker, presets):
        called["tracker"] = tracker
        return presets

    monkeypatch.setattr(eg, "adapt_presets", fake_adapt)

    out = eg.generate_presets_from_history(Path("nested") / "sandbox_data", 1)
    assert called["tracker"].loaded == history
    assert out == [{"CPU_LIMIT": "1"}]
