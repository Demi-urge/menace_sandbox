import os
import sys
import types
import argparse
from pathlib import Path
import sandbox_runner.cli as cli

os.environ["MENACE_LIGHT_IMPORTS"] = "1"

class DummyTracker:
    def __init__(self):
        self.metrics_history = {}
        self.roi_history = []
        self.module_deltas = {}

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.roi_history.append(curr)
        if metrics:
            for k, v in metrics.items():
                self.metrics_history.setdefault(k, []).append(v)
        for k in self.metrics_history:
            if not metrics or k not in metrics:
                self.metrics_history[k].append(self.metrics_history[k][-1])

    def diminishing(self):
        return 0.01

    def rankings(self):
        return [("m", 0.1, 0.1)]

class CaptureImprover:
    def __init__(self, *a, **k):
        self.tracker = None
        self.states = []

    def _policy_state(self):
        t = self.tracker
        val = 0.0
        if t:
            val = t.metrics_history.get("synergy_roi", [0.0])[-1]
        self.states.append(val)
        return (0,) * 15 + (
            int(round(val * 10)),
            0,
            0,
            0,
        )

    def run_cycle(self):
        self._policy_state()
        return types.SimpleNamespace(roi=types.SimpleNamespace(roi=0.0))

class DummyAgent:
    calls: list[float] = []

    def __init__(self, path):
        self.path = path

    def decide(self, tracker):
        DummyAgent.calls.append(
            tracker.metrics_history.get("synergy_roi", [0.0])[-1]
        )
        return {"cpu": 1}

    def save(self):
        pass

class SingleVA:
    calls = 0
    active = False

    def __init__(self, *a, **k):
        pass

    def ask(self, msgs):
        if SingleVA.active:
            raise RuntimeError("busy")
        SingleVA.active = True
        SingleVA.calls += 1
        SingleVA.active = False
        return {"choices": [{"message": {"content": ""}}]}

def test_full_autonomous_synergy_rl(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_CYCLES", "3")
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", str(tmp_path / "p.pkl"))

    prl = types.ModuleType("menace_sandbox.preset_rl_agent")
    prl.PresetRLAgent = DummyAgent
    sys.modules["menace_sandbox.preset_rl_agent"] = prl

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.environment_generator",
        str(Path(__file__).resolve().parents[1] / "environment_generator.py"),  # path-ignore
    )
    eg = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.environment_generator"] = eg
    sys.modules.setdefault("environment_generator", eg)
    spec.loader.exec_module(eg)

    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"CPU_LIMIT": "1"}])

    values = iter([0.05, 0.06, 0.07])

    def fake_cycle(ctx, section, snippet, tracker, scenario=None):
        ctx.improver.tracker = tracker
        val = next(values)
        tracker.update(0.0, val, metrics={"security_score": 70, "synergy_roi": val})

    improver = CaptureImprover()

    trackers = []

    def fake_capture(preset, args):
        tracker = DummyTracker()
        for _ in range(3):
            val = next(values)
            tracker.update(0.0, val, metrics={"security_score": 70, "synergy_roi": val})
            improver.tracker = tracker
            improver.run_cycle()
        SingleVA().ask([{"content": "x"}])
        eg.adapt_presets(tracker, [preset])
        trackers.append(tracker)
        return tracker

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    args = argparse.Namespace(
        sandbox_data_dir=str(tmp_path),
        preset_count=1,
        max_iterations=1,
        dashboard_port=None,
        roi_cycles=1,
        synergy_cycles=1,
    )

    cli.full_autonomous_run(args)

    assert improver.states[-1] == 0.07
    agent = prl.PresetRLAgent
    assert agent.calls[-1] == 0.07
    assert SingleVA.calls <= 1
    assert trackers
