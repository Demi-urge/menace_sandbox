from __future__ import annotations

import sys
import types

from menace_sandbox.foresight_tracker import ForesightTracker
from sandbox_runner import bootstrap


class DummyROITracker:
    def __init__(self, deltas):
        self._deltas = iter(deltas)
        self.raroi_history = [0.0]
        self.confidence_history = [0.0]
        self.metrics_history = {"synergy_resilience": [0.0]}

    def next_delta(self):
        delta = next(self._deltas)
        self.raroi_history.append(self.raroi_history[-1] + delta / 2.0)
        return delta

    def scenario_degradation(self):  # pragma: no cover - deterministic
        return 0.0


class MiniSelfImprovementEngine:
    def __init__(self, tracker, foresight_tracker):
        self.tracker = tracker
        self.foresight_tracker = foresight_tracker

    def run_cycle(self, workflow_id="wf"):
        delta = self.tracker.next_delta()
        raroi_delta = self.tracker.raroi_history[-1] - self.tracker.raroi_history[-2]
        confidence = self.tracker.confidence_history[-1]
        resilience = self.tracker.metrics_history["synergy_resilience"][-1]
        scenario_deg = self.tracker.scenario_degradation()
        self.foresight_tracker.record_cycle_metrics(
            workflow_id,
            {
                "roi_delta": float(delta),
                "raroi_delta": float(raroi_delta),
                "confidence": float(confidence),
                "resilience": float(resilience),
                "scenario_degradation": float(scenario_deg),
            },
        )


def test_launch_sandbox_runs_cycle(tmp_path, monkeypatch):
    # Stub optional services required by bootstrap
    for name in ("relevancy_radar", "quick_fix_engine"):
        mod = types.ModuleType(name)
        mod.__version__ = "1.0.0"
        monkeypatch.setitem(sys.modules, name, mod)

    # Replace CLI entry point with a minimal self-improvement cycle
    events = []

    def fake_main(_args):
        ft = ForesightTracker(max_cycles=1)
        tracker = DummyROITracker([1.0])
        engine = MiniSelfImprovementEngine(tracker, ft)
        engine.run_cycle()
        events.append(ft.history["wf"][0]["roi_delta"])

    monkeypatch.setattr(bootstrap, "_cli_main", fake_main)

    import importlib

    # ensure we use the real SandboxSettings in case other tests injected stubs
    sys.modules.pop("sandbox_settings", None)
    SandboxSettings = importlib.import_module("sandbox_settings").SandboxSettings

    settings = SandboxSettings(
        sandbox_data_dir=str(tmp_path), menace_env_file=str(tmp_path / ".env")
    )

    # Launch sandbox using the bootstrap helper
    bootstrap.launch_sandbox(settings, verifier=lambda s: None)

    assert events == [1.0]
