import os
import types
import sys

import pytest

# Stub heavy optional dependencies to keep import light
stub_st = types.ModuleType("sentence_transformers")
sys.modules.setdefault("sentence_transformers", stub_st)
stub_faiss = types.ModuleType("faiss")
sys.modules.setdefault("faiss", stub_faiss)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

try:
    from sandbox_runner.environment import simulate_temporal_trajectory
except SystemExit:
    simulate_temporal_trajectory = None  # type: ignore
    pytest.skip("sandbox environment requires missing system packages", allow_module_level=True)

from foresight_tracker import ForesightTracker


SCENARIOS = [
    "baseline",
    "latency_spike",
    "io_cpu_strain",
    "schema_drift",
    "chaotic_failure",
]

# record the order in which scenarios are executed
stage_calls: list[str] = []


class _DummyTracker:
    def __init__(self) -> None:
        self.roi_history = []


def _fake_run_scenarios(workflow, tracker=None, presets=None, foresight_tracker=None):
    if tracker is None:
        tracker = _DummyTracker()
    name = presets[0]["SCENARIO_NAME"]
    stage_calls.append(name)
    idx = SCENARIOS.index(name)
    roi = float(idx)
    tracker.roi_history.append(roi)
    summary = {"scenarios": {name: {"roi": roi, "metrics": {"resilience": roi}}}}
    return tracker, {}, summary


def _fake_presets():
    return [{"SCENARIO_NAME": name} for name in SCENARIOS]


def test_simulate_temporal_trajectory_order_and_history(monkeypatch):
    stage_calls.clear()
    monkeypatch.setattr(
        "sandbox_runner.environment.run_scenarios", _fake_run_scenarios
    )
    monkeypatch.setattr(
        "sandbox_runner.environment.temporal_trajectory_presets", _fake_presets
    )
    class _FakeWorkflowDB:
        def __init__(self, *a, **k):
            self.conn = self

        def execute(self, *a, **k):
            return self

        def fetchone(self):
            return {"workflow": "simple_functions.print_ten", "task_sequence": ""}

    stub_db_mod = types.ModuleType("menace.task_handoff_bot")
    stub_db_mod.WorkflowDB = _FakeWorkflowDB  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menace.task_handoff_bot", stub_db_mod)

    ft = ForesightTracker()
    simulate_temporal_trajectory(0, foresight_tracker=ft)

    # stages should execute in the expected order
    assert stage_calls == SCENARIOS

    history = list(ft.history["0"])
    assert len(history) == 5
    expected_keys = {"roi_delta", "resilience", "stability", "scenario_degradation"}
    assert all(expected_keys <= set(entry) for entry in history)
    assert [entry["roi_delta"] for entry in history] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [entry["resilience"] for entry in history] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [entry["scenario_degradation"] for entry in history] == [0.0, -1.0, -2.0, -3.0, -4.0]
