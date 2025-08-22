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


def _fake_run_scenarios(workflow, tracker=None, presets=None, foresight_tracker=None):
    summary = {
        "scenarios": {
            name: {"roi": float(i), "metrics": {"resilience": float(i)}}
            for i, name in enumerate(SCENARIOS)
        }
    }
    return object(), {}, summary


def _fake_presets():
    return [{"SCENARIO_NAME": name} for name in SCENARIOS]


def test_simulate_temporal_trajectory_updates_history(monkeypatch):
    monkeypatch.setattr(
        "sandbox_runner.environment.run_scenarios", _fake_run_scenarios
    )
    monkeypatch.setattr(
        "sandbox_runner.environment.temporal_trajectory_presets", _fake_presets
    )
    ft = ForesightTracker()
    simulate_temporal_trajectory(["simple_functions.print_ten"], foresight_tracker=ft)
    history = list(ft.history["0"])
    assert len(history) == 5
    expected_keys = {"roi_delta", "resilience", "stability", "scenario_degradation"}
    assert all(expected_keys <= set(entry) for entry in history)
    assert [entry["roi_delta"] for entry in history] == [0.0, 1.0, 2.0, 3.0, 4.0]
