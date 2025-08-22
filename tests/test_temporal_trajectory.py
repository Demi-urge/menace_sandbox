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
    from sandbox_runner.environment import (
        simulate_temporal_trajectory,
        temporal_presets,
    )
except SystemExit:
    simulate_temporal_trajectory = None  # type: ignore
    temporal_presets = None  # type: ignore
    pytest.skip(
        "sandbox environment requires missing system packages",
        allow_module_level=True,
    )

from foresight_tracker import ForesightTracker


SCENARIOS = [p["SCENARIO_NAME"] for p in temporal_presets()]

# record the order in which scenarios are executed
stage_calls: list[str] = []


def _fake_sim_env(snippet, env_input, container=None):
    stage_calls.append(env_input.get("SCENARIO_NAME"))
    return {}


async def _fake_section_worker(snippet, env_input, threshold):
    idx = len(stage_calls) - 1
    roi = float(-idx)
    metrics = {"resilience": roi}
    return {}, [(0.0, roi, metrics)]


def test_simulate_temporal_trajectory_order_and_history(monkeypatch):
    stage_calls.clear()
    monkeypatch.setattr(
        "sandbox_runner.environment.simulate_execution_environment",
        _fake_sim_env,
    )
    monkeypatch.setattr(
        "sandbox_runner.environment._section_worker", _fake_section_worker
    )

    ft = ForesightTracker()
    simulate_temporal_trajectory("0", ["simple_functions.print_ten"], foresight_tracker=ft)

    assert stage_calls == SCENARIOS

    history = list(ft.history["0"])
    assert len(history) == len(SCENARIOS)
    expected_keys = {"roi_delta", "resilience", "scenario_degradation"}
    assert all(expected_keys <= set(entry) for entry in history)
    assert [entry["roi_delta"] for entry in history] == [0.0, -1.0, -2.0, -3.0, -4.0]
    assert [entry["resilience"] for entry in history] == [0.0, -1.0, -2.0, -3.0, -4.0]
    assert [entry["scenario_degradation"] for entry in history] == [0.0, -1.0, -2.0, -3.0, -4.0]
