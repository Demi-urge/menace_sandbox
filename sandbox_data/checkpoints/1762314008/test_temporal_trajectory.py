import os
import types
import sys

import pytest

# Stub heavy optional dependencies to keep imports lightweight during tests
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

# deterministic ROI and resilience per scenario
ROI_VALUES = {name: float(-idx) for idx, name in enumerate(SCENARIOS)}
RES_VALUES = ROI_VALUES


def _fake_sim_env(snippet, env_input, container=None):
    name = env_input.get("SCENARIO_NAME")
    return {
        "roi": ROI_VALUES[name],
        "resilience": RES_VALUES[name],
    }


async def _fake_section_worker(snippet, env_input, threshold):
    name = env_input.get("SCENARIO_NAME")
    roi = ROI_VALUES[name]
    metrics = {"resilience": RES_VALUES[name]}
    return {}, [(0.0, roi, metrics)]


def test_temporal_trajectory_profile_and_logging(monkeypatch):
    ft = ForesightTracker()
    logs: list[str] = []

    monkeypatch.setattr(
        "sandbox_runner.environment.simulate_execution_environment", _fake_sim_env
    )
    monkeypatch.setattr(
        "sandbox_runner.environment._section_worker", _fake_section_worker
    )
    monkeypatch.setattr(
        "sandbox_runner.environment.logging.info", lambda msg, *a, **k: logs.append(msg % a)
    )

    simulate_temporal_trajectory("wf", ["simple_functions.print_ten"], foresight_tracker=ft)

    history = list(ft.history["wf"])
    assert len(history) == 5
    # verify that existing metrics are present on each history entry
    existing_metrics = {"roi_delta", "resilience", "scenario_degradation"}
    assert all(existing_metrics <= set(entry) for entry in history)
    # newly tracked metrics should also be included
    assert all("stage" in entry and "stability" in entry for entry in history)
    assert [entry["stage"] for entry in history] == SCENARIOS

    assert len(logs) == 5
    for rec in logs:
        assert all(key in rec for key in ["roi=", "resilience=", "stability=", "degradation="])

    profile = ft.get_temporal_profile("wf")
    assert profile == history

