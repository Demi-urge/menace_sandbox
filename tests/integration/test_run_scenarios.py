import json
from pathlib import Path
import pytest
from tests.test_menace_master import _setup_mm_stubs
from tests.test_scenario_roi_deltas import _setup_tracker


def test_run_scenarios_all_paths(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    hostile = json.loads((fixtures / "hostile_input.json").read_text())
    flaky = json.loads((fixtures / "flaky_upstream.json").read_text())

    scenario_data = {
        "normal": 3.0,
        "concurrency_spike": 1.0,
        "hostile_input": hostile["expected_roi"],
        "schema_drift": 1.9,
        "flaky_upstream": flaky["expected_roi"],
    }

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        roi = scenario_data[scen]
        if "print_ten" not in snippet:
            roi -= 0.5
        return {"exit_code": 0}, [(0.0, roi, {})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    tracker_obj, cards, summary = env.run_scenarios(
        ["simple_functions:print_ten"], tracker=rt.ROITracker(filter_outliers=False)
    )

    assert set(summary["scenarios"]) == set(scenario_data)

    for scen in scenario_data:
        info = summary["scenarios"][scen]
        assert isinstance(info["roi_delta"], float)
        assert "raroi" in info and "raroi_delta" in info
        assert {r["flag"] for r in info["runs"]} == {"on", "off"}
        assert scen in tracker_obj.scenario_raroi_delta

    assert summary["worst_scenario"] in scenario_data
    assert len(cards) == len(scenario_data)

    hi_info = summary["scenarios"]["hostile_input"]
    assert "roi" in hi_info["target_delta"]
