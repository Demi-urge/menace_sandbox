import json

import pytest
from tests.test_menace_master import _setup_mm_stubs
from tests.test_scenario_roi_deltas import _setup_tracker
from dynamic_path_router import resolve_path


def test_run_scenarios_all_paths(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    hostile = json.loads(resolve_path("tests/fixtures/hostile_input.json").read_text())
    flaky = json.loads(resolve_path("tests/fixtures/flaky_upstream.json").read_text())

    scenario_data = {
        "normal": (3.0, 2.0),
        "concurrency_spike": (1.0, 2.0),
        "hostile_input": (hostile["expected_roi"], hostile["expected_roi"] - 0.5),
        "schema_drift": (1.9, 1.7),
        "flaky_upstream": (flaky["expected_roi"], flaky["expected_roi"] - 0.3),
    }

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        roi_on, roi_off = scenario_data[scen]
        roi = roi_off if snippet.strip() == "pass" else roi_on
        return {"exit_code": 0}, [(0.0, roi, {})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    tracker_obj, cards, summary = env.run_scenarios(
        ["simple_functions:print_ten"], tracker=rt.ROITracker(filter_outliers=False)
    )

    assert set(summary["scenarios"]) == set(scenario_data)
    assert isinstance(cards, dict)
    assert set(cards) == set(scenario_data)

    baseline = scenario_data["normal"][0]
    expected_delta = {s: on - off for s, (on, off) in scenario_data.items()}
    for scen, (roi_on, _) in scenario_data.items():
        info = summary["scenarios"][scen]
        card = cards[scen]
        assert info["roi_delta"] == pytest.approx(expected_delta[scen])
        assert "raroi" in info and "raroi_delta" in info
        assert {r["flag"] for r in info["runs"]} == {"on", "off"}
        assert scen in tracker_obj.scenario_raroi_delta
        assert card.baseline_roi == pytest.approx(baseline)
        assert card.stress_roi == pytest.approx(roi_on)
        exported = summary["scorecards"][scen]
        assert exported["roi_delta"] == pytest.approx(card.roi_delta)
        assert set(exported) == {
            "scenario",
            "baseline_roi",
            "stress_roi",
            "roi_delta",
            "metrics_delta",
            "synergy",
            "recommendation",
            "status",
        }
        assert exported["status"] == "situationally weak"

    assert summary["worst_scenario"] in scenario_data
    assert len(cards) == len(scenario_data)
    assert summary["scorecards"] and set(summary["scorecards"]) == set(cards)
    assert summary["status"] == "situationally weak"
    assert (
        summary["scorecards"]["concurrency_spike"]["recommendation"]
        == "add locking or queueing"
    )
