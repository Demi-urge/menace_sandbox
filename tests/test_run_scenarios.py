import pytest
from tests.test_menace_master import _setup_mm_stubs
from tests.test_scenario_roi_deltas import _setup_tracker
from pathlib import Path
import json

from dynamic_path_router import resolve_path


def test_run_scenarios_records_all_deltas(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    # ROI returned for each scenario with workflow on/off
    scenario_data = {
        "normal": (3.0, 2.0),
        "concurrency_spike": (1.0, 2.0),
        "hostile_input": (2.5, 2.0),
        "schema_drift": (1.9, 1.7),
        "flaky_upstream": (2.2, 1.9),
    }

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        roi_on, roi_off = scenario_data[scen]
        roi = roi_off if snippet.strip() == "pass" else roi_on
        return {"exit_code": 0}, [(0.0, roi, {})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    out = resolve_path("sandbox_data") / "scenario_deltas.json"
    out.unlink(missing_ok=True)
    tracker_obj, cards, summary = env.run_scenarios(
        ["simple_functions:print_ten"], tracker=rt.ROITracker(filter_outliers=False)
    )

    expected = {scen: roi_on - roi_off for scen, (roi_on, roi_off) in scenario_data.items()}
    baseline_roi = scenario_data["normal"][0]
    expected_raroi_delta = {scen: roi_on - baseline_roi for scen, (roi_on, _) in scenario_data.items()}
    assert set(summary["scenarios"]) == set(expected)
    assert isinstance(cards, dict)
    assert set(cards) == set(expected)
    for scen, delta in expected.items():
        info = summary["scenarios"][scen]
        card = cards[scen]
        assert card.scenario == scen
        assert card.baseline_roi == pytest.approx(baseline_roi)
        assert card.stress_roi == pytest.approx(info["roi"])
        assert card.roi_delta == pytest.approx(info["roi"] - baseline_roi)
        assert card.metrics_delta == info["metrics_delta"]
        assert card.synergy == info["synergy"]
        assert info["roi_delta"] == pytest.approx(delta)
        assert "raroi" in info and isinstance(info["raroi"], float)
        assert "raroi_delta" in info and isinstance(info["raroi_delta"], float)
        if scen == "normal":
            assert info["raroi_delta"] == pytest.approx(0.0)
        assert tracker_obj.get_scenario_roi_delta(scen) == pytest.approx(delta)
        assert scen in tracker_obj.scenario_raroi_delta
        flags = {r["flag"] for r in info["runs"]}
        assert flags == {"on", "off"}
        assert info["target_delta"]["roi"] == pytest.approx(delta)
    worst = min(expected, key=lambda k: expected[k])
    assert summary["worst_scenario"] == worst
    assert tracker_obj.biggest_drop()[0] == worst

    assert summary["status"] == "situationally weak"

    # scorecards returned and included in summary
    assert len(cards) == len(expected)
    assert summary["scorecards"] and set(summary["scorecards"]) == set(cards)
    for scen, card in cards.items():
        assert summary["scorecards"][scen]["roi_delta"] == pytest.approx(card.roi_delta)
        assert summary["scorecards"][scen]["status"] == "situationally weak"
    assert (
        summary["scorecards"]["concurrency_spike"]["recommendation"]
        == "add locking or queueing"
    )
    
    data = json.loads(out.read_text())
    for scen, delta in expected.items():
        assert data[scen]["roi_delta"] == pytest.approx(delta)
        assert "raroi_delta" in data[scen]
        assert data[scen]["worst"] is (scen == worst)
