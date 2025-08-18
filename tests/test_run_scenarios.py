import pytest
from tests.test_menace_master import _setup_mm_stubs
from tests.test_scenario_roi_deltas import _setup_tracker


def test_run_scenarios_records_all_deltas(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    # base ROI and ROI with workflow for each scenario
    scenario_data = {
        "normal": (2.0, 3.0),
        "concurrency_spike": (2.0, 1.0),
        "hostile_input": (2.0, 2.5),
        "schema_drift": (2.0, 1.9),
        "flaky_upstream": (2.0, 2.2),
    }
    call_counts = {}

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        count = call_counts.get(scen, 0)
        call_counts[scen] = count + 1
        base, with_ = scenario_data[scen]
        roi = base if count == 0 else with_
        return {"exit_code": 0}, [(0.0, roi, {})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    tracker = env.run_scenarios(["simple_functions:print_ten"], tracker=rt.ROITracker())

    expected = {scen: with_ - base for scen, (base, with_) in scenario_data.items()}
    assert set(tracker.scenario_roi_deltas) == set(expected)
    for scen, delta in expected.items():
        assert tracker.scenario_roi_deltas[scen] == pytest.approx(delta)
    worst = min(expected, key=lambda k: expected[k])
    assert tracker.worst_scenario == worst
