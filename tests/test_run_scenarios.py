import pytest
from tests.test_menace_master import _setup_mm_stubs
from tests.test_scenario_roi_deltas import _setup_tracker
from pathlib import Path
import json


def test_run_scenarios_records_all_deltas(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    # ROI returned for each scenario
    scenario_data = {
        "normal": 3.0,
        "concurrency_spike": 1.0,
        "hostile_input": 2.5,
        "schema_drift": 1.9,
        "flaky_upstream": 2.2,
    }

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        roi = scenario_data[scen]
        return {"exit_code": 0}, [(0.0, roi, {})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    out = Path("sandbox_data/scenario_deltas.json")
    out.unlink(missing_ok=True)
    summary = env.run_scenarios(["simple_functions:print_ten"], tracker=rt.ROITracker())

    expected = {scen: roi - scenario_data["normal"] for scen, roi in scenario_data.items()}
    assert set(summary["scenarios"]) == set(expected)
    for scen, delta in expected.items():
        info = summary["scenarios"][scen]
        assert info["roi_delta"] == pytest.approx(delta)
        flags = {r["flag"] for r in info["runs"]}
        assert flags == {"on", "off"}
        assert info["target_delta"]["roi"] == pytest.approx(0.0)
    worst = min(expected, key=lambda k: expected[k])
    assert summary["worst_scenario"] == worst

    data = json.loads(out.read_text())
    for scen, delta in expected.items():
        assert data[scen]["roi_delta"] == pytest.approx(delta)
        assert data[scen]["worst"] is (scen == worst)
