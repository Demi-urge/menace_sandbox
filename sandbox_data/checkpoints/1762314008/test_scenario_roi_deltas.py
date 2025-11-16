import types, sys, importlib.util
from pathlib import Path
import json
import pytest
from dynamic_path_router import resolve_path
from tests.test_menace_master import _setup_mm_stubs

ROOT = Path(__file__).resolve().parents[1]

def _setup_tracker(monkeypatch):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = [str(ROOT)]
    sys.modules.setdefault("menace", menace_pkg)

    log_mod = types.ModuleType("menace.logging_utils")
    log_mod.get_logger = lambda *a, **k: types.SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    log_mod.log_record = lambda *a, **k: None
    sys.modules.setdefault("menace.logging_utils", log_mod)

    ta_mod = types.ModuleType("menace.truth_adapter")
    class TruthAdapter:
        def __init__(self):
            self.metadata = {}
    ta_mod.TruthAdapter = TruthAdapter
    sys.modules.setdefault("menace.truth_adapter", ta_mod)

    sk_mod = types.ModuleType("sklearn")
    lin_mod = types.ModuleType("sklearn.linear_model")
    lin_mod.LinearRegression = lambda *a, **k: types.SimpleNamespace(coef_=[0,0,0], fit=lambda X,y: None, predict=lambda X: [0]*len(X))
    pre_mod = types.ModuleType("sklearn.preprocessing")
    pre_mod.PolynomialFeatures = lambda degree=2: types.SimpleNamespace(fit_transform=lambda x: x)
    sk_mod.linear_model = lin_mod
    sk_mod.preprocessing = pre_mod
    sys.modules.setdefault("sklearn", sk_mod)
    sys.modules.setdefault("sklearn.linear_model", lin_mod)
    sys.modules.setdefault("sklearn.preprocessing", pre_mod)

    cfg_mod = types.ModuleType("menace.config_loader")
    cfg_mod.get_impact_severity = lambda *a, **k: 0.0
    cfg_mod.impact_severity_map = {}
    sys.modules.setdefault("menace.config_loader", cfg_mod)

    bb_mod = types.ModuleType("menace.borderline_bucket")
    class BorderlineBucket:
        pass
    bb_mod.BorderlineBucket = BorderlineBucket
    sys.modules.setdefault("menace.borderline_bucket", bb_mod)

    spec = importlib.util.spec_from_file_location("menace.roi_tracker", ROOT / "roi_tracker.py")  # path-ignore
    rt = importlib.util.module_from_spec(spec)
    sys.modules["menace.roi_tracker"] = rt
    spec.loader.exec_module(rt)
    return rt


def test_scenario_roi_deltas_and_synergy(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    scenario_data = {
        "normal": {
            "on": (1.0, {"profitability": 1.0}),
            "off": (0.0, {"profitability": 0.0}),
        },
        "concurrency_spike": {
            "on": (0.0, {"profitability": 0.0}),
            "off": (1.0, {"profitability": 1.0}),
        },
        "hostile_input": {
            "on": (1.2, {"profitability": 1.2}),
            "off": (1.2, {"profitability": 1.2}),
        },
        "schema_drift": {
            "on": (1.2, {"profitability": 1.2}),
            "off": (1.2, {"profitability": 1.2}),
        },
        "flaky_upstream": {
            "on": (1.2, {"profitability": 1.2}),
            "off": (1.2, {"profitability": 1.2}),
        },
    }

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        mode = "off" if snippet.strip() == "pass" else "on"
        roi, metrics = scenario_data[scen][mode]
        return {"exit_code": 0}, [(0.0, roi, metrics)]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    tracker_obj, cards, summary = env.run_scenarios(
        ["simple_functions:print_ten"], tracker=rt.ROITracker(filter_outliers=False)
    )

    assert summary["scenarios"]["normal"]["roi_delta"] == pytest.approx(1.0)
    cs_info = summary["scenarios"]["concurrency_spike"]
    assert cs_info["roi_delta"] == pytest.approx(-1.0)
    assert "raroi" in cs_info and "raroi_delta" in cs_info
    assert tracker_obj.get_scenario_roi_delta("concurrency_spike") == pytest.approx(-1.0)
    assert "concurrency_spike" in tracker_obj.scenario_raroi_delta
    assert summary["worst_scenario"] == "concurrency_spike"
    assert tracker_obj.biggest_drop()[0] == "concurrency_spike"

    delta_profit = (
        scenario_data["concurrency_spike"]["on"][1]["profitability"]
        - scenario_data["normal"]["on"][1]["profitability"]
    )
    assert cs_info["metrics_delta"]["profitability"] == pytest.approx(delta_profit)
    assert cs_info["synergy"]["synergy_roi"] == pytest.approx(-1.0)
    assert cs_info["synergy"]["synergy_profitability"] == pytest.approx(delta_profit)
    assert (
        tracker_obj.scenario_metrics_delta["concurrency_spike"]["profitability"]
        == pytest.approx(delta_profit)
    )
    assert (
        tracker_obj.scenario_synergy_delta["concurrency_spike"]["synergy_profitability"]
        == pytest.approx(delta_profit)
    )
    flags = {r["flag"] for r in cs_info["runs"]}
    assert flags == {"on", "off"}
    assert cs_info["target_delta"]["roi"] == pytest.approx(-1.0)
    assert summary["status"] == "situationally weak"
    assert summary["scorecards"]["concurrency_spike"]["status"] == "situationally weak"
    assert (
        summary["scorecards"]["concurrency_spike"]["recommendation"]
        == "add locking or queueing"
    )


def test_generate_scorecard(monkeypatch):
    """Each scenario should appear in the persisted workflow scorecard."""

    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    scenario_data = {
        "normal": (1.0, 0.0),
        "concurrency_spike": (0.0, 1.0),
        "hostile_input": (1.2, 1.2),
        "schema_drift": (1.2, 1.2),
        "flaky_upstream": (1.2, 1.2),
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

    card = env.generate_scorecard(["simple_functions:print_ten"], summary)
    assert set(card["scenarios"]) == set(scenario_data)
    for scen, info in summary["scenarios"].items():
        assert card["scenarios"][scen]["roi_delta"] == pytest.approx(
            info["roi_delta"]
        )
    assert card["status"] == "situationally weak"
    assert (
        card["scenarios"]["concurrency_spike"]["recommendation"]
        == "add locking or queueing"
    )

    wf_id = card["workflow_id"]
    out_path = resolve_path("sandbox_data") / f"scorecard_{wf_id}.json"
    assert out_path.exists()
    persisted = json.loads(out_path.read_text())
    assert persisted == card
