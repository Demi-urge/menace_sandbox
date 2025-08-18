import types, sys, importlib.util
from pathlib import Path
import pytest
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

    spec = importlib.util.spec_from_file_location("menace.roi_tracker", ROOT / "roi_tracker.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    sys.modules["menace.roi_tracker"] = rt
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

    tracker_obj, summary = env.run_scenarios(
        ["simple_functions:print_ten"], tracker=rt.ROITracker()
    )

    assert summary["scenarios"]["normal"]["roi_delta"] == pytest.approx(1.0)
    cs_info = summary["scenarios"]["concurrency_spike"]
    assert cs_info["roi_delta"] == pytest.approx(-1.0)
    assert tracker_obj.get_scenario_roi_delta("concurrency_spike") == pytest.approx(-1.0)
    assert summary["worst_scenario"] == "concurrency_spike"

    delta_profit = scenario_data["concurrency_spike"]["on"][1]["profitability"] - scenario_data["normal"]["on"][1]["profitability"]
    assert cs_info["metrics_delta"]["profitability"] == pytest.approx(delta_profit)
    assert cs_info["synergy"]["synergy_roi"] == pytest.approx(-1.0)
    assert cs_info["synergy"]["synergy_profitability"] == pytest.approx(delta_profit)
    flags = {r["flag"] for r in cs_info["runs"]}
    assert flags == {"on", "off"}
    assert cs_info["target_delta"]["roi"] == pytest.approx(-1.0)
