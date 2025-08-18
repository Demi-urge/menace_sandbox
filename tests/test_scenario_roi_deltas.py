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
        "normal": (1.0, {"profitability": 1.0}),
        "concurrency_spike": (0.0, {"profitability": 0.0}),
        "hostile_input": (1.2, {"profitability": 1.2}),
        "schema_drift": (1.2, {"profitability": 1.2}),
        "flaky_upstream": (1.2, {"profitability": 1.2}),
    }

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        roi, metrics = scenario_data[scen]
        return {"exit_code": 0}, [(0.0, roi, metrics)]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    summary = env.run_scenarios([], tracker=rt.ROITracker())

    assert summary["scenarios"]["normal"]["roi_delta"] == pytest.approx(0.0)
    assert summary["scenarios"]["concurrency_spike"]["roi_delta"] == pytest.approx(-1.0)
    assert summary["worst_scenario"] == "concurrency_spike"

    delta_profit = scenario_data["concurrency_spike"][1]["profitability"] - scenario_data["normal"][1]["profitability"]
    scen_info = summary["scenarios"]["concurrency_spike"]
    assert scen_info["metrics_delta"]["profitability"] == pytest.approx(delta_profit)
    assert scen_info["synergy"]["synergy_roi"] == pytest.approx(-1.0)
    assert scen_info["synergy"]["synergy_profitability"] == pytest.approx(delta_profit)
