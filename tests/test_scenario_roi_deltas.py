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


def test_scenario_roi_delta_persistence(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    rt = _setup_tracker(monkeypatch)
    import sandbox_runner.environment as env

    scenario_data = {
        "normal": (1.0, 2.0, {"profitability": 1.0}, {"profitability": 2.0}),
        "concurrency_spike": (1.0, 0.0, {"profitability": 1.0}, {"profitability": 0.0}),
        "hostile_input": (1.0, 1.2, {"profitability": 1.0}, {"profitability": 1.2}),
        "schema_drift": (1.0, 1.2, {"profitability": 1.0}, {"profitability": 1.2}),
        "flaky_upstream": (1.0, 1.2, {"profitability": 1.0}, {"profitability": 1.2}),
    }
    call_counts = {}

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get("SCENARIO_NAME")
        count = call_counts.get(scen, 0)
        call_counts[scen] = count + 1
        base, with_, base_metrics, with_metrics = scenario_data[scen]
        if count == 0:
            roi, metrics = base, base_metrics
        else:
            roi, metrics = with_, with_metrics
        return {"exit_code": 0}, [(0.0, roi, metrics)]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    tracker = env.run_scenarios([], tracker=rt.ROITracker())
    assert tracker.scenario_roi_deltas["normal"] == pytest.approx(1.0)
    assert tracker.scenario_roi_deltas["concurrency_spike"] == pytest.approx(-1.0)
    assert tracker.worst_scenario == "concurrency_spike"

    path_json = tmp_path / "roi.json"
    tracker.save_history(str(path_json))
    t2 = rt.ROITracker()
    t2.load_history(str(path_json))
    assert t2.scenario_roi_deltas == tracker.scenario_roi_deltas
    assert t2.worst_scenario == tracker.worst_scenario
    assert t2.get_scenario_roi_delta("normal") == pytest.approx(1.0)

    path_db = tmp_path / "roi.db"
    tracker.save_history(str(path_db))
    t3 = rt.ROITracker()
    t3.load_history(str(path_db))
    assert t3.scenario_roi_deltas == tracker.scenario_roi_deltas
    assert t3.worst_scenario == tracker.worst_scenario
