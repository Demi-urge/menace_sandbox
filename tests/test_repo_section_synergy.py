import sys
import types
import pytest
from tests.test_menace_master import _setup_mm_stubs
from tests.test_sandbox_section_simulations import _stub_module


class DummyBot:
    def __init__(self, *a, **k):
        pass


class DummyTracker:
    def __init__(self, *a, **k):
        self.calls = []
        self.roi_history = []
        self.metrics_history = {}
        self.synergy_metrics_history = {}
        self.scenario_synergy = {}

    def register_metrics(self, *names):
        for n in names:
            target = (
                self.synergy_metrics_history if str(n).startswith("synergy_") else self.metrics_history
            )
            target.setdefault(str(n), [])

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.calls.append({"modules": modules, "metrics": metrics})
        self.roi_history.append(curr)
        for k, v in (metrics or {}).items():
            hist = self.synergy_metrics_history if k.startswith("synergy_") else self.metrics_history
            hist.setdefault(k, []).append(v)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, *a, **k):
        pass

    def record_metric_prediction(self, *a, **k):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        pass

    def get_scenario_synergy(self, name):
        return self.scenario_synergy.get(name, [])


def test_run_repo_section_synergy(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")  # path-ignore

    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummyBot)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "networkx")
    sqla = types.ModuleType("sqlalchemy")
    sqla_engine = types.ModuleType("sqlalchemy.engine")
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqla)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqla_engine)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: {"m.py": {"sec": ["pass"]}},  # path-ignore
        raising=False,
    )

    calls = []

    async def fake_worker(snippet, env_input, threshold):
        calls.append(env_input.get("SCENARIO_NAME"))
        roi = float(len(calls))
        metrics = {"profitability": roi}
        return {"exit_code": 0}, [(0.0, roi, metrics)]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    presets = [{"SCENARIO_NAME": "dev"}, {"SCENARIO_NAME": "prod"}]

    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], env_presets=presets
    )

    dev = tracker.scenario_synergy["dev"][0]
    prod = tracker.scenario_synergy["prod"][0]
    assert dev["synergy_roi"] == pytest.approx(2.0)
    assert dev["synergy_profitability"] == pytest.approx(2.0)
    assert prod["synergy_roi"] == pytest.approx(2.0)
    assert prod["synergy_profitability"] == pytest.approx(2.0)

    section_calls = [c for c in tracker.calls if ":" in c["modules"][0]]
    combined_calls = [c for c in tracker.calls if ":" not in c["modules"][0]]
    assert section_calls and combined_calls
    assert not any("synergy_roi" in c["metrics"] for c in section_calls)
    assert any("synergy_roi" in c["metrics"] for c in combined_calls)
    for call in combined_calls:
        if "synergy_roi" in call["metrics"]:
            assert "synergy_profitability" in call["metrics"]


def test_repo_section_synergy_cli(tmp_path, capsys):
    import importlib
    import types, sys
    from pathlib import Path
    sk_mod = types.ModuleType("sklearn")
    lin_mod = types.ModuleType("sklearn.linear_model")
    lin_mod.LinearRegression = lambda *a, **k: types.SimpleNamespace(coef_=[0, 0, 0], fit=lambda X, y: None, predict=lambda X: [0] * len(X))
    pre_mod = types.ModuleType("sklearn.preprocessing")
    pre_mod.PolynomialFeatures = lambda degree=2: types.SimpleNamespace(fit_transform=lambda x: x)
    sk_mod.linear_model = lin_mod
    sk_mod.preprocessing = pre_mod
    sys.modules.setdefault("sklearn", sk_mod)
    sys.modules.setdefault("sklearn.linear_model", lin_mod)
    sys.modules.setdefault("sklearn.preprocessing", pre_mod)

    ROOT = Path(__file__).resolve().parents[1]
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
    sys.modules.setdefault("menace.logging_utils", log_mod)

    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("menace.roi_tracker", Path("roi_tracker.py"))  # path-ignore
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    sys.modules.setdefault("menace.roi_tracker", rt)

    cli = importlib.import_module("sandbox_runner.cli")

    t1 = rt.ROITracker()
    t1.scenario_synergy = {"dev": [{"synergy_roi": 0.4}], "prod": [{"synergy_roi": 0.2}]}
    f1 = tmp_path / "a.json"
    t1.save_history(str(f1))

    t2 = rt.ROITracker()
    t2.scenario_synergy = {"dev": [{"synergy_roi": 0.1}], "prod": [{"synergy_roi": 0.5}]}
    f2 = tmp_path / "b.json"
    t2.save_history(str(f2))

    cli.rank_scenario_synergy([str(f1), str(f2)])
    out = capsys.readouterr().out.strip().splitlines()
    assert out and all("roi=" in line for line in out)
