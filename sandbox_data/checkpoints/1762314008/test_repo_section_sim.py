import types
import sys
import asyncio
from dynamic_path_router import resolve_dir, resolve_path


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg, _, sub = name.partition(".")
    pkg_mod = sys.modules.get(pkg)
    if pkg_mod and sub:
        setattr(pkg_mod, sub, mod)
    return mod


class DummyBot:
    def __init__(self, *a, **k):
        pass


class DummySandbox:
    def __init__(self, *a, **k):
        pass

    def analyse_and_fix(self):
        pass


class _ROITracker:
    def __init__(self, *a, **k):
        self.calls = []
        self.roi_history = []
        self.synergy_history = []

    def update(self, prev_roi, roi, modules=None, resources=None, metrics=None):
        self.calls.append(modules)
        self.roi_history.append(roi)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, predicted, actual, *a, **k):
        pass

    def record_metric_prediction(self, metric, predicted, actual):
        pass

    def predict_all_metrics(self, manager, features):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        pass


def test_repo_section_sim(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    # stub objects used by sandbox_runner before import
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(
        monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot
    )
    _stub_module(
        monkeypatch,
        "menace.menace_memory_manager",
        MenaceMemoryManager=DummyBot,
        MemoryEntry=None,
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    _stub_module(monkeypatch, "networkx")
    sqla = types.ModuleType("sqlalchemy")
    sqla_engine = types.ModuleType("sqlalchemy.engine")
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqla)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqla_engine)
    _stub_module(monkeypatch, "networkx")
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)

    import sandbox_runner

    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text("def foo():\n    return 1\n")

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": ["x"]}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)
    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{"a": 1}], env_presets=[{"env": "dev"}]
    )
    assert tracker.roi_history


class _PredTracker(_ROITracker):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.predictions = []

    def record_metric_prediction(self, metric, predicted, actual):
        self.predictions.append((metric, predicted, actual))


def test_repo_section_sim_multi_env_predictions(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(
        monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot
    )
    _stub_module(
        monkeypatch,
        "menace.menace_memory_manager",
        MenaceMemoryManager=DummyBot,
        MemoryEntry=None,
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_PredTracker)
    _stub_module(monkeypatch, "networkx")
    sqla = types.ModuleType("sqlalchemy")
    sqla_engine = types.ModuleType("sqlalchemy.engine")
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqla)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqla_engine)
    _stub_module(monkeypatch, "networkx")
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)

    import sandbox_runner

    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text("def foo():\n    return 1\n")

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": ["x"]}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)
    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path),
        input_stubs=[{"a": 1}],
        env_presets=[{"env": "dev"}, {"env": "prod"}],
    )
    assert len(tracker.roi_history) >= 1
    assert len(tracker.predictions) >= 2


def test_section_worker_concurrency(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    flask_stub = types.ModuleType("flask")

    class _DummyFlask:
        def add_url_rule(self, *a, **k):
            pass

        def run(self, host="0.0.0.0", port=0):
            pass

    flask_stub.Flask = lambda *a, **k: _DummyFlask()
    flask_stub.jsonify = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "flask", flask_stub)

    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(
        monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot
    )
    _stub_module(
        monkeypatch,
        "menace.menace_memory_manager",
        MenaceMemoryManager=DummyBot,
        MemoryEntry=None,
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "menace.metrics_dashboard", MetricsDashboard=DummyBot)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(resolve_path("sandbox_runner.py")),  # path-ignore
        submodule_search_locations=[
            str(resolve_dir("sandbox_runner"))
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    import sandbox_runner.environment as env

    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env.multiprocessing, "cpu_count", lambda: 2)

    sections = {f"m{i}.py": {f"s{j}": ["pass"]} for i in range(5) for j in range(2)}  # path-ignore
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: sections,
    )

    running = 0
    peak = 0

    async def fake_worker(snippet, env_input, threshold):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return {"exit_code": 0}, []

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], env_presets=[{"CPU_LIMIT": 1}]
    )

    assert peak <= 2
