import types
import sys

from dynamic_path_router import resolve_path

if "filelock" not in sys.modules:
    fl = types.ModuleType("filelock")

    class _FL:
        def __init__(self, *a, **k):
            self.lock_file = "x"
            self.is_locked = False

        def acquire(self, *a, **k):
            self.is_locked = True

        def release(self):
            self.is_locked = False

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()
            return False

    fl.FileLock = _FL
    fl.Timeout = type("Timeout", (Exception,), {})
    sys.modules["filelock"] = fl


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
        self.roi_history = []
        self.calls = []
        self.synergy_history = []
        self.scenario_synergy = {}

    def get_scenario_synergy(self, name):
        return self.scenario_synergy.get(name, [])

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


def test_workflow_sim(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    # stub heavy modules used by sandbox_runner
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
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)

    from pathlib import Path as _P
    import importlib.util

    ROOT = _P(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        resolve_path("task_handoff_bot.py"),  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["simple_functions:print_ten"], title="t"))

    import sandbox_runner

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": ["x"]}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)

    tracker = sandbox_runner.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"), env_presets=[{"env": "dev"}]
    )

    assert tracker.roi_history


class _PredTracker(_ROITracker):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.predictions = []

    def record_metric_prediction(self, metric, predicted, actual):
        self.predictions.append((metric, predicted, actual))


class _MetricTracker:
    def __init__(self, *a, **k):
        self.metrics_history = {}
        self.roi_history = []

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.roi_history.append(curr)
        metrics = metrics or {}
        for m, v in metrics.items():
            self.metrics_history.setdefault(m, []).append(v)
        for name in self.metrics_history:
            if name not in metrics:
                last = (
                    self.metrics_history[name][-1]
                    if self.metrics_history[name]
                    else 0.0
                )
                self.metrics_history[name].append(last)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def register_metrics(self, *names):
        for n in names:
            self.metrics_history.setdefault(str(n), [0.0] * len(self.roi_history))

    def record_prediction(self, *a, **k):
        pass

    def record_metric_prediction(self, *a, **k):
        pass


def test_workflow_sim_multi_env(monkeypatch, tmp_path):
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
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_PredTracker)
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)

    from pathlib import Path as _P
    import importlib.util

    ROOT = _P(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]

    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        resolve_path("task_handoff_bot.py"),  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["simple_functions:print_ten"], title="t"))

    import sandbox_runner

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": ["x"]}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)

    tracker = sandbox_runner.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"),
        env_presets=[{"env": "dev"}, {"env": "prod"}],
    )

    assert len(tracker.roi_history) >= 1
    assert hasattr(tracker, "predictions") and len(tracker.predictions) >= 2


def test_workflow_sim_combined(monkeypatch, tmp_path):
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
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)

    from pathlib import Path as _P
    import importlib.util

    ROOT = _P(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]

    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        resolve_path("task_handoff_bot.py"),  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["simple_functions:print_ten"], title="t1"))
    wf_db.add(
        thb.WorkflowRecord(workflow=["simple_functions:print_eleven"], title="t2")
    )

    import sandbox_runner

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": ["x"]}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)

    tracker = sandbox_runner.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"), env_presets=[{"env": "dev"}]
    )

    assert any("all_workflows" in (c or []) for c in tracker.calls)


def test_workflow_function_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    import types, sys, importlib.util
    from pathlib import Path as _P

    ROOT = _P(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]

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
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)

    step_mod = tmp_path / "wf_steps.py"  # path-ignore
    step_mod.write_text(
        "import os\n"
        "def mark():\n"
        "    p=os.path.join(os.environ['WORKFLOW_FLAG_DIR'],'flag');\n"
        "    open(p,'w').write('x')\n"
        "    return True\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        resolve_path("task_handoff_bot.py"),  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["wf_steps:mark"], title="t"))

    import sandbox_runner

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": []}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)

    tracker = sandbox_runner.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"),
        env_presets=[{"WORKFLOW_FLAG_DIR": str(tmp_path), "PYTHONPATH": str(tmp_path)}],
    )

    assert (tmp_path / "flag").exists()


def test_workflow_sim_details(monkeypatch, tmp_path):
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
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_MetricTracker)
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)

    from pathlib import Path as _P
    import importlib.util

    ROOT = _P(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]

    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        resolve_path("task_handoff_bot.py"),  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["simple_functions:print_ten"], title="t1"))
    wf_db.add(
        thb.WorkflowRecord(workflow=["simple_functions:print_eleven"], title="t2")
    )

    import sandbox_runner
    import sandbox_runner.environment as env

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": []}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)

    async def fake_worker(snippet, env_input, threshold):
        scenario = env_input.get("env") or env_input.get("SCENARIO_NAME")
        roi = 1.0 if scenario == "dev" else 2.0
        metrics = {"profitability": roi}
        return {"exit_code": 0}, [(0.0, roi, metrics)]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    tracker, details = sandbox_runner.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"),
        env_presets=[{"env": "dev"}, {"env": "prod"}],
        return_details=True,
    )

    assert tracker.roi_history == [
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        1.0,
        2.0,
        2.0,
    ]
    synergy = tracker.metrics_history.get("synergy_roi")
    assert synergy and synergy[-2:] == [-1.0, -2.0]
    assert set(details) == {"1", "2", "_combined"}
    assert len(details["1"]) == 2
    assert len(details["2"]) == 2
    assert len(details["_combined"]) == 2
