import types
import sys
from pathlib import Path


import pytest
import contextvars


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg, _, sub = name.partition('.')
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
        self.metrics = []
        self.scenario_synergy = {}

    def update(self, prev_roi, roi, modules=None, resources=None, metrics=None):
        self.calls.append(modules)
        self.metrics.append(metrics or {})
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

    def register_metrics(self, *m):
        pass


def test_run_sandbox_multiscenario(monkeypatch, tmp_path):
    monkeypatch.setenv('MENACE_LIGHT_IMPORTS', '1')

    _stub_module(monkeypatch, 'menace.unified_event_bus', UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, 'menace.menace_orchestrator', MenaceOrchestrator=DummyBot)
    _stub_module(monkeypatch, 'menace.self_improvement_policy', SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, 'menace.self_improvement', SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, 'menace.self_test_service', SelfTestService=DummyBot)
    _stub_module(monkeypatch, 'menace.self_debugger_sandbox', SelfDebuggerSandbox=DummySandbox)
    _stub_module(
        monkeypatch,
        'menace.self_coding_engine',
        SelfCodingEngine=DummyBot,
        MANAGER_CONTEXT=contextvars.ContextVar('MANAGER_CONTEXT'),
    )
    _stub_module(monkeypatch, 'menace.code_database', PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, 'menace.menace_memory_manager', MenaceMemoryManager=DummyBot, MemoryEntry=None)
    _stub_module(monkeypatch, 'menace.discrepancy_detection_bot', DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, 'menace.audit_trail', AuditTrail=DummyBot)
    _stub_module(monkeypatch, 'menace.error_bot', ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, 'menace.data_bot', MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, 'menace.roi_tracker', ROITracker=_ROITracker)
    _stub_module(monkeypatch, 'networkx', DiGraph=lambda *a, **k: None)
    sqla = types.ModuleType('sqlalchemy')
    sqla_engine = types.ModuleType('sqlalchemy.engine')
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, 'sqlalchemy', sqla)
    monkeypatch.setitem(sys.modules, 'sqlalchemy.engine', sqla_engine)
    _stub_module(monkeypatch, 'jinja2', Template=lambda *a, **k: None)

    import sandbox_runner
    import sandbox_runner.environment as env
    import sandbox_runner.metrics_plugins as mp
    monkeypatch.setattr(mp, 'discover_metrics_plugins', lambda env: [])
    monkeypatch.setattr(mp, 'collect_plugin_metrics', lambda plugins, prev, actual, metrics: {})

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get('SCENARIO_NAME')
        exit_code = 1 if scen == 'three' else 0
        roi = 0.0 if exit_code else 1.0
        metrics = {'m': roi}
        return {'exit_code': exit_code}, [(0.0, roi, metrics)]

    def fake_scan_repo_sections(repo_path, modules=None):
        mod_file = Path(repo_path) / 'mod.py'  # path-ignore
        return {'mod.py': {'sec': mod_file.read_text().splitlines()}}  # path-ignore
    monkeypatch.setattr(sandbox_runner, 'scan_repo_sections', fake_scan_repo_sections, raising=False)

    monkeypatch.setattr(env, 'simulate_execution_environment', lambda *a, **k: {})
    monkeypatch.setattr(env, '_section_worker', fake_worker)

    mod = tmp_path / 'mod.py'  # path-ignore
    mod.write_text('def foo():\n    return 1\n')

    presets = [
        {'SCENARIO_NAME': 'one'},
        {'SCENARIO_NAME': 'two'},
        {'SCENARIO_NAME': 'three', 'FAIL': 1},
    ]

    tracker, details = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], env_presets=presets, return_details=True
    )

    scenarios = {mods[1] for mods in tracker.calls}
    assert scenarios == {'one', 'two', 'three'}

    for scen in ['one', 'two', 'three']:
        assert any(f'm:{scen}' in m for m in tracker.metrics)

    assert details['mod.py']['three'][0]['result']['exit_code'] == 1  # path-ignore
    assert details['mod.py']['one'][0]['result']['exit_code'] == 0  # path-ignore
    assert details['mod.py']['two'][0]['result']['exit_code'] == 1  # path-ignore


def test_run_sandbox_hostile_misuse_concurrency_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv('MENACE_LIGHT_IMPORTS', '1')

    _stub_module(monkeypatch, 'menace.unified_event_bus', UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, 'menace.menace_orchestrator', MenaceOrchestrator=DummyBot)
    _stub_module(monkeypatch, 'menace.self_improvement_policy', SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, 'menace.self_improvement', SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, 'menace.self_test_service', SelfTestService=DummyBot)
    _stub_module(monkeypatch, 'menace.self_debugger_sandbox', SelfDebuggerSandbox=DummySandbox)
    _stub_module(
        monkeypatch,
        'menace.self_coding_engine',
        SelfCodingEngine=DummyBot,
        MANAGER_CONTEXT=contextvars.ContextVar('MANAGER_CONTEXT'),
    )
    _stub_module(monkeypatch, 'menace.code_database', PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, 'menace.menace_memory_manager', MenaceMemoryManager=DummyBot, MemoryEntry=None)
    _stub_module(monkeypatch, 'menace.discrepancy_detection_bot', DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, 'menace.audit_trail', AuditTrail=DummyBot)
    _stub_module(monkeypatch, 'menace.error_bot', ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, 'menace.data_bot', MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, 'menace.roi_tracker', ROITracker=_ROITracker)
    _stub_module(monkeypatch, 'networkx', DiGraph=lambda *a, **k: None)
    sqla = types.ModuleType('sqlalchemy')
    sqla_engine = types.ModuleType('sqlalchemy.engine')
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, 'sqlalchemy', sqla)
    monkeypatch.setitem(sys.modules, 'sqlalchemy.engine', sqla_engine)
    _stub_module(monkeypatch, 'jinja2', Template=lambda *a, **k: None)

    import sandbox_runner
    import sandbox_runner.environment as env
    import sandbox_runner.metrics_plugins as mp
    monkeypatch.setattr(mp, 'discover_metrics_plugins', lambda env: [])
    monkeypatch.setattr(mp, 'collect_plugin_metrics', lambda plugins, prev, actual, metrics: {})

    metric_map = {
        'hostile_input': {'resilience': 10.0, 'hostile_failures': 2},
        'user_misuse': {'resilience': 20.0, 'misuse_failures': 3},
        'concurrency_spike': {'resilience': 30.0, 'concurrency_throughput': 5.0},
    }

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get('SCENARIO_NAME')
        roi = 1.0
        metrics = metric_map[scen]
        return {'exit_code': 0}, [(0.0, roi, metrics)]

    def fake_scan_repo_sections(repo_path, modules=None):
        mod_file = Path(repo_path) / 'mod.py'  # path-ignore
        return {'mod.py': {'sec': mod_file.read_text().splitlines()}}  # path-ignore
    monkeypatch.setattr(sandbox_runner, 'scan_repo_sections', fake_scan_repo_sections, raising=False)

    monkeypatch.setattr(env, 'simulate_execution_environment', lambda *a, **k: {})
    monkeypatch.setattr(env, '_section_worker', fake_worker)

    mod = tmp_path / 'mod.py'  # path-ignore
    mod.write_text('def foo():\n    return 1\n')

    presets = [
        {'SCENARIO_NAME': 'hostile_input'},
        {'SCENARIO_NAME': 'user_misuse'},
        {'SCENARIO_NAME': 'concurrency_spike'},
    ]

    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], env_presets=presets
    )

    scenarios = {mods[1] for mods in tracker.calls}
    assert scenarios == {'hostile_input', 'user_misuse', 'concurrency_spike'}

    expected_keys = {
        'hostile_input': 'synergy_hostile_failures',
        'user_misuse': 'synergy_misuse_failures',
        'concurrency_spike': 'synergy_concurrency_throughput',
    }
    for scen, key in expected_keys.items():
        data = tracker.scenario_synergy.get(scen)
        assert data and key in data[0] and 'synergy_resilience' in data[0]


@pytest.mark.parametrize(
    "steps",
    [
        ["simple_functions:foo"],
        ["self_improvement:cycle"],
    ],
)
def test_run_workflow_multiscenario(monkeypatch, tmp_path, steps):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox)
    _stub_module(
        monkeypatch,
        "menace.self_coding_engine",
        SelfCodingEngine=DummyBot,
        MANAGER_CONTEXT=contextvars.ContextVar("MANAGER_CONTEXT"),
    )
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_ROITracker)
    _stub_module(
        monkeypatch,
        "menace.environment_generator",
        generate_canonical_presets=lambda: [],
        generate_presets=lambda profiles=None: [],
        _CPU_LIMITS={},
        _MEMORY_LIMITS={},
    )

    class FakeWorkflowRecord:
        def __init__(self, workflow, wid=1, title="t"):
            self.workflow = workflow
            self.wid = wid
            self.title = title

    class FakeWorkflowDB:
        def __init__(self, path):
            self.path = Path(path)

        def fetch(self, limit=None):
            return [FakeWorkflowRecord(steps, wid=1)]

    _stub_module(
        monkeypatch,
        "menace.task_handoff_bot",
        WorkflowDB=FakeWorkflowDB,
        WorkflowRecord=FakeWorkflowRecord,
    )

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(env, "simulate_execution_environment", lambda *a, **k: {})

    async def fake_worker(snippet, env_input, threshold):
        return {"exit_code": 0}, [(0.0, 1.0, {"m": 1.0})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    presets = [
        {"SCENARIO_NAME": "one"},
        {"SCENARIO_NAME": "two"},
    ]

    tracker = sandbox_runner.run_workflow_simulations("wf.db", env_presets=presets)

    expected = {p["SCENARIO_NAME"] for p in presets}
    scenarios = {mods[1] for mods in tracker.calls if mods}
    assert expected <= scenarios
    for scen in expected:
        assert any(f"m:{scen}" in m for m in tracker.metrics)
