import sys
import types
from pathlib import Path

import pytest


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


def test_sandbox_scenario_diversity(monkeypatch, tmp_path):
    monkeypatch.setenv('MENACE_LIGHT_IMPORTS', '1')

    _stub_module(
        monkeypatch,
        'sandbox_settings',
        SandboxSettings=lambda: types.SimpleNamespace(
            scenario_metric_thresholds={
                'latency_error_rate': 0.5,
                'hostile_failures': 1.0,
                'misuse_failures': 2.0,
                'concurrency_throughput': 1.0,
                'schema_mismatch_rate': 0.05,
                'upstream_failure_rate': 0.05,
            },
            fail_on_missing_scenarios=False,
        ),
    )

    _stub_module(monkeypatch, 'menace.unified_event_bus', UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, 'menace.menace_orchestrator', MenaceOrchestrator=DummyBot)
    _stub_module(monkeypatch, 'menace.self_improvement_policy', SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, 'menace.self_improvement', SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, 'menace.self_test_service', SelfTestService=DummyBot)
    _stub_module(monkeypatch, 'menace.self_debugger_sandbox', SelfDebuggerSandbox=DummySandbox)
    _stub_module(monkeypatch, 'menace.self_coding_engine', SelfCodingEngine=DummyBot)
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
        'high_latency_api': {'resilience': 5.0, 'error_rate': 1.0},
        'hostile_input': {
            'resilience': 10.0,
            'failure_count': 2.0,
            'sanitization_failures': 1.0,
            'validation_failures': 0.0,
        },
        'user_misuse': {
            'resilience': 20.0,
            'failure_count': 3.0,
            'invalid_call_count': 1.0,
            'recovery_attempts': 2.0,
        },
        'concurrency_spike': {
            'resilience': 30.0,
            'throughput': 5.0,
            'concurrency_error_rate': 0.1,
            'concurrency_level': 2.0,
        },
        'schema_drift': {
            'resilience': 40.0,
            'schema_mismatches': 1.0,
            'schema_checks': 10.0,
        },
        'flaky_upstream': {
            'resilience': 50.0,
            'upstream_failures': 1.0,
            'upstream_requests': 10.0,
        },
    }
    roi_map = {
        'high_latency_api': 1.0,
        'hostile_input': 2.0,
        'user_misuse': 3.0,
        'concurrency_spike': 4.0,
        'schema_drift': 5.0,
        'flaky_upstream': 6.0,
    }
    call_count = {k: 0 for k in metric_map}

    async def fake_worker(snippet, env_input, threshold):
        scen = env_input.get('SCENARIO_NAME')
        call_count[scen] += 1
        roi = roi_map[scen] * call_count[scen]
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
        {'SCENARIO_NAME': 'high_latency_api'},
        {'SCENARIO_NAME': 'hostile_input'},
        {'SCENARIO_NAME': 'user_misuse'},
        {'SCENARIO_NAME': 'concurrency_spike'},
    ]

    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], env_presets=presets
    )

    scenarios = {mods[1] for mods in tracker.calls}
    assert scenarios == {
        'high_latency_api',
        'hostile_input',
        'user_misuse',
        'concurrency_spike',
        'schema_drift',
        'flaky_upstream',
    }

    expected_keys = {
        'high_latency_api': 'synergy_latency_error_rate',
        'hostile_input': 'synergy_hostile_failures',
        'user_misuse': 'synergy_misuse_failures',
        'concurrency_spike': 'synergy_concurrency_throughput',
        'schema_drift': 'synergy_schema_mismatch_rate',
        'flaky_upstream': 'synergy_upstream_failure_rate',
    }
    for scen, key in expected_keys.items():
        data = tracker.scenario_synergy.get(scen)
        assert data and key in data[0] and 'synergy_resilience' in data[0]

    roi_values = [tracker.scenario_synergy[s][0]['synergy_roi'] for s in expected_keys]
    assert len(roi_values) == len(expected_keys)

    found = set()
    for m in tracker.metrics:
        if (
            'latency_error_rate:high_latency_api' in m
            and m.get('latency_error_rate_breach') == 1.0
        ):
            found.add('high_latency_api')
        if (
            'hostile_failures:hostile_input' in m
            and m.get('hostile_failures_breach') == 1.0
        ):
            found.add('hostile_input')
        if (
            'misuse_failures:user_misuse' in m
            and m.get('misuse_failures_breach') == 1.0
        ):
            found.add('user_misuse')
        if (
            'concurrency_throughput:concurrency_spike' in m
            and m.get('concurrency_throughput_breach') == 1.0
        ):
            found.add('concurrency_spike')
        if (
            'schema_mismatch_rate:schema_drift' in m
            and m.get('schema_mismatch_rate_breach') == 1.0
        ):
            found.add('schema_drift')
        if (
            'upstream_failure_rate:flaky_upstream' in m
            and m.get('upstream_failure_rate_breach') == 1.0
        ):
            found.add('flaky_upstream')
    assert found == {
        'high_latency_api',
        'hostile_input',
        'user_misuse',
        'concurrency_spike',
        'schema_drift',
        'flaky_upstream',
    }
