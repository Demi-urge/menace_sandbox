import importlib.util
import sys
import types
from pathlib import Path
from dynamic_path_router import resolve_dir, resolve_path

import pytest

import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
spec = importlib.util.spec_from_file_location(
    "menace", resolve_path("__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)


class DummyBot:
    def __init__(self, *a, **k):
        pass


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


def _setup_mm_stubs(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "menace", pkg)

    _stub_module(monkeypatch, "menace.unified_config_store", UnifiedConfigStore=DummyBot)
    _stub_module(monkeypatch, "menace.dependency_self_check", self_check=lambda: None)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(monkeypatch, "menace.self_coding_manager", PatchApprovalPolicy=DummyBot, SelfCodingManager=DummyBot)
    _stub_module(monkeypatch, "menace.advanced_error_management", AutomatedRollbackManager=DummyBot)
    _stub_module(monkeypatch, "menace.environment_bootstrap", EnvironmentBootstrapper=DummyBot)
    _stub_module(monkeypatch, "menace.auto_env_setup", ensure_env=lambda *a, **k: None, interactive_setup=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.auto_resource_setup", ensure_proxies=lambda *a, **k: None, ensure_accounts=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.external_dependency_provisioner", ExternalDependencyProvisioner=DummyBot)
    _stub_module(monkeypatch, "menace.disaster_recovery", DisasterRecovery=DummyBot)
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyBot)
    _stub_module(monkeypatch, "menace.chatgpt_enhancement_bot", ChatGPTEnhancementBot=DummyBot, EnhancementDB=DummyBot)
    _stub_module(monkeypatch, "menace.self_learning_service", main=lambda **k: None)
    _stub_module(monkeypatch, "menace.self_service_override", SelfServiceOverride=DummyBot)
    _stub_module(monkeypatch, "menace.resource_allocation_optimizer", ROIDB=DummyBot)
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.retry_utils", retry=lambda *a, **k: (lambda f: f))
    _stub_module(monkeypatch, "menace.override_policy", OverridePolicyManager=DummyBot, OverrideDB=DummyBot)
    _stub_module(monkeypatch, "menace.unified_update_service", UnifiedUpdateService=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.metrics_dashboard", MetricsDashboard=DummyBot)
    _stub_module(monkeypatch, "menace.discrepancy_detection_bot", DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummyBot)


class DummyTracker:
    reliability_value = 0.0

    def __init__(self, *a, **k):
        self.roi_history = []
        self.metrics_history = {}
        self.synergy_history = [{"synergy_roi": 0.0}]

    def update(self, prev_roi, roi, modules=None, resources=None, metrics=None):
        self.roi_history.append(roi)
        return 0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def register_metrics(self, *names):
        pass

    def record_metric_prediction(self, name, predicted, actual):
        pass

    def record_prediction(self, predicted, actual, *a, **k):
        pass

    def predict_all_metrics(self, manager, features):
        pass

    def predict_synergy(self, window=5):
        return 0.0

    def diminishing(self):
        return 0.01

    def synergy_reliability(self):
        return 0.0

    def reliability(self, metric=None, window=None, cv=None):
        if metric == "synergy_roi":
            return float(self.reliability_value)
        return 0.0


def _run_sandbox_loop(monkeypatch, tmp_path, reliability):
    _setup_mm_stubs(monkeypatch)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.patch_suggestion_db", PatchSuggestionDB=DummyBot)

    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub._run_sandbox = lambda *a, **k: None
    cli_stub.rank_scenarios = lambda *a, **k: None
    cli_stub.main = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)

    import argparse

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(resolve_path("sandbox_runner.py")),  # path-ignore
        submodule_search_locations=[str(resolve_dir("sandbox_runner"))],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    DummyTracker.reliability_value = reliability
    call_count = {"n": 0}

    def fake_cycle(ctx, sec, snip, tracker, scenario=None):
        call_count["n"] += 1
        roi = 0.0 if call_count["n"] == 1 else 0.02
        tracker.update(0.0, roi)

    monkeypatch.setattr(sandbox_runner, "_sandbox_cycle_runner", fake_cycle)

    class DummyMeta:
        def __init__(self):
            self.flagged_sections = {"a"}

        def rankings(self):
            return {}

        def diminishing(self, threshold=None, consecutive=3, entropy_threshold=None):
            return []

    class DummyCtx:
        def __init__(self):
            self.sections = {}
            self.all_section_names = set()
            self.meta_log = DummyMeta()
            self.tracker = DummyTracker()
            self.res_db = None
            self.prev_roi = 0.0
            self.predicted_roi = None
            self.roi_history_file = tmp_path / "roi.json"
            self.synergy_needed = False
            self.best_roi = 0.0
            self.best_synergy_metrics = {}
            self.brainstorm_history = []
            self.conversations = {}
            self.gpt_client = None
            self.brainstorm_interval = 0
            self.brainstorm_retries = 0
            self.adapt_presets = False

    monkeypatch.setattr(
        sandbox_runner, "_sandbox_init", lambda preset, args, context_builder: DummyCtx()
    )
    monkeypatch.setattr(sandbox_runner, "_sandbox_cleanup", lambda ctx: None)

    args = argparse.Namespace(workflow_db=str(tmp_path / "wf.db"), sandbox_data_dir=str(tmp_path), no_workflow_run=True)

    class DummyContextBuilder:
        def refresh_db_weights(self):
            pass

    monkeypatch.setattr(sandbox_runner, "ContextBuilder", DummyContextBuilder)

    sandbox_runner._sandbox_main({}, args)
    return call_count["n"]


@pytest.mark.parametrize("reliability", [0.9, 0.1])
def test_synergy_reliability_affects_loop(monkeypatch, tmp_path, reliability):
    count = _run_sandbox_loop(monkeypatch, tmp_path, reliability)
    if reliability > 0.8:
        assert count == 2
    else:
        assert count == 3


@pytest.mark.parametrize(
    "reliability,expected",
    [
        (1.0, 2),
        (0.95, 2),
        (0.81, 2),
        (0.8, 2),
        (0.3, 3),
    ],
)
def test_varied_reliability_levels(monkeypatch, tmp_path, reliability, expected):
    count = _run_sandbox_loop(monkeypatch, tmp_path, reliability)
    assert count == expected


def test_synergy_weight_influences_actions():
    import menace.self_improvement_policy as sip

    base = (0,) * 15 + (0, 0, 0, 0)
    next_high = (0,) * 15 + (10, 0, 0, 0)
    next_low = base

    high = sip.SelfImprovementPolicy(epsilon=0.0)
    high.update(base, -1.0, action=1)
    high.update(base, 0.0, action=0)
    high.update(base, 0.0, next_high, action=1)
    assert high.select_action(base) == 1

    low = sip.SelfImprovementPolicy(epsilon=0.0)
    low.update(base, -1.0, action=1)
    low.update(base, 0.0, action=0)
    low.update(base, 0.0, next_low, action=1)
    assert low.select_action(base) == 0
