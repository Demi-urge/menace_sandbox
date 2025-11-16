import builtins
import importlib.util
import sys
from pathlib import Path
import argparse
import types
import pytest
from dynamic_path_router import resolve_dir, resolve_path


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


class DummyDataBot:
    def __init__(self, *a, **k):
        pass


class DummyTracker:
    def __init__(self, *a, **k):
        self.module_deltas = {}
        self.metrics_history = {}
        self.roi_history = []
        self.best_synergy_metrics = {}
        self.best_roi = 0.0

    def update(self, *a, **k):
        return 0.0, [], False

    def load_history(self, path):
        pass


class DummySandbox:
    def __init__(self, *a, **k):
        pass

    def analyse_and_fix(self):
        pass


class DummyTester:
    def __init__(self, *a, **k):
        pass

    def _run_once(self):
        pass


class DummyOrch:
    def create_oversight(self, *a, **k):
        pass

    def run_cycle(self, *a, **k):
        class R:
            roi = None
        return R()


class DummyEngine:
    def __init__(self, *a, **k):
        pass


class DummyImprover:
    def __init__(self, *a, **k):
        pass

    def run_cycle(self):
        class R:
            roi = None
        return R()

    def _policy_state(self):
        return ()


class DummyBus:
    def __init__(self, persist_path=None, **kw):
        pass


def test_sandbox_init_fallback(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sc_mod = types.ModuleType("menace.startup_checks")
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement", SelfImprovementEngine=lambda *a, **k: DummyImprover())
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    class DummyContextBuilder:
        def refresh_db_weights(self):
            pass
    _stub_module(
        monkeypatch,
        "vector_service",
        ContextBuilder=DummyContextBuilder,
        FallbackResult=None,
    )
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.discrepancy_detection_bot", DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(monkeypatch, "menace.metrics_dashboard", MetricsDashboard=DummyBot)
    _stub_module(monkeypatch, "yaml", safe_load=lambda *a, **k: {}, safe_dump=lambda *a, **k: "")
    _stub_module(monkeypatch, "sandbox_runner.metrics_plugins", discover_metrics_plugins=lambda env: [], load_metrics_plugins=lambda names: [])
    _stub_module(monkeypatch, "menace.metrics_plugins", discover_metrics_plugins=lambda env: [], load_metrics_plugins=lambda names: [])

    pre_mod = types.ModuleType("menace.pre_execution_roi_bot")
    class PreExecutionROIBotStub:
        def __init__(self, *a, **k):
            pass
    pre_mod.PreExecutionROIBotStub = PreExecutionROIBotStub
    monkeypatch.setitem(sys.modules, "menace.pre_execution_roi_bot", pre_mod)


    path = resolve_path("sandbox_runner.py")  # path-ignore
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(path),
        submodule_search_locations=[str(resolve_dir("sandbox_runner"))],
    )
    sr = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sr
    spec.loader.exec_module(sr)

    caplog.set_level("INFO")
    ctx = sr._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path)), sr.ContextBuilder()
    )

    assert ctx.pre_roi_bot.__class__.__name__ == "PreExecutionROIBotStub"
    assert "PreExecutionROIBotStub" in caplog.text


def test_generate_input_stubs_seed(monkeypatch):
    path = resolve_path("sandbox_runner/environment.py")  # path-ignore
    spec = importlib.util.spec_from_file_location("sandbox_runner.environment", str(path))
    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = [str(path.parent)]
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    env = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(env)

    stubs1 = env.generate_input_stubs(1, strategy_order=["random"], seed=42)
    stubs2 = env.generate_input_stubs(1, strategy_order=["random"], seed=42)

    assert stubs1 == stubs2
