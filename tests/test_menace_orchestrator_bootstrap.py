import sys
import time
import types
import threading
import pathlib
from unittest import mock
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

for _mod in (
    "menace_sandbox.oversight_bots",
    "oversight_bots",
    "menace_sandbox.capital_management_bot",
    "menace_sandbox.bot_registry",
    "menace_sandbox.code_database",
    "code_database",
):
    sys.modules.pop(_mod, None)


code_db_stub = types.ModuleType("code_database")
code_db_stub.CodeDB = object
code_db_stub.PatchRecord = object
sys.modules["code_database"] = code_db_stub
sys.modules["menace_sandbox.code_database"] = code_db_stub
_real_import = __import__


def _intercept_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in _STUB_NAMES:
        stub = _STUB_MODULES.get(name) or sys.modules.get(name)
        if stub is not None:
            return stub
        return sys.modules.setdefault(name, types.ModuleType(name))
    if name.startswith(("menace", "menace_sandbox")) and name not in {
        "menace_sandbox",
        "menace_sandbox.menace_orchestrator",
    }:
        return sys.modules.setdefault(name, types.ModuleType(name))
    if name.startswith("menace") and name in sys.modules:
        return sys.modules[name]
    return _real_import(name, globals, locals, fromlist, level)


def _stub_module(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _StubBotRegistry:
    def __init__(self) -> None:
        self.graph = types.SimpleNamespace(nodes={})

    def force_internalization_retry(self, *_, **__) -> bool:
        return False


_stub_module("menace_sandbox.knowledge_graph", KnowledgeGraph=type("KnowledgeGraph", (), {}))
_stub_module(
    "menace_sandbox.advanced_error_management",
    AutomatedRollbackManager=type("AutomatedRollbackManager", (), {}),
)
_stub_module("menace_sandbox.self_coding_engine", SelfCodingEngine=type("SelfCodingEngine", (), {}))
_stub_module("menace_sandbox.rollback_validator", RollbackValidator=type("RollbackValidator", (), {}))
oversight_stub = types.ModuleType("menace_sandbox.oversight_bots")
for _name in (
    "L1OversightBot",
    "L2OversightBot",
    "L3OversightBot",
    "M1OversightBot",
    "M2OversightBot",
    "M3OversightBot",
    "H1OversightBot",
    "H2OversightBot",
    "H3OversightBot",
):
    setattr(oversight_stub, _name, type(_name, (), {}))
sys.modules["menace_sandbox.oversight_bots"] = oversight_stub
oversight_stub.__spec__ = __import__("importlib.util").util.spec_from_loader(
    "menace_sandbox.oversight_bots", loader=None
)
_stub_module(
    "menace_sandbox.capital_management_bot",
    CapitalManagementBot=type("CapitalManagementBot", (), {}),
)
sys.modules["menace_sandbox.capital_management_bot"].__spec__ = __import__(
    "importlib.util"
).util.spec_from_loader("menace_sandbox.capital_management_bot", loader=None)
_stub_module("menace_sandbox.bot_registry", BotRegistry=_StubBotRegistry)
sys.modules["menace_sandbox.bot_registry"].__spec__ = __import__(
    "importlib.util"
).util.spec_from_loader("menace_sandbox.bot_registry", loader=None)

_stub_module(
    "menace_sandbox.model_automation_pipeline",
    ModelAutomationPipeline=type("ModelAutomationPipeline", (), {}),
    AutomationResult=type("AutomationResult", (), {}),
)
_stub_module(
    "menace_sandbox.performance_assessment_bot",
    PerformanceAssessmentBot=type("PerformanceAssessmentBot", (), {}),
    SimpleRL=type("SimpleRL", (), {}),
)
_stub_module(
    "menace_sandbox.operational_monitor_bot",
    OperationalMonitoringBot=type("OperationalMonitoringBot", (), {}),
)
_stub_module(
    "menace_sandbox.resource_allocation_optimizer",
    ResourceAllocationOptimizer=type("ResourceAllocationOptimizer", (), {}),
    ROIDB=type("ROIDB", (), {}),
)
_stub_module(
    "menace_sandbox.research_aggregator_bot",
    ResearchAggregatorBot=type("ResearchAggregatorBot", (), {}),
    InfoDB=type("InfoDB", (), {}),
    ResearchItem=type("ResearchItem", (), {}),
)
_stub_module("menace_sandbox.discrepancy_detection_bot", DiscrepancyDetectionBot=type("DiscrepancyDetectionBot", (), {}))
_stub_module("menace_sandbox.efficiency_bot", EfficiencyBot=type("EfficiencyBot", (), {}))
_stub_module("menace_sandbox.neuroplasticity", Outcome=type("Outcome", (), {}), PathwayDB=type("PathwayDB", (), {}), PathwayRecord=type("PathwayRecord", (), {}))
_stub_module("menace_sandbox.ad_integration", AdIntegration=type("AdIntegration", (), {}))
_stub_module(
    "menace_sandbox.watchdog",
    Watchdog=type("Watchdog", (), {"__init__": lambda self, *a, **k: None}),
    ContextBuilder=type("ContextBuilder", (), {}),
)
_stub_module(
    "menace_sandbox.error_bot", ErrorDB=type("ErrorDB", (), {"__init__": lambda self, *a, **k: None})
)
_stub_module(
    "menace_sandbox.data_bot",
    MetricsDB=type("MetricsDB", (), {}),
    DataBot=type("DataBot", (), {"__init__": lambda self, *a, **k: None}),
    MetricRecord=type("MetricRecord", (), {}),
    persist_sc_thresholds=lambda *_, **__: None,
)
_stub_module("menace_sandbox.trending_scraper", TrendingScraper=type("TrendingScraper", (), {}))
_stub_module("menace_sandbox.self_learning_service", main=lambda *_, **__: None)
_stub_module(
    "menace_sandbox.strategic_planner",
    StrategicPlanner=type("StrategicPlanner", (), {"__init__": lambda self, *a, **k: None}),
)
_stub_module(
    "menace_sandbox.strategy_prediction_bot",
    StrategyPredictionBot=type("StrategyPredictionBot", (), {"__init__": lambda self, *a, **k: None}),
)
_stub_module("menace_sandbox.autoscaler", Autoscaler=type("Autoscaler", (), {"__init__": lambda self, *a, **k: None}))
_stub_module(
    "menace_sandbox.trend_predictor", TrendPredictor=type("TrendPredictor", (), {"__init__": lambda self, *a, **k: None})
)
_stub_module("menace_sandbox.identity_seeder", seed_identity=lambda *_, **__: None)
_stub_module("menace_sandbox.session_vault", SessionVault=type("SessionVault", (), {}))
_stub_module(
    "menace_sandbox.cognition_layer",
    build_cognitive_context=lambda *_, **__: (None, None),
    log_feedback=lambda *_, **__: None,
)
_stub_module(
    "menace_sandbox.db_router",
    DBRouter=type("DBRouter", (), {"__init__": lambda self, *a, **k: None}),
)
_stub_module("db_router", DBRouter=type("DBRouter", (), {"__init__": lambda self, *a, **k: None}))
_stub_module(
    "bootstrap_timeout_policy",
    compute_prepare_pipeline_component_budgets=lambda *_0, **_1: None,
    read_bootstrap_heartbeat=lambda *_0, **_1: {"active": False},
)
_stub_module(
    "dynamic_path_router",
    resolve_path=lambda *_, **__: None,
    get_project_root=lambda *_, **__: None,
    repo_root=lambda: None,
)
_STUB_NAMES = {
    "code_database",
    "menace_sandbox.code_database",
    "menace.code_database",
    "menace_sandbox.knowledge_graph",
    "menace.knowledge_graph",
    "menace_sandbox.advanced_error_management",
    "menace.advanced_error_management",
    "menace_sandbox.self_coding_engine",
    "menace.self_coding_engine",
    "menace_sandbox.rollback_validator",
    "menace.rollback_validator",
    "menace_sandbox.oversight_bots",
    "menace.oversight_bots",
    "menace_sandbox.capital_management_bot",
    "menace.capital_management_bot",
    "menace_sandbox.bot_registry",
    "menace.bot_registry",
    "menace_sandbox.model_automation_pipeline",
    "menace.model_automation_pipeline",
    "menace_sandbox.performance_assessment_bot",
    "menace.performance_assessment_bot",
    "menace_sandbox.operational_monitor_bot",
    "menace.operational_monitor_bot",
    "menace_sandbox.discrepancy_detection_bot",
    "menace.discrepancy_detection_bot",
    "menace_sandbox.efficiency_bot",
    "menace.efficiency_bot",
    "menace_sandbox.neuroplasticity",
    "menace.neuroplasticity",
    "menace_sandbox.ad_integration",
    "menace.ad_integration",
    "menace_sandbox.watchdog",
    "menace.watchdog",
    "menace_sandbox.error_bot",
    "menace.error_bot",
    "menace_sandbox.resource_allocation_optimizer",
    "menace.resource_allocation_optimizer",
    "menace_sandbox.data_bot",
    "menace.data_bot",
    "menace_sandbox.research_aggregator_bot",
    "menace.research_aggregator_bot",
    "menace_sandbox.trending_scraper",
    "menace.trending_scraper",
    "menace_sandbox.self_learning_service",
    "menace.self_learning_service",
    "menace_sandbox.strategic_planner",
    "menace.strategic_planner",
    "menace_sandbox.strategy_prediction_bot",
    "menace.strategy_prediction_bot",
    "menace_sandbox.autoscaler",
    "menace.autoscaler",
    "menace_sandbox.trend_predictor",
    "menace.trend_predictor",
    "menace_sandbox.identity_seeder",
    "menace.identity_seeder",
    "menace_sandbox.session_vault",
    "menace.session_vault",
    "menace_sandbox.cognition_layer",
    "menace.cognition_layer",
    "menace_sandbox.db_router",
    "menace.db_router",
    "db_router",
    "bootstrap_timeout_policy",
    "dynamic_path_router",
}

for _alias, _target in [
    ("menace.code_database", "menace_sandbox.code_database"),
    ("menace.oversight_bots", "menace_sandbox.oversight_bots"),
    ("menace.capital_management_bot", "menace_sandbox.capital_management_bot"),
    ("menace.bot_registry", "menace_sandbox.bot_registry"),
    ("menace.model_automation_pipeline", "menace_sandbox.model_automation_pipeline"),
    ("menace.performance_assessment_bot", "menace_sandbox.performance_assessment_bot"),
    ("menace.operational_monitor_bot", "menace_sandbox.operational_monitor_bot"),
    ("menace.data_bot", "menace_sandbox.data_bot"),
    ("menace.research_aggregator_bot", "menace_sandbox.research_aggregator_bot"),
    ("menace.resource_allocation_optimizer", "menace_sandbox.resource_allocation_optimizer"),
]:
    sys.modules[_alias] = sys.modules[_target]

_STUB_MODULES = {name: module for name, module in sys.modules.items() if name in _STUB_NAMES}


def test_orchestrator_reuses_active_bootstrap(monkeypatch):
    import builtins
    import relevancy_radar

    monkeypatch.setattr(builtins, "__import__", _intercept_import)
    setattr(relevancy_radar, "original_import", __import__)
    setattr(relevancy_radar, "tracked_import", __import__)
    monkeypatch.setattr(relevancy_radar, "original_import", _intercept_import, raising=False)
    monkeypatch.setattr(relevancy_radar, "tracked_import", _intercept_import, raising=False)
    try:
        import menace_sandbox.menace_orchestrator as mo
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"menace_orchestrator unavailable: {exc}")

    pipeline = object()

    class DummyBroker:
        def __init__(self) -> None:
            self.pipeline = None
            self.sentinel = None
            self.advertised: list[tuple[object | None, object | None]] = []

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, *, pipeline=None, sentinel=None):
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            self.advertised.append((self.pipeline, self.sentinel))

    broker = DummyBroker()

    class DummyBuilder:
        def refresh_db_weights(self, **_):
            return None

    monkeypatch.setattr(mo, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(
        mo,
        "_current_bootstrap_context",
        lambda: types.SimpleNamespace(pipeline=pipeline, manager=None),
    )
    monkeypatch.setattr(mo, "_resolve_bootstrap_wait_timeout", lambda: 0.1)

    def _fail_prepare(**_: object) -> None:
        raise AssertionError("prepare_pipeline_for_bootstrap should not run")

    monkeypatch.setattr(mo, "prepare_pipeline_for_bootstrap", _fail_prepare)
    monkeypatch.setattr(mo, "read_bootstrap_heartbeat", lambda max_age=None: {"active": True})

    orchestrator = mo.MenaceOrchestrator(context_builder=DummyBuilder())

    assert orchestrator.pipeline is pipeline
    assert broker.advertised, "orchestrator should advertise the reused pipeline"
    assert orchestrator.model_id is None
    monkeypatch.setattr(builtins, "__import__", _intercept_import)


def test_orchestrator_waits_for_bootstrap_promise(monkeypatch):
    import builtins
    import relevancy_radar
    import menace_sandbox.coding_bot_interface as cbi

    monkeypatch.setattr(builtins, "__import__", _intercept_import)
    setattr(relevancy_radar, "original_import", __import__)
    setattr(relevancy_radar, "tracked_import", __import__)
    monkeypatch.setattr(relevancy_radar, "original_import", _intercept_import, raising=False)
    monkeypatch.setattr(relevancy_radar, "tracked_import", _intercept_import, raising=False)

    try:
        import menace_sandbox.menace_orchestrator as mo
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"menace_orchestrator unavailable: {exc}")

    pipeline = object()

    class DummyBroker:
        def __init__(self) -> None:
            self.pipeline = None
            self.sentinel = None
            self.advertised: list[tuple[object | None, object | None]] = []

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, *, pipeline=None, sentinel=None):
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            self.advertised.append((self.pipeline, self.sentinel))

    broker = DummyBroker()

    class DummyBuilder:
        def refresh_db_weights(self, **_):
            return None

    monkeypatch.setattr(mo, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(mo, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(mo, "_resolve_bootstrap_wait_timeout", lambda: 0.4)

    promise = cbi._BootstrapPipelinePromise()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = promise

    def _resolve_later():
        time.sleep(0.05)
        promise.resolve((pipeline, lambda *_: None))

    threading.Thread(target=_resolve_later, daemon=True).start()

    def _fail_prepare(**_: object) -> None:
        raise AssertionError("prepare_pipeline_for_bootstrap should not run when waiting on promise")

    monkeypatch.setattr(mo, "prepare_pipeline_for_bootstrap", _fail_prepare)
    monkeypatch.setattr(mo, "read_bootstrap_heartbeat", lambda max_age=None: {"active": True})

    orchestrator = mo.MenaceOrchestrator(context_builder=DummyBuilder())

    assert orchestrator.pipeline is pipeline
    assert broker.advertised, "orchestrator should advertise pipeline resolved from promise"
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = None
    monkeypatch.setattr(builtins, "__import__", _intercept_import)


def test_orchestrator_respects_guard_promise(monkeypatch):
    import builtins
    import relevancy_radar
    import menace_sandbox.coding_bot_interface as cbi

    monkeypatch.setattr(builtins, "__import__", _intercept_import)
    setattr(relevancy_radar, "original_import", __import__)
    setattr(relevancy_radar, "tracked_import", __import__)
    monkeypatch.setattr(relevancy_radar, "original_import", _intercept_import, raising=False)
    monkeypatch.setattr(relevancy_radar, "tracked_import", _intercept_import, raising=False)

    try:
        import menace_sandbox.menace_orchestrator as mo
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"menace_orchestrator unavailable: {exc}")

    pipeline = object()

    class DummyBroker:
        def __init__(self) -> None:
            self.pipeline = None
            self.sentinel = None
            self.advertised: list[tuple[object | None, object | None]] = []

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, *, pipeline=None, sentinel=None):
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            self.advertised.append((self.pipeline, self.sentinel))

    broker = DummyBroker()

    class DummyBuilder:
        def refresh_db_weights(self, **_):
            return None

    guard = object()
    cbi._BOOTSTRAP_STATE.active_bootstrap_guard = guard
    cbi._ensure_owner_promise(guard)

    monkeypatch.setattr(mo, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(mo, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(mo, "_resolve_bootstrap_wait_timeout", lambda: 0.4)

    def _resolve_broker():
        time.sleep(0.05)
        broker.advertise(pipeline=pipeline, sentinel=None)

    threading.Thread(target=_resolve_broker, daemon=True).start()

    def _fail_prepare(**_: object) -> None:
        raise AssertionError("prepare_pipeline_for_bootstrap should not run while guard promise active")

    monkeypatch.setattr(mo, "prepare_pipeline_for_bootstrap", _fail_prepare)
    monkeypatch.setattr(mo, "read_bootstrap_heartbeat", lambda max_age=None: {"active": True})

    orchestrator = mo.MenaceOrchestrator(context_builder=DummyBuilder())

    assert orchestrator.pipeline is pipeline
    assert broker.advertised, "orchestrator should advertise pipeline provided during guard wait"
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = None
    if hasattr(cbi._BOOTSTRAP_STATE, "owner_promises"):
        cbi._BOOTSTRAP_STATE.owner_promises.clear()
    if hasattr(cbi._BOOTSTRAP_STATE, "active_bootstrap_guard"):
        delattr(cbi._BOOTSTRAP_STATE, "active_bootstrap_guard")
    monkeypatch.setattr(builtins, "__import__", _intercept_import)


def test_orchestrator_reuses_active_owner_placeholder(monkeypatch):
    import builtins
    import relevancy_radar
    import menace_sandbox.coding_bot_interface as cbi

    monkeypatch.setattr(builtins, "__import__", _intercept_import)
    setattr(relevancy_radar, "original_import", __import__)
    setattr(relevancy_radar, "tracked_import", __import__)
    monkeypatch.setattr(relevancy_radar, "original_import", _intercept_import, raising=False)
    monkeypatch.setattr(relevancy_radar, "tracked_import", _intercept_import, raising=False)

    try:
        import menace_sandbox.menace_orchestrator as mo
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"menace_orchestrator unavailable: {exc}")

    pipeline = object()

    class DummyBroker:
        def __init__(self) -> None:
            self.pipeline = None
            self.sentinel = None
            self.active_owner = True
            self.advertised: list[tuple[object | None, object | None, object | None]] = []

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, *, pipeline=None, sentinel=None, owner=None):
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            if owner is True:
                self.active_owner = True
            elif owner is False:
                self.active_owner = False
            self.advertised.append((self.pipeline, self.sentinel, owner))

    broker = DummyBroker()

    class DummyBuilder:
        def refresh_db_weights(self, **_):
            return None

    promise = cbi._BootstrapPipelinePromise()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = promise

    def _resolve_later():
        time.sleep(0.05)
        promise.resolve((pipeline, lambda *_: None))

    threading.Thread(target=_resolve_later, daemon=True).start()

    monkeypatch.setattr(mo, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(mo, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(mo, "_resolve_bootstrap_wait_timeout", lambda: 0.4)

    def _fail_prepare(**_: object) -> None:
        raise AssertionError("prepare_pipeline_for_bootstrap should not run when owner active")

    monkeypatch.setattr(mo, "prepare_pipeline_for_bootstrap", _fail_prepare)
    monkeypatch.setattr(mo, "read_bootstrap_heartbeat", lambda max_age=None: {"active": True})

    orchestrator = mo.MenaceOrchestrator(context_builder=DummyBuilder())

    assert orchestrator.pipeline is pipeline
    assert any(getattr(entry[0], "bootstrap_placeholder", False) for entry in broker.advertised)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = None
    monkeypatch.setattr(builtins, "__import__", _intercept_import)


def test_orchestrator_seeds_placeholder_when_signals_without_placeholder(monkeypatch):
    import builtins
    import relevancy_radar
    import menace_sandbox.coding_bot_interface as cbi

    monkeypatch.setattr(builtins, "__import__", _intercept_import)
    setattr(relevancy_radar, "original_import", __import__)
    setattr(relevancy_radar, "tracked_import", __import__)
    monkeypatch.setattr(relevancy_radar, "original_import", _intercept_import, raising=False)
    monkeypatch.setattr(relevancy_radar, "tracked_import", _intercept_import, raising=False)

    try:
        import menace_sandbox.menace_orchestrator as mo
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"menace_orchestrator unavailable: {exc}")

    class DummyBroker:
        def __init__(self) -> None:
            self.pipeline = None
            self.sentinel = None
            self.active_owner = True
            self.advertised = []

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, *, pipeline=None, sentinel=None, owner=None):
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            if owner is not None:
                self.active_owner = owner
            self.advertised.append((pipeline, sentinel, owner))

    class DummyBuilder:
        def refresh_db_weights(self, **_):
            return None

    broker = DummyBroker()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = None
    monkeypatch.setattr(mo, "_BOOTSTRAP_BROKER", broker)
    monkeypatch.setattr(mo, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(mo, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(mo, "_resolve_bootstrap_wait_timeout", lambda: 0.05)
    monkeypatch.setattr(mo, "read_bootstrap_heartbeat", lambda max_age=None: {"active": True})

    def _fail_prepare(**_: object):
        raise AssertionError("prepare_pipeline_for_bootstrap should not run when signals active")

    monkeypatch.setattr(mo, "prepare_pipeline_for_bootstrap", _fail_prepare)

    orchestrator = mo.MenaceOrchestrator(context_builder=DummyBuilder())

    assert orchestrator.pipeline is not None
    assert any(
        getattr(entry[0], "bootstrap_placeholder", False) for entry in broker.advertised
    )

    monkeypatch.setattr(builtins, "__import__", _intercept_import)


def test_orchestrator_reuses_broker_pipeline_when_heartbeat_active(monkeypatch):
    import builtins
    import relevancy_radar

    monkeypatch.setattr(builtins, "__import__", _intercept_import)
    setattr(relevancy_radar, "original_import", __import__)
    setattr(relevancy_radar, "tracked_import", __import__)
    monkeypatch.setattr(relevancy_radar, "original_import", _intercept_import, raising=False)
    monkeypatch.setattr(relevancy_radar, "tracked_import", _intercept_import, raising=False)

    try:
        import menace_sandbox.menace_orchestrator as mo
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"menace_orchestrator unavailable: {exc}")

    pipeline = object()

    class DummyBroker:
        def __init__(self) -> None:
            self.pipeline = None
            self.sentinel = None

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, *, pipeline=None, sentinel=None):
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel

    broker = DummyBroker()

    class DummyBuilder:
        def refresh_db_weights(self, **_):
            return None

    monkeypatch.setattr(mo, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(mo, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(mo, "_resolve_bootstrap_wait_timeout", lambda: 0.5)
    monkeypatch.setattr(mo, "read_bootstrap_heartbeat", lambda max_age=None: {"active": True})

    def _resolve_broker():
        time.sleep(0.05)
        broker.advertise(pipeline=pipeline, sentinel=None)

    threading.Thread(target=_resolve_broker, daemon=True).start()

    def _fail_prepare(**_: object) -> None:
        raise AssertionError("prepare_pipeline_for_bootstrap should not run while broker publishes")

    monkeypatch.setattr(mo, "prepare_pipeline_for_bootstrap", _fail_prepare)

    orchestrator = mo.MenaceOrchestrator(context_builder=DummyBuilder())

    assert orchestrator.pipeline is pipeline
    monkeypatch.setattr(builtins, "__import__", _intercept_import)

def test_menace_orchestrator_placeholder_reused_when_broker_inactive(monkeypatch):
    import importlib

    module = importlib.reload(importlib.import_module("menace_sandbox.menace_orchestrator"))

    placeholder_pipeline = object()
    broker = types.SimpleNamespace(active_owner=False)

    monkeypatch.setattr(
        module,
        "resolve_bootstrap_placeholders",
        lambda **_: (placeholder_pipeline, object(), broker),
    )
    advertise = mock.Mock(side_effect=AssertionError("advertise_bootstrap_placeholder should not run"))
    monkeypatch.setattr(module, "advertise_bootstrap_placeholder", advertise)

    pipeline, resolved_broker = module._seed_bootstrap_placeholder()

    assert pipeline is placeholder_pipeline
    assert resolved_broker is broker
    advertise.assert_not_called()
