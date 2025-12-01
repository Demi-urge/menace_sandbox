import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class SpyBroker:
    def __init__(self) -> None:
        self.advertise_calls: list[tuple[object, object, bool]] = []
        self.active_owner = True

    def advertise(self, pipeline: object, sentinel: object, owner: bool = True) -> None:
        self.advertise_calls.append((pipeline, sentinel, owner))
        self.active_pipeline = pipeline
        self.active_sentinel = sentinel


_ORCHESTRATOR_STUBS: dict[str, dict[str, object]] = {
    "menace_sandbox.cognition_layer": {
        "build_cognitive_context": lambda *_, **__: None,
        "log_feedback": lambda *_, **__: None,
    },
    "menace_sandbox.dynamic_path_router": {
        "resolve_path": lambda *_, **__: "",
        "get_project_root": lambda: "",
    },
    "dynamic_path_router": {
        "resolve_path": lambda *_, **__: "",
        "get_project_root": lambda: "",
    },
    "menace_sandbox.knowledge_graph": {"KnowledgeGraph": object},
    "menace_sandbox.advanced_error_management": {"AutomatedRollbackManager": object},
    "menace_sandbox.self_coding_engine": {"SelfCodingEngine": object},
    "menace_sandbox.rollback_validator": {"RollbackValidator": object},
    "menace_sandbox.oversight_bots": {
        "L1OversightBot": object,
        "L2OversightBot": object,
        "L3OversightBot": object,
        "M1OversightBot": object,
        "M2OversightBot": object,
        "M3OversightBot": object,
        "H1OversightBot": object,
        "H2OversightBot": object,
        "H3OversightBot": object,
    },
    "menace_sandbox.model_automation_pipeline": {
        "ModelAutomationPipeline": object,
        "AutomationResult": object,
    },
    "menace_sandbox.bootstrap_timeout_policy": {
        "compute_prepare_pipeline_component_budgets": lambda *_, **__: {},
        "read_bootstrap_heartbeat": lambda *_, **__: None,
    },
    "menace_sandbox.discrepancy_detection_bot": {"DiscrepancyDetectionBot": object},
    "menace_sandbox.efficiency_bot": {"EfficiencyBot": object},
    "menace_sandbox.neuroplasticity": {
        "Outcome": object,
        "PathwayDB": object,
        "PathwayRecord": object,
    },
    "menace_sandbox.ad_integration": {"AdIntegration": object},
    "menace_sandbox.watchdog": {"Watchdog": object, "ContextBuilder": object},
    "menace_sandbox.error_bot": {"ErrorDB": object},
    "menace_sandbox.resource_allocation_optimizer": {"ROIDB": object},
    "menace_sandbox.data_bot": {"MetricsDB": object},
    "menace_sandbox.trending_scraper": {"TrendingScraper": object},
    "menace_sandbox.self_learning_service": {"main": lambda *_, **__: None},
    "menace_sandbox.strategic_planner": {"StrategicPlanner": object},
    "menace_sandbox.strategy_prediction_bot": {"StrategyPredictionBot": object},
    "menace_sandbox.autoscaler": {"Autoscaler": object},
    "menace_sandbox.trend_predictor": {"TrendPredictor": object},
    "menace_sandbox.identity_seeder": {"seed_identity": lambda *_, **__: None},
    "menace_sandbox.session_vault": {"SessionVault": object},
    "menace_sandbox.db_router": {"DBRouter": object},
}

_AGGREGATOR_STUBS: dict[str, dict[str, object]] = {
    "menace_sandbox.bot_registry": {"BotRegistry": object},
    "menace_sandbox.data_bot": {
        "DataBot": object,
        "persist_sc_thresholds": lambda *_, **__: None,
        "MetricsDB": object,
    },
    "menace_sandbox.bootstrap_helpers": {
        "bootstrap_state_snapshot": lambda: {"ready": True},
        "ensure_bootstrapped": lambda: None,
    },
    "menace_sandbox.bootstrap_readiness": {
        "readiness_signal": lambda: SimpleNamespace(
            await_ready=lambda timeout=None: None, describe=lambda: "ready"
        )
    },
    "menace_sandbox.self_coding_manager": {
        "SelfCodingManager": object,
        "internalize_coding_bot": lambda *_, **__: None,
    },
    "menace_sandbox.self_coding_engine": {"SelfCodingEngine": object},
    "menace_sandbox.threshold_service": {"ThresholdService": object},
    "menace_sandbox.code_database": {"CodeDB": object},
    "menace_sandbox.gpt_memory": {"GPTMemoryManager": object},
    "menace_sandbox.self_coding_thresholds": {"get_thresholds": lambda *_, **__: None},
    "vector_service.context_builder": {"ContextBuilder": object},
    "vector_service": {"EmbeddableDBMixin": object, "ContextBuilder": object},
    "menace_sandbox.shared_evolution_orchestrator": {"get_orchestrator": lambda *_, **__: None},
    "context_builder_util": {"create_context_builder": lambda *_, **__: None},
    "snippet_compressor": {"compress_snippets": lambda *_, **__: None},
    "menace_sandbox.menace_db": {"MenaceDB": object},
    "menace_sandbox.unified_event_bus": {"UnifiedEventBus": object},
    "menace_sandbox.chatgpt_enhancement_bot": {
        "EnhancementDB": object,
        "ChatGPTEnhancementBot": object,
        "Enhancement": object,
    },
    "menace_sandbox.chatgpt_prediction_bot": {
        "ChatGPTPredictionBot": object,
        "IdeaFeatures": object,
    },
    "menace_sandbox.text_research_bot": {"TextResearchBot": object},
    "menace_sandbox.video_research_bot": {"VideoResearchBot": object},
    "menace_sandbox.chatgpt_research_bot": {
        "ChatGPTResearchBot": object,
        "Exchange": object,
        "summarise_text": lambda *_, **__: None,
    },
    "menace_sandbox.database_manager": {
        "get_connection": lambda *_, **__: None,
        "DB_PATH": "",
    },
    "menace_sandbox.db_router": {
        "DBRouter": object,
        "GLOBAL_ROUTER": None,
        "init_db_router": lambda *_, **__: None,
    },
}


@pytest.fixture
def bootstrap_spy(monkeypatch: pytest.MonkeyPatch):
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

    heavy_stubs = (
        "stripe",
        "tensorflow",
        "torch",
        "faiss",
        "datasets",
        "matplotlib",
    )
    for module_name in heavy_stubs:
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    package = types.ModuleType("menace_sandbox")
    package.__path__ = [str(Path(__file__).resolve().parents[2])]
    monkeypatch.setitem(sys.modules, "menace_sandbox", package)

    broker = SpyBroker()
    pipeline = SimpleNamespace()
    manager = SimpleNamespace()
    prepare_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_resolve_bootstrap_placeholders(**_: object):
        return pipeline, manager, broker

    def fake_prepare_pipeline_for_bootstrap(*args: object, **kwargs: object) -> None:
        prepare_calls.append((args, kwargs))

    ready = SimpleNamespace(
        await_ready=lambda timeout=None: None,  # noqa: ARG005 - signature mirror
        describe=lambda: "ready",
    )

    def advertise_stub(
        *,
        dependency_broker: SpyBroker | None = None,
        pipeline: object | None = None,
        manager: object | None = None,
        owner: bool = True,
    ) -> tuple[object, object]:
        active_broker = dependency_broker or broker
        sentinel = manager or SimpleNamespace()
        pipeline_candidate = pipeline or SimpleNamespace(manager=sentinel)
        active_broker.advertise(pipeline_candidate, sentinel, owner)
        return pipeline_candidate, sentinel

    cbi_stub = types.ModuleType("coding_bot_interface")
    cbi_stub._bootstrap_dependency_broker = lambda: broker
    cbi_stub.advertise_bootstrap_placeholder = advertise_stub
    cbi_stub.read_bootstrap_heartbeat = lambda: None
    cbi_stub.get_active_bootstrap_pipeline = lambda: (None, None)
    cbi_stub._current_bootstrap_context = lambda: None
    cbi_stub._using_bootstrap_sentinel = lambda manager=None: False
    cbi_stub._peek_owner_promise = lambda: None
    cbi_stub._GLOBAL_BOOTSTRAP_COORDINATOR = None
    cbi_stub._resolve_caller_module_name = lambda: "test"
    cbi_stub._resolve_bootstrap_wait_timeout = lambda: 0.0
    cbi_stub.claim_bootstrap_dependency_entry = lambda *args, **kwargs: None
    cbi_stub.prepare_pipeline_for_bootstrap = fake_prepare_pipeline_for_bootstrap
    cbi_stub.self_coding_managed = lambda fn=None, **_: fn
    cbi_stub._looks_like_pipeline_candidate = lambda candidate: bool(candidate)
    cbi_stub._BOOTSTRAP_STATE = SimpleNamespace(helper_promotion_callbacks=[])

    monkeypatch.setitem(sys.modules, "coding_bot_interface", cbi_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", cbi_stub)

    monkeypatch.setattr(
        "bootstrap_gate.resolve_bootstrap_placeholders", fake_resolve_bootstrap_placeholders
    )
    monkeypatch.setattr("bootstrap_readiness.readiness_signal", lambda: ready)

    return broker, prepare_calls


@pytest.mark.parametrize(
    ("module_name", "stubbed_modules"),
    [
        ("menace_sandbox.research_aggregator_bot", _AGGREGATOR_STUBS),
        ("menace_sandbox.cognition_layer", {}),
        ("menace_sandbox.menace_orchestrator", _ORCHESTRATOR_STUBS),
    ],
)
def test_bootstrap_dependency_consumers_advertise_once(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    stubbed_modules: dict[str, dict[str, object]],
    bootstrap_spy: tuple[SpyBroker, list[tuple[tuple[object, ...], dict[str, object]]]],
) -> None:
    broker, prepare_calls = bootstrap_spy

    for stub_name, attrs in stubbed_modules.items():
        stub = types.ModuleType(stub_name)
        for attr_name, value in attrs.items():
            setattr(stub, attr_name, value)
        monkeypatch.setitem(sys.modules, stub_name, stub)

    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    assert module is not None
    assert len(broker.advertise_calls) == 1
    assert broker.advertise_calls[0][2] is True
    assert prepare_calls == []
