import importlib
import threading
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_bot_interface as cbi

if "menace_sandbox" not in sys.modules:
    pkg_stub = types.ModuleType("menace_sandbox")
    pkg_stub.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["menace_sandbox"] = pkg_stub


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    broker = cbi._bootstrap_dependency_broker()
    broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    yield
    broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()


def _install_package_stub():
    pkg = sys.modules.get("menace_sandbox")
    if pkg is None:
        pkg = types.ModuleType("menace_sandbox")
        pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["menace_sandbox"] = pkg
    cbi_module = importlib.import_module("coding_bot_interface")
    sys.modules.setdefault("menace_sandbox.coding_bot_interface", cbi_module)
    setattr(pkg, "coding_bot_interface", cbi_module)


def _install_research_stubs():
    _install_package_stub()
    vector_service_stub = types.SimpleNamespace(
        ContextBuilder=object, EmbeddableDBMixin=object
    )
    sys.modules.setdefault("vector_service", vector_service_stub)
    sys.modules.setdefault(
        "menace_sandbox.unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
    )

    def _auto_link_stub(*_a, **_k):
        def decorator(fn):
            return fn

        return decorator

    sys.modules.setdefault(
        "menace_sandbox.auto_link", types.SimpleNamespace(auto_link=_auto_link_stub)
    )
    sys.modules.setdefault(
        "menace_sandbox.chatgpt_enhancement_bot",
        types.SimpleNamespace(
            EnhancementDB=object,
            ChatGPTEnhancementBot=object,
            Enhancement=types.SimpleNamespace,
        ),
    )
    sys.modules.setdefault(
        "menace_sandbox.chatgpt_prediction_bot",
        types.SimpleNamespace(ChatGPTPredictionBot=object, IdeaFeatures=object),
    )
    sys.modules.setdefault(
        "menace_sandbox.text_research_bot", types.SimpleNamespace(TextResearchBot=object)
    )
    sys.modules.setdefault(
        "menace_sandbox.video_research_bot",
        types.SimpleNamespace(VideoResearchBot=object),
    )
    sys.modules.setdefault(
        "menace_sandbox.chatgpt_research_bot",
        types.SimpleNamespace(
            ChatGPTResearchBot=object,
            Exchange=types.SimpleNamespace,
            summarise_text=lambda *_a, **_k: "",
        ),
    )
    sys.modules.setdefault(
        "menace_sandbox.database_manager",
        types.SimpleNamespace(
            get_connection=lambda *a, **k: types.SimpleNamespace(
                __enter__=lambda s: s,
                __exit__=lambda *_a: False,
                execute=lambda *_a, **_k: types.SimpleNamespace(fetchall=lambda: []),
            ),
            DB_PATH="",
        ),
    )
    sys.modules.setdefault(
        "menace_sandbox.capital_management_bot",
        types.SimpleNamespace(CapitalManagementBot=object),
    )
    sys.modules.setdefault(
        "menace_sandbox.db_router",
        types.SimpleNamespace(
            DBRouter=object, GLOBAL_ROUTER=None, init_db_router=lambda *_a, **_k: None
        ),
    )
    sys.modules.setdefault("menace_sandbox.menace_db", types.SimpleNamespace(MenaceDB=object))
    sys.modules.setdefault(
        "snippet_compressor", types.SimpleNamespace(compress_snippets=lambda meta, **_: meta)
    )


def _install_cognition_stubs():
    _install_package_stub()
    context_builder_stub = types.ModuleType("vector_service.context_builder")
    context_builder_stub.ContextBuilder = type("ContextBuilder", (), {})
    sys.modules.setdefault("vector_service.context_builder", context_builder_stub)

    class _FakeCognitionLayer:
        def __init__(self, *, context_builder: object, roi_tracker: object) -> None:
            self.context_builder = context_builder
            self.roi_tracker = roi_tracker
            self.session_ids: list[object] = []

        def query(self, *_args, **_kwargs):
            sid = object()
            self.session_ids.append(sid)
            return "ctx", sid

        async def query_async(self, *_args, **_kwargs):  # pragma: no cover - async helper
            _, sid = self.query()
            return "ctx", sid

        def record_patch_outcome(self, *_args, **_kwargs):
            return None

        async def record_patch_outcome_async(self, *_args, **_kwargs):  # pragma: no cover - async helper
            return None

        def reload_ranker_model(self, *_args, **_kwargs):
            return None

        def reload_reliability_scores(self, *_args, **_kwargs):
            return None

    cognition_layer_stub = types.ModuleType("vector_service.cognition_layer")
    cognition_layer_stub.CognitionLayer = _FakeCognitionLayer
    sys.modules.setdefault("vector_service.cognition_layer", cognition_layer_stub)

    roi_tracker_stub = types.ModuleType("roi_tracker")
    roi_tracker_stub.ROITracker = type("ROITracker", (), {})
    sys.modules.setdefault("roi_tracker", roi_tracker_stub)


def _install_orchestrator_stubs():
    _install_package_stub()
    sys.modules.setdefault(
        "dynamic_path_router",
        types.SimpleNamespace(
            resolve_path=lambda path: path, get_project_root=lambda: Path.cwd()
        ),
    )
    sys.modules.setdefault(
        "menace_sandbox.knowledge_graph",
        types.SimpleNamespace(KnowledgeGraph=type("KG", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.advanced_error_management",
        types.SimpleNamespace(AutomatedRollbackManager=type("ARM", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.self_coding_engine",
        types.SimpleNamespace(SelfCodingEngine=type("SCE", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.rollback_validator",
        types.SimpleNamespace(RollbackValidator=type("RV", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.oversight_bots",
        types.SimpleNamespace(
            L1OversightBot=type("L1", (), {}),
            L2OversightBot=type("L2", (), {}),
            L3OversightBot=type("L3", (), {}),
            M1OversightBot=type("M1", (), {}),
            M2OversightBot=type("M2", (), {}),
            M3OversightBot=type("M3", (), {}),
            H1OversightBot=type("H1", (), {}),
            H2OversightBot=type("H2", (), {}),
            H3OversightBot=type("H3", (), {}),
        ),
    )
    sys.modules.setdefault(
        "menace_sandbox.model_automation_pipeline",
        types.SimpleNamespace(
            ModelAutomationPipeline=type("Pipeline", (), {}),
            AutomationResult=type("AutomationResult", (), {}),
        ),
    )
    sys.modules.setdefault(
        "bootstrap_timeout_policy",
        types.SimpleNamespace(
            compute_prepare_pipeline_component_budgets=lambda: {},
            read_bootstrap_heartbeat=lambda: {"active": True},
        ),
    )
    sys.modules.setdefault(
        "menace_sandbox.discrepancy_detection_bot",
        types.SimpleNamespace(DiscrepancyDetectionBot=type("DDB", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.efficiency_bot",
        types.SimpleNamespace(EfficiencyBot=type("EFB", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.neuroplasticity",
        types.SimpleNamespace(Outcome=object, PathwayDB=type("PathwayDB", (), {}), PathwayRecord=object),
    )
    sys.modules.setdefault(
        "menace_sandbox.ad_integration", types.SimpleNamespace(AdIntegration=type("AdIntegration", (), {}))
    )
    sys.modules.setdefault(
        "menace_sandbox.watchdog",
        types.SimpleNamespace(Watchdog=type("Watchdog", (), {}), ContextBuilder=type("ContextBuilder", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.error_bot", types.SimpleNamespace(ErrorDB=type("ErrorDB", (), {}))
    )
    sys.modules.setdefault(
        "menace_sandbox.resource_allocation_optimizer",
        types.SimpleNamespace(ROIDB=type("ROIDB", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.data_bot", types.SimpleNamespace(MetricsDB=type("MetricsDB", (), {}))
    )
    sys.modules.setdefault(
        "menace_sandbox.trending_scraper",
        types.SimpleNamespace(TrendingScraper=type("TrendingScraper", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.self_learning_service", types.SimpleNamespace(main=lambda *_a, **_k: None)
    )
    sys.modules.setdefault(
        "menace_sandbox.strategic_planner",
        types.SimpleNamespace(StrategicPlanner=type("StrategicPlanner", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.strategy_prediction_bot",
        types.SimpleNamespace(StrategyPredictionBot=type("StrategyPredictionBot", (), {})),
    )
    sys.modules.setdefault(
        "menace_sandbox.autoscaler", types.SimpleNamespace(Autoscaler=type("Autoscaler", (), {}))
    )
    sys.modules.setdefault(
        "menace_sandbox.trend_predictor", types.SimpleNamespace(TrendPredictor=type("TrendPredictor", (), {}))
    )
    sys.modules.setdefault(
        "menace_sandbox.identity_seeder", types.SimpleNamespace(seed_identity=lambda *_a, **_k: None)
    )
    sys.modules.setdefault(
        "menace_sandbox.session_vault", types.SimpleNamespace(SessionVault=type("SessionVault", (), {}))
    )
    sys.modules.setdefault(
        "menace_sandbox.cognition_layer",
        types.SimpleNamespace(
            build_cognitive_context=lambda *_a, **_k: None,
            log_feedback=lambda *_a, **_k: None,
        ),
    )


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_advertised_placeholder_reused_across_consumers(monkeypatch, caplog):
    _install_research_stubs()
    _install_cognition_stubs()
    _install_orchestrator_stubs()

    pkg_stub = types.ModuleType("menace_sandbox")
    pkg_stub.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["menace_sandbox"] = pkg_stub
    sys.modules.setdefault("menace_sandbox.coding_bot_interface", cbi)
    setattr(pkg_stub, "coding_bot_interface", cbi)

    broker = cbi._bootstrap_dependency_broker()
    broker.clear()

    caplog.set_level("INFO", logger=cbi.logger.name)

    placeholder_pipeline, placeholder_manager = cbi.advertise_bootstrap_placeholder(
        dependency_broker=broker, owner=True
    )

    prepare_calls: list[object] = []
    original_prepare = cbi.prepare_pipeline_for_bootstrap

    def _tracked_prepare(**kwargs: object):
        prepare_calls.append(kwargs)
        return original_prepare(**kwargs)

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _tracked_prepare)

    class _Pipeline:
        pass

    pipeline, promoter = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=_Pipeline,
        context_builder=SimpleNamespace(),
        bot_registry=SimpleNamespace(),
        data_bot=SimpleNamespace(),
    )

    assert pipeline is placeholder_pipeline
    assert promoter is not None
    assert len(prepare_calls) == 1
    assert any(
        r.message.startswith("prepare_pipeline.bootstrap.preflight_broker_short_circuit")
        for r in caplog.records
    )

    sys.modules.pop("menace_sandbox.research_aggregator_bot", None)
    sys.modules.pop("menace_sandbox.prediction_manager_bot", None)
    sys.modules.pop("cognition_layer", None)
    sys.modules.pop("menace_sandbox.menace_orchestrator", None)

    readiness_stub = types.SimpleNamespace(
        await_ready=lambda *_, **__: None, describe=lambda *_a, **_k: "ready"
    )
    sys.modules["bootstrap_readiness"] = types.SimpleNamespace(
        readiness_signal=lambda: readiness_stub
    )
    sys.modules["bootstrap_gate"] = types.SimpleNamespace(
        resolve_bootstrap_placeholders=lambda **_: (
            placeholder_pipeline,
            placeholder_manager,
            broker,
        )
    )

    rab = importlib.import_module("menace_sandbox.research_aggregator_bot")
    rab.prepare_pipeline_for_bootstrap = _tracked_prepare

    prediction_manager = importlib.import_module(
        "menace_sandbox.prediction_manager_bot"
    )
    prediction_manager.prepare_pipeline_for_bootstrap = _tracked_prepare

    cognition_module = importlib.import_module("cognition_layer")
    cognition_module.prepare_pipeline_for_bootstrap = _tracked_prepare

    orchestrator_module = importlib.import_module("menace_sandbox.menace_orchestrator")
    orchestrator_module.prepare_pipeline_for_bootstrap = _tracked_prepare

    agg_pipeline, agg_manager, agg_broker = rab._bootstrap_placeholders()
    pred_pipeline, pred_manager, pred_broker = prediction_manager._bootstrap_placeholders()
    cog_pipeline, cog_manager, cog_broker = cognition_module._bootstrap_placeholders()

    orchestrator = orchestrator_module.MenaceOrchestrator(
        context_builder=SimpleNamespace()
    )

    assert broker.resolve() == (placeholder_pipeline, placeholder_manager)
    assert (agg_pipeline, agg_manager, agg_broker) == (
        placeholder_pipeline,
        placeholder_manager,
        broker,
    )
    assert (pred_pipeline, pred_manager, pred_broker) == (
        placeholder_pipeline,
        placeholder_manager,
        broker,
    )
    assert (cog_pipeline, cog_manager, cog_broker) == (
        placeholder_pipeline,
        placeholder_manager,
        broker,
    )
    assert orchestrator.pipeline is placeholder_pipeline
    assert len(prepare_calls) == 1
    assert not [record for record in caplog.records if "recursion" in record.message]


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_research_aggregator_import_reuses_placeholder(monkeypatch, caplog):
    _install_research_stubs()
    rab = importlib.import_module("menace_sandbox.research_aggregator_bot")
    importlib.reload(rab)

    rab._runtime_state = None
    rab._runtime_placeholder = None
    rab._runtime_initializing = False
    rab.pipeline = None
    rab.manager = None

    sentinel = SimpleNamespace(bootstrap_placeholder=True)
    pipeline = cbi._build_bootstrap_placeholder_pipeline(sentinel)
    broker = cbi._bootstrap_dependency_broker()
    broker.advertise(pipeline=pipeline, sentinel=sentinel)

    rab._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    monkeypatch.setattr(rab, "read_bootstrap_heartbeat", lambda: {"heartbeat": True})
    monkeypatch.setattr(
        rab, "_looks_like_pipeline_candidate", lambda obj: getattr(obj, "bootstrap_placeholder", False)
    )

    prepare_calls: list[object] = []

    def _fail_prepare(*_a, **_k):  # pragma: no cover - safety guard
        prepare_calls.append(object())
        raise AssertionError("prepare_pipeline_for_bootstrap should not run")

    monkeypatch.setattr(rab, "prepare_pipeline_for_bootstrap", _fail_prepare)

    caplog.set_level("INFO", logger=cbi.logger.name)

    deps = rab._ensure_runtime_dependencies(bootstrap_owner=object())

    assert deps.pipeline is pipeline
    assert deps.manager is sentinel
    assert broker.resolve() == (pipeline, sentinel)
    assert not prepare_calls
    assert not [r for r in caplog.records if "single-flight owner" in r.message]


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_menace_orchestrator_import_reuses_bootstrap_promise(monkeypatch, caplog):
    _install_orchestrator_stubs()
    broker = cbi._bootstrap_dependency_broker()
    sentinel = SimpleNamespace(bootstrap_placeholder=True)
    pipeline = cbi._build_bootstrap_placeholder_pipeline(sentinel)
    broker.advertise(pipeline=pipeline, sentinel=sentinel)

    active = cbi._BootstrapPipelinePromise()
    active.waiters = 1  # type: ignore[attr-defined]
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = active  # type: ignore[attr-defined]
    active.resolve((pipeline, lambda *_a, **_k: None))

    caplog.set_level("INFO")
    monkeypatch.setattr(
        "menace_sandbox.menace_orchestrator.read_bootstrap_heartbeat",
        lambda: {"active": True},
    )
    monkeypatch.setattr(
        "menace_sandbox.menace_orchestrator.ModelAutomationPipeline", object
    )
    monkeypatch.setattr(
        "menace_sandbox.menace_orchestrator.compute_prepare_pipeline_component_budgets",
        lambda: {},
    )

    prepare_calls: list[object] = []

    def _fail_prepare(*_a, **_k):  # pragma: no cover - safety guard
        prepare_calls.append(object())
        raise AssertionError("prepare_pipeline_for_bootstrap should not run")

    monkeypatch.setattr(
        "menace_sandbox.menace_orchestrator.prepare_pipeline_for_bootstrap",
        _fail_prepare,
    )

    from menace_sandbox.menace_orchestrator import MenaceOrchestrator

    orchestrator = MenaceOrchestrator(context_builder=SimpleNamespace())

    assert orchestrator.pipeline is pipeline
    assert orchestrator.pipeline_promoter is not None
    assert not prepare_calls
    assert any(
        r.message.startswith("menace orchestrator reusing bootstrap pipeline")
        for r in caplog.records
    )
    assert not [r for r in caplog.records if "single-flight owner" in r.message]


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_concurrent_helper_imports_reuse_single_flight(monkeypatch, caplog):
    _install_research_stubs()
    _install_orchestrator_stubs()

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)

    broker = cbi._bootstrap_dependency_broker()
    broker.clear()

    start_event = threading.Event()
    release_event = threading.Event()
    promotions: list[object] = []
    prepare_invocations: list[object] = []

    def _stub_inner(**_kwargs):
        prepare_invocations.append(_kwargs)
        broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)
        start_event.set()
        release_event.wait(timeout=5)
        return pipeline_placeholder, lambda manager: promotions.append(manager)

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl_inner", _stub_inner)

    class _Pipeline:
        vector_bootstrap_heavy = True

        def __init__(self, *, manager: object, **_kwargs) -> None:
            self.manager = manager

    caplog.set_level("INFO", logger=cbi.logger.name)

    owner_thread = threading.Thread(
        target=cbi.prepare_pipeline_for_bootstrap,
        kwargs={
            "pipeline_cls": _Pipeline,
            "context_builder": SimpleNamespace(),
            "bot_registry": SimpleNamespace(),
            "data_bot": SimpleNamespace(),
        },
    )
    owner_thread.start()
    assert start_event.wait(timeout=5)

    rab = importlib.import_module("menace_sandbox.research_aggregator_bot")
    importlib.reload(rab)
    rab.registry = None
    rab.data_bot = None
    rab._context_builder = None
    rab.engine = None
    rab._PipelineCls = None
    rab.pipeline = None
    rab.evolution_orchestrator = None
    rab.manager = None
    rab._runtime_state = None
    rab._runtime_placeholder = None
    rab._runtime_initializing = False
    rab._self_coding_configured = False

    class _Registry:
        pass

    class _DataBot:
        def __init__(self, *args, **kwargs) -> None:
            return None

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    monkeypatch.setattr(rab, "BotRegistry", _Registry)
    monkeypatch.setattr(rab, "DataBot", _DataBot)
    monkeypatch.setattr(rab, "create_context_builder", lambda: _ContextBuilder())
    monkeypatch.setattr(rab, "SelfCodingEngine", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "CodeDB", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "GPTMemoryManager", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "get_orchestrator", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        rab,
        "internalize_coding_bot",
        lambda *args, **kwargs: SimpleNamespace(pipeline=pipeline_placeholder),
    )
    monkeypatch.setattr(
        rab,
        "self_coding_managed",
        lambda **_: (lambda cls: cls),
    )

    aggregator_state: dict[str, object] = {}

    def _bootstrap_aggregator() -> None:
        try:
            aggregator_state["state"] = rab._ensure_runtime_dependencies(
                promote_pipeline=lambda manager: promotions.append(manager),
            )
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            aggregator_state["error"] = exc

    orchestrator_module = importlib.import_module("menace_sandbox.menace_orchestrator")
    importlib.reload(orchestrator_module)
    monkeypatch.setattr(
        "menace_sandbox.menace_orchestrator.compute_prepare_pipeline_component_budgets",
        lambda: {},
    )

    orchestrator_state: dict[str, object] = {}

    def _start_orchestrator() -> None:
        try:
            orchestrator_state["instance"] = orchestrator_module.MenaceOrchestrator(
                context_builder=SimpleNamespace()
            )
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            orchestrator_state["error"] = exc

    aggregator_thread = threading.Thread(target=_bootstrap_aggregator)
    orchestrator_thread = threading.Thread(target=_start_orchestrator)

    aggregator_thread.start()
    orchestrator_thread.start()

    release_event.set()
    owner_thread.join(timeout=5)
    aggregator_thread.join(timeout=5)
    orchestrator_thread.join(timeout=5)

    assert len(prepare_invocations) == 1
    assert broker.resolve() == (pipeline_placeholder, sentinel_placeholder)
    assert "error" not in aggregator_state
    assert "error" not in orchestrator_state
    assert "state" in aggregator_state
    assert "instance" in orchestrator_state
    assert aggregator_state["state"].pipeline is pipeline_placeholder
    assert orchestrator_state["instance"].pipeline is pipeline_placeholder
    assert not [r for r in caplog.records if "recursion" in r.message]


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_bootstrap_consumers_share_broker_placeholder(monkeypatch, caplog):
    _install_research_stubs()
    _install_orchestrator_stubs()
    _install_cognition_stubs()

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)

    broker = cbi._bootstrap_dependency_broker()
    broker.clear()

    start_event = threading.Event()
    release_event = threading.Event()
    promotions: list[object] = []
    prepare_invocations: list[object] = []

    def _stub_inner(**_kwargs):
        prepare_invocations.append(_kwargs)
        broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)
        start_event.set()
        release_event.wait(timeout=5)
        return pipeline_placeholder, lambda manager: promotions.append(manager)

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl_inner", _stub_inner)

    class _Pipeline:
        vector_bootstrap_heavy = True

        def __init__(self, *, manager: object, **_kwargs) -> None:
            self.manager = manager

    caplog.set_level("INFO", logger=cbi.logger.name)

    owner_thread = threading.Thread(
        target=cbi.prepare_pipeline_for_bootstrap,
        kwargs={
            "pipeline_cls": _Pipeline,
            "context_builder": SimpleNamespace(),
            "bot_registry": SimpleNamespace(),
            "data_bot": SimpleNamespace(),
        },
    )
    owner_thread.start()
    assert start_event.wait(timeout=5)

    rab = importlib.import_module("menace_sandbox.research_aggregator_bot")
    importlib.reload(rab)
    rab.registry = None
    rab.data_bot = None
    rab._context_builder = None
    rab.engine = None
    rab._PipelineCls = None
    rab.pipeline = None
    rab.evolution_orchestrator = None
    rab.manager = None
    rab._runtime_state = None
    rab._runtime_placeholder = None
    rab._runtime_initializing = False
    rab._self_coding_configured = False

    class _Registry:
        pass

    class _DataBot:
        def __init__(self, *args, **kwargs) -> None:
            return None

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    monkeypatch.setattr(rab, "BotRegistry", _Registry)
    monkeypatch.setattr(rab, "DataBot", _DataBot)
    monkeypatch.setattr(rab, "create_context_builder", lambda: _ContextBuilder())
    monkeypatch.setattr(rab, "SelfCodingEngine", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "CodeDB", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "GPTMemoryManager", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rab, "get_orchestrator", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        rab,
        "internalize_coding_bot",
        lambda *args, **kwargs: SimpleNamespace(pipeline=pipeline_placeholder),
    )
    monkeypatch.setattr(
        rab,
        "self_coding_managed",
        lambda **_: (lambda cls: cls),
    )
    monkeypatch.setattr(rab, "persist_sc_thresholds", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rab,
        "get_thresholds",
        lambda *_a, **_k: SimpleNamespace(roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0),
    )

    aggregator_state: dict[str, object] = {}

    def _bootstrap_aggregator() -> None:
        try:
            aggregator_state["state"] = rab._ensure_runtime_dependencies(
                promote_pipeline=lambda manager: promotions.append(manager),
            )
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            aggregator_state["error"] = exc

    orchestrator_module = importlib.import_module("menace_sandbox.menace_orchestrator")
    importlib.reload(orchestrator_module)
    monkeypatch.setattr(
        "menace_sandbox.menace_orchestrator.compute_prepare_pipeline_component_budgets",
        lambda: {},
    )

    orchestrator_state: dict[str, object] = {}

    def _start_orchestrator() -> None:
        try:
            orchestrator_state["instance"] = orchestrator_module.MenaceOrchestrator(
                context_builder=SimpleNamespace()
            )
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            orchestrator_state["error"] = exc

    cog_module = importlib.import_module("cognition_layer")

    aggregator_thread = threading.Thread(target=_bootstrap_aggregator)
    orchestrator_thread = threading.Thread(target=_start_orchestrator)

    aggregator_thread.start()
    orchestrator_thread.start()

    release_event.set()
    owner_thread.join(timeout=5)
    aggregator_thread.join(timeout=5)
    orchestrator_thread.join(timeout=5)

    assert len(prepare_invocations) == 1
    assert broker.resolve() == (pipeline_placeholder, sentinel_placeholder)
    assert "error" not in aggregator_state
    assert "error" not in orchestrator_state
    assert "state" in aggregator_state
    assert "instance" in orchestrator_state
    assert aggregator_state["state"].pipeline is pipeline_placeholder
    assert orchestrator_state["instance"].pipeline is pipeline_placeholder
    assert getattr(cog_module, "_BOOTSTRAP_PLACEHOLDER_PIPELINE") is pipeline_placeholder
    assert getattr(cog_module, "_BOOTSTRAP_PLACEHOLDER_MANAGER") is sentinel_placeholder
    assert promotions == [sentinel_placeholder]
    assert not [r for r in caplog.records if "recursion" in r.message]


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_cold_start_reuses_dependency_broker(monkeypatch, caplog):
    sentinel = SimpleNamespace(bootstrap_placeholder=True)
    pipeline = cbi._build_bootstrap_placeholder_pipeline(sentinel)

    broker = cbi._bootstrap_dependency_broker()
    broker.advertise(pipeline=pipeline, sentinel=sentinel, owner=True)

    def _fail_prepare(**_kwargs):  # pragma: no cover - safety guard
        raise AssertionError("prepare pipeline should not run during broker reuse")

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl", _fail_prepare)

    class _Pipeline:
        pass

    caplog.set_level("INFO", logger=cbi.logger.name)

    reused_pipeline, promote = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=_Pipeline,
        context_builder=SimpleNamespace(),
        bot_registry=SimpleNamespace(),
        data_bot=SimpleNamespace(),
    )

    assert reused_pipeline is pipeline
    promote(object())
    assert broker.resolve() == (pipeline, sentinel)
    assert any(
        r.message.startswith("prepare_pipeline.bootstrap.preflight_broker_short_circuit")
        for r in caplog.records
    )


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_cold_start_waits_for_active_promise(monkeypatch, caplog):
    promise = cbi._BootstrapPipelinePromise()
    sentinel = SimpleNamespace()
    pipeline = SimpleNamespace(manager=sentinel)
    resolved_promote = lambda *_a, **_k: None  # noqa: E731
    promise.waiters = 1  # type: ignore[attr-defined]
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = promise  # type: ignore[attr-defined]

    def _resolve_active() -> None:
        promise.resolve((pipeline, resolved_promote))

    resolver = threading.Timer(0.01, _resolve_active)
    resolver.start()

    def _fail_prepare(**_kwargs):  # pragma: no cover - safety guard
        raise AssertionError("prepare pipeline should not run during active promise reuse")

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl", _fail_prepare)

    class _Pipeline:
        pass

    caplog.set_level("INFO", logger=cbi.logger.name)

    reused_pipeline, reused_promote = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=_Pipeline,
        context_builder=SimpleNamespace(),
        bot_registry=SimpleNamespace(),
        data_bot=SimpleNamespace(),
    )

    resolver.cancel()
    assert reused_pipeline is pipeline
    assert reused_promote is resolved_promote
    assert promise.waiters == 2  # type: ignore[attr-defined]
    assert any(
        r.message.startswith("prepare_pipeline.bootstrap.preflight_promise_short_circuit")
        for r in caplog.records
    )


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_bootstrap_consumer_warns_without_placeholder(monkeypatch, caplog):
    _install_research_stubs()
    _install_cognition_stubs()

    broker = cbi._bootstrap_dependency_broker()
    broker.clear()
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]

    rab = importlib.import_module("menace_sandbox.research_aggregator_bot")
    importlib.reload(rab)
    rab._runtime_state = None
    rab._runtime_placeholder = None
    rab._runtime_initializing = False
    rab.pipeline = None
    rab.manager = None
    rab.registry = None
    rab.data_bot = None
    rab._context_builder = None
    rab.engine = None
    rab._self_coding_configured = False

    monkeypatch.setattr(
        rab,
        "advertise_bootstrap_placeholder",
        lambda **_: (_ for _ in ()).throw(RuntimeError("dependency broker missing")),
    )
    monkeypatch.setattr(rab, "prepare_pipeline_for_bootstrap", lambda **_: (_ for _ in ()).throw(AssertionError("prepare should not run")))

    caplog.set_level("WARNING", logger=rab.logger.name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = rab._ensure_runtime_dependencies(bootstrap_owner=False)

    assert state.pipeline is None
    assert any(
        "Failed to advertise early bootstrap placeholder" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        "No bootstrap dependency broker placeholder available" in str(w.message)
        for w in caught
    )


@pytest.mark.integration
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_orchestrator_placeholder_health_bootstrap_dedupes_prepare(monkeypatch):
    _install_orchestrator_stubs()

    import menace_sandbox.menace_orchestrator as mo

    prepare_calls: list[dict[str, object]] = []

    def _prepare_pipeline_for_bootstrap(**kwargs: object):
        prepare_calls.append(kwargs)
        return SimpleNamespace(manager=SimpleNamespace(from_prepare=True)), (
            lambda *_a, **_k: None
        )

    class _Broker:
        def __init__(self) -> None:
            self.active_owner = False
            self.active_pipeline: object | None = None
            self.active_sentinel: object | None = None

        def resolve(self):
            return self.active_pipeline, self.active_sentinel

        def advertise(self, *, pipeline=None, sentinel=None, owner=None):
            if owner is True and not self.active_owner:
                self.active_owner = True
                prepared_pipeline, _ = _prepare_pipeline_for_bootstrap()
                self.active_pipeline = prepared_pipeline
                self.active_sentinel = getattr(prepared_pipeline, "manager", None)
            elif owner is False:
                self.active_owner = False
            if pipeline is not None:
                self.active_pipeline = pipeline
            if sentinel is not None:
                self.active_sentinel = sentinel

    broker = _Broker()

    monkeypatch.setattr(mo, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(mo, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(mo, "_resolve_bootstrap_wait_timeout", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(mo, "prepare_pipeline_for_bootstrap", _prepare_pipeline_for_bootstrap)
    monkeypatch.setattr(mo, "read_bootstrap_heartbeat", lambda *_a, **_k: {"active": True})

    def _advertise_bootstrap_placeholder(*, dependency_broker=None, **_kwargs):
        (dependency_broker or broker).advertise(owner=True)
        return SimpleNamespace(bootstrap_placeholder=True, manager=None), None

    monkeypatch.setattr(mo, "advertise_bootstrap_placeholder", _advertise_bootstrap_placeholder)

    monkeypatch.setattr(mo, "ErrorDB", lambda *_, **__: SimpleNamespace())
    monkeypatch.setattr(mo, "Watchdog", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(mo, "StrategicPlanner", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(mo, "StrategyPredictionBot", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(mo, "Autoscaler", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(mo, "TrendPredictor", lambda *_a, **_k: SimpleNamespace())

    orchestrator = mo.MenaceOrchestrator(
        context_builder=SimpleNamespace(refresh_db_weights=lambda: None)
    )

    assert orchestrator.pipeline is not None
    assert len(prepare_calls) == 1
