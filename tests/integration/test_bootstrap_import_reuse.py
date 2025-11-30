import importlib
import threading
import sys
import types
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
    pipeline = SimpleNamespace(
        manager=sentinel, initial_manager=sentinel, bootstrap_placeholder=True
    )
    cbi._mark_bootstrap_placeholder(sentinel)
    cbi._mark_bootstrap_placeholder(pipeline)
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
    pipeline = SimpleNamespace(
        manager=sentinel, initial_manager=sentinel, bootstrap_placeholder=True
    )
    cbi._mark_bootstrap_placeholder(sentinel)
    cbi._mark_bootstrap_placeholder(pipeline)
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
