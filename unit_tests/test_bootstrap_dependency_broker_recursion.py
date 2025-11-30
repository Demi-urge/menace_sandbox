"""Ensure bootstrap helpers reuse the shared dependency broker placeholder."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


class _LazyNamespace(types.SimpleNamespace):
    def __getattr__(self, name: str):  # pragma: no cover - defensive fallback
        value = types.SimpleNamespace()
        setattr(self, name, value)
        return value


class _DummyBroker:
    def __init__(self) -> None:
        self.pipeline = "broker-pipeline"
        self.manager = "broker-manager"
        self.advertise_calls: list[tuple[object | None, object | None]] = []

    def resolve(self) -> tuple[object | None, object | None]:
        return self.pipeline, self.manager

    def advertise(
        self, *, pipeline: object | None = None, sentinel: object | None = None, owner: bool | None = None
    ) -> None:
        if pipeline is not None:
            self.pipeline = pipeline
        if sentinel is not None:
            self.manager = sentinel
        self.advertise_calls.append((pipeline, sentinel))


class _DummyCoordinator:
    def __init__(self, pipeline: object) -> None:
        self.prepare_calls = 0
        self._pipeline = pipeline

    def claim(self):  # pragma: no cover - claim path unused in placeholder reuse
        self.prepare_calls += 1
        promise = types.SimpleNamespace(wait=lambda: (self._pipeline, lambda *_: None), waiters=1)
        return True, promise

    def peek_active(self):  # pragma: no cover - not exercised in these tests
        return None


def _install_stub(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    module.__file__ = str(Path(__file__).resolve())
    sys.modules[name] = module
    return module


def _prepare_stub_environment(monkeypatch):
    broker = _DummyBroker()
    coordinator = _DummyCoordinator(broker.pipeline)
    prepare_calls = {"count": 0}

    def _fake_advertise_bootstrap_placeholder(*, dependency_broker, pipeline=None, manager=None, **_kwargs):
        dependency_broker.advertise(pipeline=pipeline, sentinel=manager, owner=True)
        return (pipeline or dependency_broker.pipeline), (manager or dependency_broker.manager)

    def _fake_prepare_pipeline_for_bootstrap(**_kwargs):
        prepare_calls["count"] += 1
        return broker.pipeline, lambda *_: None

    cbi_stub = _install_stub(
        "coding_bot_interface",
        _BOOTSTRAP_STATE=types.SimpleNamespace(depth=1, helper_promotion_callbacks=[]),
        _bootstrap_dependency_broker=lambda: broker,
        advertise_bootstrap_placeholder=_fake_advertise_bootstrap_placeholder,
        get_active_bootstrap_pipeline=lambda: broker.resolve(),
        _GLOBAL_BOOTSTRAP_COORDINATOR=coordinator,
        prepare_pipeline_for_bootstrap=_fake_prepare_pipeline_for_bootstrap,
        _looks_like_pipeline_candidate=lambda obj: obj is not None,
        _current_bootstrap_context=lambda: None,
        _using_bootstrap_sentinel=lambda *_: True,
        _peek_owner_promise=lambda *_: None,
        _resolve_bootstrap_wait_timeout=lambda: 0.01,
        read_bootstrap_heartbeat=lambda: True,
        self_coding_managed=lambda **_: (lambda cls: cls),
    )
    sys.modules["menace_sandbox.coding_bot_interface"] = cbi_stub

    _install_stub("bot_registry", BotRegistry=type("BotRegistry", (), {}))
    _install_stub(
        "data_bot",
        DataBot=type("DataBot", (), {"__init__": lambda self, *_, **__: None}),
        MetricsDB=type("MetricsDB", (), {}),
        persist_sc_thresholds=lambda *_: None,
    )
    _install_stub("threshold_service", ThresholdService=type("ThresholdService", (), {}))
    _install_stub("code_database", CodeDB=type("CodeDB", (), {"__init__": lambda self, *_, **__: None}))
    _install_stub("gpt_memory", GPTMemoryManager=type("GPTMemoryManager", (), {"__init__": lambda self, *_, **__: None}))
    _install_stub("self_coding_manager", SelfCodingManager=type("SelfCodingManager", (), {}), internalize_coding_bot=lambda *_: None)
    _install_stub("self_coding_engine", SelfCodingEngine=type("SelfCodingEngine", (), {}))
    _install_stub("self_coding_thresholds", get_thresholds=lambda *_: types.SimpleNamespace())
    _install_stub("shared_evolution_orchestrator", get_orchestrator=lambda *_: None)
    _install_stub("context_builder_util", create_context_builder=lambda *_, **__: _LazyNamespace())

    vector_context = _install_stub("vector_service.context_builder", ContextBuilder=type("ContextBuilder", (), {}))
    sys.modules.setdefault("vector_service", types.ModuleType("vector_service")).ContextBuilder = vector_context.ContextBuilder
    sys.modules["vector_service"].EmbeddableDBMixin = type("EmbeddableDBMixin", (), {})
    _install_stub("vector_service.cognition_layer", CognitionLayer=type("CognitionLayer", (), {}))
    _install_stub("roi_tracker", ROITracker=type("ROITracker", (), {}))

    _install_stub("chatgpt_enhancement_bot", EnhancementDB=object, ChatGPTEnhancementBot=object, Enhancement=object)
    _install_stub("chatgpt_prediction_bot", ChatGPTPredictionBot=object, IdeaFeatures=object)
    _install_stub("text_research_bot", TextResearchBot=object)
    _install_stub("chatgpt_research_bot", ChatGPTResearchBot=object, Exchange=object, summarise_text=lambda *_: "")
    _install_stub("database_manager", get_connection=lambda *_: None, DB_PATH=str(Path(__file__).resolve()))
    _install_stub("db_router", DBRouter=type("DBRouter", (), {}), GLOBAL_ROUTER=None, init_db_router=lambda *_: None)
    _install_stub("snippet_compressor", compress_snippets=lambda *_: [])
    _install_stub("menace_db", MenaceDB=None)

    _install_stub("prediction_registry", PredictionRegistry=object)
    _install_stub("uuid", uuid4=lambda: "uuid")
    _install_stub("pandas", DataFrame=object)

    _install_stub("fastapi", FastAPI=lambda *_args, **_kwargs: _LazyNamespace(on_event=lambda *_: lambda *_: None))
    _install_stub("uvicorn", run=lambda *_args, **_kwargs: None)
    _install_stub("vector_service.vectorizer", SharedVectorService=type("SharedVectorService", (), {"__init__": lambda self: None}))
    _install_stub(
        "vector_service.embedding_backfill",
        watch_databases=lambda *_: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda *_: None),
        check_staleness=lambda *_: None,
        ensure_embeddings_fresh=lambda *_: None,
        schedule_backfill=lambda *_: None,
        StaleEmbeddingsError=RuntimeError,
    )
    _install_stub("vector_service.embedding_scheduler", start_scheduler_from_env=lambda *_: None)

    _install_stub("dynamic_path_router", resolve_path=lambda *_: "", get_project_root=lambda: Path.cwd())
    _install_stub("knowledge_graph", KnowledgeGraph=type("KnowledgeGraph", (), {}))
    _install_stub("advanced_error_management", AutomatedRollbackManager=type("AutomatedRollbackManager", (), {}))
    _install_stub("rollback_validator", RollbackValidator=type("RollbackValidator", (), {}))
    _install_stub("self_learning_service", main=lambda *_: None)
    _install_stub("strategic_planner", StrategicPlanner=type("StrategicPlanner", (), {}))
    _install_stub("strategy_prediction_bot", StrategyPredictionBot=type("StrategyPredictionBot", (), {}))
    _install_stub("autoscaler", Autoscaler=type("Autoscaler", (), {}))
    _install_stub("trend_predictor", TrendPredictor=type("TrendPredictor", (), {}))
    _install_stub("identity_seeder", seed_identity=lambda *_: None)
    _install_stub("session_vault", SessionVault=type("SessionVault", (), {}))
    _install_stub("watchdog", Watchdog=type("Watchdog", (), {}), ContextBuilder=vector_context.ContextBuilder)
    _install_stub("error_bot", ErrorDB=type("ErrorDB", (), {}))
    _install_stub("resource_allocation_optimizer", ROIDB=type("ROIDB", (), {}))
    _install_stub("data_bot", MetricsDB=type("MetricsDB", (), {}), DataBot=type("DataBot", (), {"__init__": lambda self, *_, **__: None}))
    _install_stub("trending_scraper", TrendingScraper=type("TrendingScraper", (), {}))
    _install_stub("oversight_bots", L1OversightBot=object, L2OversightBot=object, L3OversightBot=object, M1OversightBot=object, M2OversightBot=object, M3OversightBot=object, H1OversightBot=object, H2OversightBot=object, H3OversightBot=object)

    return broker, coordinator, prepare_calls


def _load_real_module(module_name: str, file_name: str):
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("menace_sandbox")
    package.__path__ = [str(root)]
    package.__spec__ = importlib.util.spec_from_loader("menace_sandbox", loader=None, is_package=True)
    sys.modules.setdefault("menace_sandbox", package)

    module_path = root / file_name
    spec = importlib.util.spec_from_file_location(
        f"menace_sandbox.{module_name}", module_path, submodule_search_locations=package.__path__
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_shared_dependency_broker_prevents_recursion(monkeypatch):
    broker, coordinator, prepare_calls = _prepare_stub_environment(monkeypatch)

    research_aggregator_bot = _load_real_module("research_aggregator_bot", "research_aggregator_bot.py")
    prediction_manager_bot = _load_real_module("prediction_manager_bot", "prediction_manager_bot.py")
    cognition_layer = _load_real_module("cognition_layer", "cognition_layer.py")
    menace_orchestrator = _load_real_module("menace_orchestrator", "menace_orchestrator.py")
    vector_database_service = _load_real_module(
        "vector_service.vector_database_service", "vector_service/vector_database_service.py"
    )

    deps = research_aggregator_bot._initialize_runtime(bootstrap_owner=object())

    assert deps.pipeline == broker.pipeline
    assert deps.manager == broker.manager
    assert getattr(prediction_manager_bot, "_BOOTSTRAP_PLACEHOLDER_PIPELINE", None) == broker.pipeline
    assert getattr(prediction_manager_bot, "_BOOTSTRAP_PLACEHOLDER_MANAGER", None) == broker.manager
    assert getattr(cognition_layer, "_BOOTSTRAP_PLACEHOLDER_PIPELINE", None) == broker.pipeline
    assert getattr(cognition_layer, "_BOOTSTRAP_PLACEHOLDER_MANAGER", None) == broker.manager
    assert getattr(menace_orchestrator, "_BOOTSTRAP_PLACEHOLDER", None) == (broker.pipeline, broker.manager)
    assert getattr(vector_database_service, "_BOOTSTRAP_PLACEHOLDER_PIPELINE", None) == broker.pipeline
    assert getattr(vector_database_service, "_BOOTSTRAP_PLACEHOLDER_MANAGER", None) == broker.manager

    assert prepare_calls["count"] <= 1
    assert coordinator.prepare_calls <= 1
