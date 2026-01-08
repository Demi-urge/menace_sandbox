import importlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_stub(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(f"menace_sandbox.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    module.__spec__ = importlib.util.spec_from_loader(module.__name__, loader=None)
    module.__file__ = str(Path(__file__).resolve())
    sys.modules[module.__name__] = module
    return module


def _load_module_with_stubs(monkeypatch, *, bootstrap_broker=None):
    root = Path(__file__).resolve().parents[1]
    package = ModuleType("menace_sandbox")
    package.__path__ = [str(root)]
    package.__spec__ = importlib.util.spec_from_loader(
        "menace_sandbox", loader=None, is_package=True
    )
    sys.modules["menace_sandbox"] = package

    _install_stub("bot_registry", BotRegistry=type("Registry", (), {}))
    _install_stub(
        "data_bot",
        DataBot=type("DataBot", (), {"__init__": lambda self, *_, **__: None}),
        persist_sc_thresholds=lambda *_: None,
    )

    bootstrap_state = SimpleNamespace(depth=1, helper_promotion_callbacks=[])

    class Coordinator:
        @staticmethod
        def peek_active():
            return None

    _install_stub(
        "coding_bot_interface",
        _BOOTSTRAP_STATE=bootstrap_state,
        _looks_like_pipeline_candidate=lambda obj: obj is not None,
        _bootstrap_dependency_broker=lambda: bootstrap_broker,
        advertise_bootstrap_placeholder=lambda **_: (None, None),
        read_bootstrap_heartbeat=lambda: False,
        get_active_bootstrap_pipeline=lambda: (None, None),
        _current_bootstrap_context=lambda: None,
        _using_bootstrap_sentinel=lambda *_: False,
        _peek_owner_promise=lambda *_: None,
        _GLOBAL_BOOTSTRAP_COORDINATOR=Coordinator(),
        _resolve_bootstrap_wait_timeout=lambda: 0.01,
        prepare_pipeline_for_bootstrap=lambda **_: ("prepared", lambda *_: None),
        self_coding_managed=lambda **_: (lambda cls: cls),
    )

    _install_stub(
        "self_coding_manager",
        SelfCodingManager=type("SelfCodingManager", (), {"__init__": lambda self, *_, **__: None}),
        internalize_coding_bot=lambda *_, **__: SimpleNamespace(),
    )
    _install_stub(
        "self_coding_engine",
        SelfCodingEngine=type("SelfCodingEngine", (), {"__init__": lambda self, *_, **__: None}),
    )
    _install_stub(
        "threshold_service",
        ThresholdService=type("ThresholdService", (), {"__init__": lambda self, *_, **__: None}),
    )
    _install_stub("code_database", CodeDB=type("CodeDB", (), {"__init__": lambda self, *_, **__: None}))
    _install_stub(
        "gpt_memory",
        GPTMemoryManager=type("GPTMemoryManager", (), {"__init__": lambda self, *_, **__: None}),
    )
    _install_stub("self_coding_thresholds", get_thresholds=lambda *_, **__: SimpleNamespace())
    _install_stub("shared_evolution_orchestrator", get_orchestrator=lambda *_: None)
    _install_stub("context_builder_util", create_context_builder=lambda *_, **__: SimpleNamespace())

    vector_context = ModuleType("vector_service.context_builder")
    setattr(vector_context, "ContextBuilder", type("ContextBuilder", (), {}))
    sys.modules["vector_service.context_builder"] = vector_context

    vector_service = ModuleType("vector_service")
    vector_service.ContextBuilder = getattr(vector_context, "ContextBuilder")
    vector_service.EmbeddableDBMixin = type("EmbeddableDBMixin", (), {})
    sys.modules["vector_service"] = vector_service

    _install_stub(
        "chatgpt_enhancement_bot",
        EnhancementDB=type("EnhancementDB", (), {}),
        ChatGPTEnhancementBot=type("ChatGPTEnhancementBot", (), {}),
        Enhancement=type("Enhancement", (), {}),
    )
    _install_stub(
        "chatgpt_prediction_bot",
        ChatGPTPredictionBot=type("ChatGPTPredictionBot", (), {}),
        IdeaFeatures=type("IdeaFeatures", (), {}),
    )
    _install_stub("text_research_bot", TextResearchBot=type("TextResearchBot", (), {}))
    _install_stub(
        "chatgpt_research_bot",
        ChatGPTResearchBot=type("ChatGPTResearchBot", (), {}),
        Exchange=type("Exchange", (), {}),
        summarise_text=lambda *_: "",
    )
    _install_stub("database_manager", get_connection=lambda *_: None, DB_PATH=str(root / "db"))
    _install_stub(
        "db_router",
        DBRouter=type("DBRouter", (), {}),
        GLOBAL_ROUTER=None,
        init_db_router=lambda *_: None,
    )
    _install_stub("snippet_compressor", compress_snippets=lambda *_: [])
    _install_stub("menace_db", MenaceDB=None)

    real_import = __import__

    def _stub_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("menace_sandbox"):
            if name not in sys.modules:
                stub = ModuleType(name)
                stub.__spec__ = importlib.util.spec_from_loader(name, loader=None)
                stub.__file__ = str(root)
                sys.modules[name] = stub
            return sys.modules[name]
        if globals and globals.get("__package__", "").startswith("menace_sandbox"):
            alt_name = f"menace_sandbox.{name}"
            if alt_name in sys.modules:
                return sys.modules[alt_name]
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _stub_import)

    module_name = "menace_sandbox.research_aggregator_bot"
    module_path = root / "research_aggregator_bot.py"
    spec = importlib.util.spec_from_file_location(
        module_name, module_path, submodule_search_locations=package.__path__
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _reset_runtime_state(module) -> None:
    module.registry = None
    module.data_bot = None
    module._context_builder = SimpleNamespace()
    module.engine = SimpleNamespace()
    module._PipelineCls = type("DummyPipeline", (), {})
    module.pipeline = None
    module.evolution_orchestrator = None
    module.manager = None
    module._runtime_state = None
    module._runtime_placeholder = None
    module._runtime_initializing = False


def test_bootstrap_reuses_broker_pipeline(monkeypatch):
    rab = _load_module_with_stubs(monkeypatch)

    _reset_runtime_state(rab)

    class DummyBroker:
        def __init__(self) -> None:
            self.active_pipeline = "broker-pipeline"
            self.active_sentinel = "broker-manager"
            self.active_owner = True

        def resolve(self):
            return self.active_pipeline, self.active_sentinel

        def advertise(self, pipeline=None, sentinel=None):
            self.active_pipeline = pipeline
            self.active_sentinel = sentinel

    broker = DummyBroker()

    rab._bootstrap_dependency_broker = lambda: broker
    rab._resolve_bootstrap_wait_timeout = lambda: 0.01
    rab._using_bootstrap_sentinel = lambda *_: False
    rab.read_bootstrap_heartbeat = lambda: True
    rab.get_active_bootstrap_pipeline = (
        lambda: (broker.active_pipeline, broker.active_sentinel)
    )
    rab._current_bootstrap_context = lambda: SimpleNamespace(pipeline=None, manager=None)
    rab._looks_like_pipeline_candidate = lambda obj: obj is not None

    class DummyPromise:
        done = True

        def wait(self):
            return broker.active_pipeline, lambda *_args: None

    rab._GLOBAL_BOOTSTRAP_COORDINATOR.peek_active = lambda: DummyPromise()

    prepare_called = False

    def _fail_prepare(**_kwargs):
        nonlocal prepare_called
        prepare_called = True
        raise AssertionError(
            "prepare_pipeline_for_bootstrap should not be called when bootstrap is active"
        )

    rab.prepare_pipeline_for_bootstrap = _fail_prepare

    deps = rab._initialize_runtime(bootstrap_owner=object())

    assert deps.pipeline == broker.active_pipeline
    assert deps.manager == broker.active_sentinel
    assert prepare_called is False


def test_bootstrap_placeholders_advertise_active_when_owner_missing(monkeypatch):
    module = _load_module_with_stubs(monkeypatch)

    placeholder_pipeline = object()
    placeholder_manager = object()
    broker = SimpleNamespace(active_owner=False)

    advertise_calls: list[tuple[object | None, object | None]] = []

    def _advertise(**kwargs):
        advertise_calls.append((kwargs.get("pipeline"), kwargs.get("manager")))
        broker.active_pipeline = kwargs.get("pipeline")
        broker.active_sentinel = kwargs.get("manager")
        return kwargs.get("pipeline"), kwargs.get("manager")

    module._BOOTSTRAP_PLACEHOLDER = None
    module._BOOTSTRAP_SENTINEL = None
    module._BOOTSTRAP_BROKER = None

    module.get_active_bootstrap_pipeline = lambda: (
        placeholder_pipeline,
        placeholder_manager,
    )
    module._bootstrap_dependency_broker = lambda: broker
    module.resolve_bootstrap_placeholders = lambda **_: (_ for _ in ()).throw(
        AssertionError("resolve_bootstrap_placeholders should not run")
    )
    module.advertise_bootstrap_placeholder = _advertise
    module.prepare_pipeline_for_bootstrap = lambda **_: (_ for _ in ()).throw(
        AssertionError("prepare_pipeline_for_bootstrap should not be used")
    )

    pipeline, manager, resolved_broker = module._bootstrap_placeholders()

    assert pipeline is placeholder_pipeline
    assert manager is placeholder_manager
    assert resolved_broker is broker
    assert advertise_calls == [(placeholder_pipeline, placeholder_manager)]


def test_import_allows_degraded_bootstrap_when_owner_inactive(monkeypatch):
    broker = SimpleNamespace(active_owner=False)

    module = _load_module_with_stubs(monkeypatch, bootstrap_broker=broker)

    assert module._BOOTSTRAP_PLACEHOLDER is None
    assert module._BOOTSTRAP_SENTINEL is None
    assert module._BOOTSTRAP_BROKER is broker
