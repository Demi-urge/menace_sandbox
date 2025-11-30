import importlib
import inspect
import sys
from importlib.util import spec_from_loader
from pathlib import Path
from types import ModuleType
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import logging

import pytest

import coding_bot_interface as cbi


@pytest.mark.parametrize(
    "module_names",
    [
        (
            "menace_sandbox.research_aggregator_bot",
            "menace_sandbox.prediction_manager_bot",
            "menace_sandbox.cognition_layer",
            "menace_sandbox.menace_orchestrator",
            "menace_sandbox.vector_service.vector_database_service",
        )
    ],
)
def test_placeholder_promise_reused_during_concurrent_imports(
    module_names, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    package = ModuleType("menace_sandbox")
    package.__path__ = []
    package.__spec__ = spec_from_loader("menace_sandbox", loader=None, is_package=True)
    sys.modules.setdefault("menace_sandbox", package)
    sys.modules.setdefault("menace_sandbox.coding_bot_interface", cbi)

    shared_broker = cbi._BootstrapDependencyBroker()
    placeholder_manager = SimpleNamespace(bootstrap_placeholder=True)
    placeholder_pipeline = cbi._build_bootstrap_placeholder_pipeline(placeholder_manager)
    shared_broker.advertise(
        pipeline=placeholder_pipeline, sentinel=placeholder_manager, owner=True
    )

    def _install_stub(name: str, **attrs: object) -> ModuleType:
        module = ModuleType(f"menace_sandbox.{name}")
        for key, value in attrs.items():
            setattr(module, key, value)
        module.__spec__ = spec_from_loader(module.__name__, loader=None)
        module.__file__ = str(Path(__file__).resolve())
        sys.modules[module.__name__] = module
        return module

    _install_stub("bot_registry", BotRegistry=type("Registry", (), {}))
    _install_stub(
        "data_bot",
        DataBot=type("DataBot", (), {"__init__": lambda self, *_, **__: None}),
        MetricsDB=type("MetricsDB", (), {}),
        MetricRecord=type("MetricRecord", (), {}),
        persist_sc_thresholds=lambda *_: None,
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
        threshold_service=lambda *_: None,
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
    vector_context.__spec__ = spec_from_loader("vector_service.context_builder", loader=None)
    sys.modules["vector_service.context_builder"] = vector_context

    vector_service = ModuleType("vector_service")
    vector_service.ContextBuilder = getattr(vector_context, "ContextBuilder")
    vector_service.EmbeddableDBMixin = type("EmbeddableDBMixin", (), {})
    vector_service.__spec__ = spec_from_loader("vector_service", loader=None, is_package=True)
    vector_service.__path__ = []
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
    _install_stub("database_manager", get_connection=lambda *_: None, DB_PATH=str(Path(__file__).resolve()))
    _install_stub(
        "db_router",
        DBRouter=type("DBRouter", (), {}),
        GLOBAL_ROUTER=None,
        init_db_router=lambda *_: None,
        LOCAL_TABLES=(),
    )
    _install_stub("snippet_compressor", compress_snippets=lambda *_: [])
    _install_stub("menace_db", MenaceDB=None)

    def _install_placeholder_module(name: str) -> ModuleType:
        module_name = f"menace_sandbox.{name}"
        sys.modules.pop(module_name, None)
        parent = module_name.rsplit(".", 1)[0]
        sys.modules.pop(parent, None)
        module = ModuleType(f"menace_sandbox.{name}")
        module.__spec__ = spec_from_loader(module.__name__, loader=None)
        (
            module._BOOTSTRAP_PLACEHOLDER_PIPELINE,
            module._BOOTSTRAP_PLACEHOLDER_MANAGER,
        ) = cbi.advertise_bootstrap_placeholder(
            dependency_broker=shared_broker,
            pipeline=shared_broker.active_pipeline,
            manager=shared_broker.active_sentinel,
        )
        sys.modules[module.__name__] = module
        return module

    for advertised in (
        "research_aggregator_bot",
        "prediction_manager_bot",
        "cognition_layer",
        "menace_orchestrator",
        "vector_service.vector_database_service",
    ):
        _install_placeholder_module(advertised)

    placeholder_cache: dict[str, ModuleType] = {
        name: sys.modules[name] for name in module_names
    }

    real_import_module = importlib.import_module

    def _fake_import(name: str, package: str | None = None):
        if name in placeholder_cache:
            return placeholder_cache[name]
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: shared_broker)
    monkeypatch.setattr(
        cbi, "get_active_bootstrap_pipeline", lambda: (placeholder_pipeline, placeholder_manager)
    )

    promise = cbi._BootstrapPipelinePromise()
    promise.resolve((placeholder_pipeline, lambda *_: None))
    stub_coordinator = SimpleNamespace(
        peek_active=lambda: promise,
        claim=lambda: (True, promise),
        settle=lambda *_, **__: None,
    )
    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", stub_coordinator)

    prepare_calls: list[str] = []

    def _fail_prepare(**_: object) -> tuple[object, object]:
        caller = inspect.stack()[1]
        prepare_calls.append(f"{caller.filename}:{caller.lineno}")
        raise AssertionError(
            "prepare_pipeline_for_bootstrap should not run when a placeholder is active"
        )

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _fail_prepare)

    def _import_module(name: str) -> object:
        sys.modules.pop(name, None)
        if name.startswith("menace_sandbox.vector_service"):
            sys.modules.pop("menace_sandbox.vector_service", None)
            sys.modules.pop("vector_service", None)
        return importlib.import_module(name)

    with ThreadPoolExecutor(max_workers=len(module_names)) as executor:
        list(executor.map(_import_module, module_names))

    assert shared_broker.active_pipeline is placeholder_pipeline
    assert shared_broker.active_sentinel is placeholder_manager
    assert prepare_calls == []
    assert not any("recursion" in record.getMessage() for record in caplog.records)
