import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _install_stub(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    module.__file__ = str(Path(__file__).resolve())
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    sys.modules[name] = module
    return module


class _StubBroker:
    def __init__(self) -> None:
        self.sentinel = SimpleNamespace(bootstrap_placeholder=True)
        self.pipeline = SimpleNamespace(manager=self.sentinel, bootstrap_placeholder=True)
        self.calls: list[tuple[object, object, object | None]] = []

    def resolve(self) -> tuple[object, object]:
        return self.pipeline, self.sentinel

    def advertise(
        self, *, pipeline: object | None = None, sentinel: object | None = None, owner: object | None = None
    ) -> tuple[object, object]:
        if pipeline is not None:
            self.pipeline = pipeline
        if sentinel is not None:
            self.sentinel = sentinel
        self.calls.append((self.pipeline, self.sentinel, owner))
        return self.pipeline, self.sentinel


def test_bootstrap_modules_reuse_dependency_broker(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    for name in (
        "menace_sandbox.coding_bot_interface",
        "coding_bot_interface",
        "menace_sandbox.automated_reviewer",
        "automated_reviewer",
        "menace_sandbox.task_validation_bot",
        "task_validation_bot",
        "import_compat",
        "menace_sandbox.import_compat",
    ):
        sys.modules.pop(name, None)

    pkg = sys.modules.setdefault("menace_sandbox", types.ModuleType("menace_sandbox"))
    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    pkg.__file__ = str(Path(__file__).resolve().parents[1] / "__init__.py")

    broker = _StubBroker()
    # Seed the broker with an existing placeholder that downstream imports must reuse.
    broker.advertise(pipeline=broker.pipeline, sentinel=broker.sentinel, owner=True)

    prepare_calls = {"count": 0}

    def _prepare_pipeline_for_bootstrap(**_kwargs: object):
        prepare_calls["count"] += 1
        raise AssertionError(
            "prepare_pipeline_for_bootstrap should not run when a broker placeholder is already advertised"
        )

    def _advertise_bootstrap_placeholder(
        *, dependency_broker: _StubBroker | None = None, pipeline=None, manager=None, owner: bool = True
    ):
        active_broker = dependency_broker or broker
        return active_broker.advertise(
            pipeline=pipeline or active_broker.pipeline,
            sentinel=manager or active_broker.sentinel,
            owner=owner,
        )

    _install_stub(
        "coding_bot_interface",
        _BOOTSTRAP_STATE=SimpleNamespace(depth=1, helper_promotion_callbacks=[]),
        _GLOBAL_BOOTSTRAP_COORDINATOR=SimpleNamespace(peek_active=lambda: None),
        _bootstrap_dependency_broker=lambda: broker,
        advertise_bootstrap_placeholder=_advertise_bootstrap_placeholder,
        get_active_bootstrap_pipeline=broker.resolve,
        prepare_pipeline_for_bootstrap=_prepare_pipeline_for_bootstrap,
        _current_bootstrap_context=lambda: None,
        _resolve_bootstrap_wait_timeout=lambda: 0.01,
        self_coding_managed=lambda **_: (lambda cls: cls),
    )
    sys.modules["menace_sandbox.coding_bot_interface"] = sys.modules["coding_bot_interface"]

    _install_stub("bot_registry", BotRegistry=type("BotRegistry", (), {}))
    _install_stub(
        "data_bot",
        DataBot=type("DataBot", (), {"__init__": lambda self, *_, **__: None}),
        persist_sc_thresholds=lambda *_a, **_k: None,
    )
    _install_stub("self_coding_thresholds", get_thresholds=lambda *_a, **_k: SimpleNamespace())
    _install_stub("threshold_service", ThresholdService=type("ThresholdService", (), {}))
    _install_stub("code_database", CodeDB=type("CodeDB", (), {"__init__": lambda self, *_, **__: None}))
    _install_stub("menace_memory_manager", MenaceMemoryManager=type("MenaceMemoryManager", (), {}))
    _install_stub("self_coding_engine", SelfCodingEngine=type("SelfCodingEngine", (), {}))
    _install_stub(
        "self_coding_manager",
        SelfCodingManager=type("SelfCodingManager", (), {"register_bot": lambda *_: None}),
        internalize_coding_bot=lambda *_, **__: SimpleNamespace(),
        _manager_generate_helper_with_builder=lambda *_, **__: None,
    )
    _install_stub("model_automation_pipeline", ModelAutomationPipeline=type("ModelAutomationPipeline", (), {}))
    _install_stub("shared_evolution_orchestrator", get_orchestrator=lambda *_a, **_k: SimpleNamespace())
    _install_stub(
        "context_builder_util",
        create_context_builder=lambda *_, **__: SimpleNamespace(refresh_db_weights=lambda: None),
        ensure_fresh_weights=lambda *_a, **_k: None,
    )
    _install_stub("snippet_compressor", compress_snippets=lambda *_a, **_k: [])

    vector_service = _install_stub(
        "vector_service",
        CognitionLayer=type("CognitionLayer", (), {"__init__": lambda self, *_, **__: None}),
        ContextBuilder=type("ContextBuilder", (), {"__init__": lambda self, *_, **__: None}),
        FallbackResult=type("FallbackResult", (), {}),
        ErrorResult=type("ErrorResult", (Exception,), {}),
    )
    vector_service.__path__ = []

    _install_stub("bootstrap_timeout_policy", read_bootstrap_heartbeat=lambda *_a, **_k: True)
    _install_stub("self_coding_dependency_probe", ensure_self_coding_ready=lambda: (False, ["placeholder"]))
    _install_stub("synthesis_models", SynthesisTask=type("SynthesisTask", (), {}))

    simple_validation = _install_stub(
        "simple_validation",
        SimpleSchema=type("SimpleSchema", (), {}),
        fields=SimpleNamespace(Str=lambda **_: None, Int=lambda **_: None),
        ValidationError=type("ValidationError", (Exception,), {}),
    )
    sys.modules.setdefault("menace_sandbox.simple_validation", simple_validation)

    celery_stub = _install_stub("celery", Celery=type("Celery", (), {}))
    pandas_stub = _install_stub("pandas", DataFrame=type("DataFrame", (), {}))
    zmq_stub = _install_stub("zmq", Context=type("Context", (), {"instance": staticmethod(lambda: None)}), NOBLOCK=0)
    sys.modules.update(
        {
            "menace_sandbox.zmq": zmq_stub,
            "menace_sandbox.celery": celery_stub,
            "menace_sandbox.pandas": pandas_stub,
        }
    )

    import_compat = _install_stub(
        "import_compat",
        bootstrap=lambda *_a, **_k: None,
        load_internal=lambda name: sys.modules[name],
    )
    sys.modules["menace_sandbox.import_compat"] = import_compat

    automated_reviewer = importlib.import_module("menace_sandbox.automated_reviewer")
    task_validation_bot = importlib.import_module("menace_sandbox.task_validation_bot")

    assert prepare_calls["count"] == 0
    assert automated_reviewer._BOOTSTRAP_PLACEHOLDER_PIPELINE is broker.pipeline
    assert automated_reviewer._BOOTSTRAP_PLACEHOLDER_MANAGER is broker.sentinel
    assert task_validation_bot._management_cache is not None

    pipelines = {id(pipeline) for pipeline, _sentinel, _owner in broker.calls}
    sentinels = {id(sentinel) for _pipeline, sentinel, _owner in broker.calls}
    assert len(pipelines) == 1
    assert len(sentinels) == 1
    assert broker.calls, "broker should advertise the placeholder at least once"

