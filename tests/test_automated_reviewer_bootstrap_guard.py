import importlib
import os
import sys
import types
from unittest import mock

import pytest

from tests.test_automated_reviewer import DummyDB, DummyEscalation


os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def _stub_vector_service(monkeypatch):
    vector_service = types.ModuleType("vector_service")

    class CognitionLayer:
        def __init__(self, *, context_builder=None, **_):
            self.context_builder = context_builder

    class ContextBuilder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def build(self, prompt, session_id=None, include_vectors=False, **_):
            return {"prompt": prompt, "session": session_id, "vectors": include_vectors}

        def refresh_db_weights(self):
            pass

    class FallbackResult:  # pragma: no cover - compatibility stub
        pass

    class ErrorResult(Exception):  # pragma: no cover - compatibility stub
        pass

    vector_service.CognitionLayer = CognitionLayer
    vector_service.ContextBuilder = ContextBuilder
    vector_service.FallbackResult = FallbackResult
    vector_service.ErrorResult = ErrorResult

    monkeypatch.setitem(sys.modules, "vector_service", vector_service)


def test_bootstrap_reuses_active_pipeline(monkeypatch):
    _stub_vector_service(monkeypatch)
    for name in (
        "menace_sandbox.model_automation_pipeline",
        "model_automation_pipeline",
        "menace_sandbox.self_coding_engine",
        "self_coding_engine",
        "menace_sandbox.menace_memory_manager",
        "menace_memory_manager",
        "menace_sandbox.self_coding_thresholds",
        "self_coding_thresholds",
        "menace_sandbox.self_coding_manager",
        "self_coding_manager",
        "menace_sandbox.threshold_service",
        "threshold_service",
        "menace_sandbox.shared_evolution_orchestrator",
        "shared_evolution_orchestrator",
        "menace_sandbox.orchestrator_loader",
        "orchestrator_loader",
        "menace_sandbox.capital_management_bot",
    ):
        sys.modules.pop(name, None)
    pipeline_mod = types.ModuleType("menace_sandbox.model_automation_pipeline")

    class ModelAutomationPipeline:
        def __init__(self, *_, **__):
            self.manager = None

    pipeline_mod.ModelAutomationPipeline = ModelAutomationPipeline
    monkeypatch.setitem(sys.modules, "menace_sandbox.model_automation_pipeline", pipeline_mod)
    monkeypatch.setitem(sys.modules, "model_automation_pipeline", pipeline_mod)

    engine_mod = types.ModuleType("menace_sandbox.self_coding_engine")

    class SelfCodingEngine:
        def __init__(self, *_, **__):
            pass

    engine_mod.SelfCodingEngine = SelfCodingEngine
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", engine_mod)
    monkeypatch.setitem(sys.modules, "self_coding_engine", engine_mod)

    memory_mod = types.ModuleType("menace_sandbox.menace_memory_manager")

    class MenaceMemoryManager:
        def __init__(self, *_, **__):
            pass

    memory_mod.MenaceMemoryManager = MenaceMemoryManager
    monkeypatch.setitem(sys.modules, "menace_sandbox.menace_memory_manager", memory_mod)
    monkeypatch.setitem(sys.modules, "menace_memory_manager", memory_mod)

    thresholds_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")
    thresholds_mod.get_thresholds = lambda *_: types.SimpleNamespace(
        roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_thresholds", thresholds_mod)
    monkeypatch.setitem(sys.modules, "self_coding_thresholds", thresholds_mod)

    manager_mod = types.ModuleType("menace_sandbox.self_coding_manager")

    class SelfCodingManager:
        def __init__(self, *_, **__):
            self.bot_name = "AutomatedReviewer"

        def register_bot(self, *_args, **_kwargs):
            return None

    manager_mod.SelfCodingManager = SelfCodingManager
    manager_mod.internalize_coding_bot = lambda *_, **__: SelfCodingManager()
    manager_mod._manager_generate_helper_with_builder = lambda *_, **__: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", manager_mod)
    monkeypatch.setitem(sys.modules, "self_coding_manager", manager_mod)

    threshold_service_mod = types.ModuleType("menace_sandbox.threshold_service")

    class ThresholdService:
        def __init__(self, *_, **__):
            pass

    threshold_service_mod.ThresholdService = ThresholdService
    monkeypatch.setitem(sys.modules, "menace_sandbox.threshold_service", threshold_service_mod)
    monkeypatch.setitem(sys.modules, "threshold_service", threshold_service_mod)

    orch_mod = types.ModuleType("menace_sandbox.shared_evolution_orchestrator")
    orch_mod.get_orchestrator = lambda *_, **__: types.SimpleNamespace(
        register_bot=lambda *a, **k: None, _ensure_degradation_subscription=lambda: None
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.shared_evolution_orchestrator", orch_mod)
    monkeypatch.setitem(sys.modules, "shared_evolution_orchestrator", orch_mod)
    orch_loader_mod = types.ModuleType("menace_sandbox.orchestrator_loader")
    orch_loader_mod.get_orchestrator = orch_mod.get_orchestrator
    monkeypatch.setitem(sys.modules, "menace_sandbox.orchestrator_loader", orch_loader_mod)
    monkeypatch.setitem(sys.modules, "orchestrator_loader", orch_loader_mod)
    import menace_sandbox.automated_reviewer as ar
    import menace_sandbox.coding_bot_interface as cbi
    import vector_service

    monkeypatch.setattr(ar, "_context_builder", None, raising=False)
    monkeypatch.setattr(ar, "_engine", None, raising=False)
    monkeypatch.setattr(ar, "_pipeline", None, raising=False)
    monkeypatch.setattr(ar, "_pipeline_promoter", None, raising=False)
    monkeypatch.setattr(ar, "_evolution_orchestrator", None, raising=False)
    monkeypatch.setattr(ar, "_thresholds", None, raising=False)
    monkeypatch.setattr(ar, "_manager_instance", None, raising=False)

    sentinel_manager = types.SimpleNamespace(
        bot_name="AutomatedReviewer", register_bot=lambda *_, **__: None, evolution_orchestrator=None
    )
    promoter_calls: list[types.SimpleNamespace] = []

    class SentinelPipeline:
        def __init__(self, manager):
            self.manager = manager

        def _pipeline_promoter(self, manager):
            promoter_calls.append(manager)

    sentinel_pipeline = SentinelPipeline(sentinel_manager)

    monkeypatch.setattr(cbi, "get_active_bootstrap_pipeline", lambda: (sentinel_pipeline, sentinel_manager))

    class DummyBroker:
        def resolve(self):
            return sentinel_pipeline, sentinel_manager

    monkeypatch.setattr(ar, "_bootstrap_dependency_broker", lambda: DummyBroker())
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1, raising=False)

    def _fail_prepare(*_, **__):  # pragma: no cover - exercised via absence of raise
        raise AssertionError("prepare_pipeline_for_bootstrap should not be called")

    monkeypatch.setattr(ar, "prepare_pipeline_for_bootstrap", _fail_prepare)
    monkeypatch.setattr(ar, "manager", sentinel_manager, raising=False)
    monkeypatch.setattr(ar.AutomatedReviewer, "manager", sentinel_manager, raising=False)

    def _fake_ensure() -> object:
        ar._pipeline = sentinel_pipeline
        ar._manager_instance = sentinel_manager
        return sentinel_manager

    monkeypatch.setattr(ar, "_ensure_self_coding_manager", _fake_ensure)

    ar._ensure_self_coding_manager()

    builder = vector_service.ContextBuilder(
        bot_db="bots.db", code_db="code.db", error_db="errors.db", workflow_db="workflows.db"
    )

    ar.AutomatedReviewer(
        context_builder=builder,
        bot_db=DummyDB(),
        escalation_manager=DummyEscalation(),
        manager=None,
    )

    assert ar._manager_instance is sentinel_manager
    assert ar._pipeline is sentinel_pipeline
    assert promoter_calls == []


def test_broker_pipeline_advertised_and_reused(monkeypatch):
    _stub_vector_service(monkeypatch)
    for name in (
        "menace_sandbox.model_automation_pipeline",
        "model_automation_pipeline",
        "menace_sandbox.self_coding_engine",
        "self_coding_engine",
        "menace_sandbox.menace_memory_manager",
        "menace_memory_manager",
        "menace_sandbox.self_coding_thresholds",
        "self_coding_thresholds",
        "menace_sandbox.self_coding_manager",
        "self_coding_manager",
        "menace_sandbox.threshold_service",
        "threshold_service",
        "menace_sandbox.shared_evolution_orchestrator",
        "shared_evolution_orchestrator",
        "menace_sandbox.orchestrator_loader",
        "orchestrator_loader",
    ):
        sys.modules.pop(name, None)

    pipeline_mod = types.ModuleType("menace_sandbox.model_automation_pipeline")

    class ModelAutomationPipeline:
        def __init__(self, *_, **__):
            self.manager = None

    pipeline_mod.ModelAutomationPipeline = ModelAutomationPipeline
    monkeypatch.setitem(sys.modules, "menace_sandbox.model_automation_pipeline", pipeline_mod)
    monkeypatch.setitem(sys.modules, "model_automation_pipeline", pipeline_mod)

    engine_mod = types.ModuleType("menace_sandbox.self_coding_engine")

    class SelfCodingEngine:
        def __init__(self, *_, **__):
            pass

    engine_mod.SelfCodingEngine = SelfCodingEngine
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", engine_mod)
    monkeypatch.setitem(sys.modules, "self_coding_engine", engine_mod)

    memory_mod = types.ModuleType("menace_sandbox.menace_memory_manager")

    class MenaceMemoryManager:
        def __init__(self, *_, **__):
            pass

    memory_mod.MenaceMemoryManager = MenaceMemoryManager
    monkeypatch.setitem(sys.modules, "menace_sandbox.menace_memory_manager", memory_mod)
    monkeypatch.setitem(sys.modules, "menace_memory_manager", memory_mod)

    thresholds_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")
    thresholds_mod.get_thresholds = lambda *_: types.SimpleNamespace(
        roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_thresholds", thresholds_mod)
    monkeypatch.setitem(sys.modules, "self_coding_thresholds", thresholds_mod)

    manager_mod = types.ModuleType("menace_sandbox.self_coding_manager")

    class SelfCodingManager:
        def __init__(self, *_, **__):
            self.bot_name = "AutomatedReviewer"

        def register_bot(self, *_args, **_kwargs):
            return None

    manager_mod.SelfCodingManager = SelfCodingManager
    manager_mod.internalize_coding_bot = lambda *_, **__: SelfCodingManager()
    manager_mod._manager_generate_helper_with_builder = lambda *_, **__: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", manager_mod)
    monkeypatch.setitem(sys.modules, "self_coding_manager", manager_mod)

    threshold_service_mod = types.ModuleType("menace_sandbox.threshold_service")

    class ThresholdService:
        def __init__(self, *_, **__):
            pass

    threshold_service_mod.ThresholdService = ThresholdService
    monkeypatch.setitem(sys.modules, "menace_sandbox.threshold_service", threshold_service_mod)
    monkeypatch.setitem(sys.modules, "threshold_service", threshold_service_mod)

    orch_mod = types.ModuleType("menace_sandbox.shared_evolution_orchestrator")
    orch_mod.get_orchestrator = lambda *_, **__: types.SimpleNamespace(
        register_bot=lambda *a, **k: None, _ensure_degradation_subscription=lambda: None
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.shared_evolution_orchestrator", orch_mod)
    monkeypatch.setitem(sys.modules, "shared_evolution_orchestrator", orch_mod)
    orch_loader_mod = types.ModuleType("menace_sandbox.orchestrator_loader")
    orch_loader_mod.get_orchestrator = orch_mod.get_orchestrator
    monkeypatch.setitem(sys.modules, "menace_sandbox.orchestrator_loader", orch_loader_mod)
    monkeypatch.setitem(sys.modules, "orchestrator_loader", orch_loader_mod)

    import menace_sandbox.automated_reviewer as ar
    import menace_sandbox.coding_bot_interface as cbi

    monkeypatch.setattr(ar, "_context_builder", None, raising=False)
    monkeypatch.setattr(ar, "_engine", None, raising=False)
    monkeypatch.setattr(ar, "_pipeline", None, raising=False)
    monkeypatch.setattr(ar, "_pipeline_promoter", None, raising=False)
    monkeypatch.setattr(ar, "_evolution_orchestrator", None, raising=False)
    monkeypatch.setattr(ar, "_thresholds", None, raising=False)
    monkeypatch.setattr(ar, "_manager_instance", None, raising=False)

    sentinel_manager = SelfCodingManager()
    promoter_calls: list[SelfCodingManager] = []

    class SentinelPipeline:
        def __init__(self, manager):
            self.manager = manager

        def _pipeline_promoter(self, manager):
            promoter_calls.append(manager)

    sentinel_pipeline = SentinelPipeline(sentinel_manager)

    advertisements: list[dict[str, object | None]] = []

    class DummyBroker:
        def resolve(self):
            return sentinel_pipeline, sentinel_manager

        def advertise(self, **kwargs):
            advertisements.append(kwargs)

    broker = DummyBroker()

    monkeypatch.setattr(ar, "prepare_pipeline_for_bootstrap", mock.Mock())
    monkeypatch.setattr(ar, "internalize_coding_bot", mock.Mock())
    monkeypatch.setattr(ar, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi, "get_active_bootstrap_pipeline", lambda: (None, None))

    resolved_manager = ar._ensure_self_coding_manager()

    assert resolved_manager is sentinel_manager
    assert ar._pipeline is sentinel_pipeline
    assert ar._pipeline_promoter is not None
    assert not ar.prepare_pipeline_for_bootstrap.called
    assert not ar.internalize_coding_bot.called
    assert advertisements == [
        {"pipeline": sentinel_pipeline, "sentinel": sentinel_manager}
    ]
    ar._pipeline_promoter(sentinel_manager)
    assert promoter_calls == [sentinel_manager]


def test_delayed_broker_advertisement_waits(monkeypatch):
    _stub_vector_service(monkeypatch)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    ar = importlib.reload(importlib.import_module("menace_sandbox.automated_reviewer"))

    ar._pipeline = None
    ar._manager_instance = None
    ar._pipeline_promoter = None

    resolve_calls = {"count": 0}

    class DummyPipeline:
        manager = None

    broker_manager = types.SimpleNamespace()
    broker_pipeline = DummyPipeline()

    def _delayed_resolve():
        resolve_calls["count"] += 1
        if resolve_calls["count"] < 3:
            return None, None
        return broker_pipeline, broker_manager

    class DummyBroker:
        def resolve(self):
            return _delayed_resolve()

        def advertise(self, *_, **__):
            return None

    monkeypatch.setattr(ar, "_bootstrap_dependency_broker", lambda: DummyBroker())
    monkeypatch.setattr(ar, "get_active_bootstrap_pipeline", lambda: (None, None))
    monkeypatch.setattr(ar, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(ar, "_GLOBAL_BOOTSTRAP_COORDINATOR", mock.Mock(peek_active=lambda: None))
    monkeypatch.setattr(ar, "_resolve_bootstrap_wait_timeout", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(ar._BOOTSTRAP_STATE, "depth", 1, raising=False)

    builder = ar.ContextBuilder()
    monkeypatch.setattr(ar, "create_context_builder", lambda: builder)
    monkeypatch.setattr(ar, "SelfCodingEngine", lambda *_a, **_k: mock.Mock())
    monkeypatch.setattr(ar, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(ar, "MenaceMemoryManager", mock.Mock(name="MenaceMemoryManager"))
    monkeypatch.setattr(ar, "get_orchestrator", lambda *_a, **_k: mock.Mock())
    monkeypatch.setattr(
        ar,
        "get_thresholds",
        lambda *_a, **_k: types.SimpleNamespace(
            roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0
        ),
    )
    monkeypatch.setattr(ar, "persist_sc_thresholds", lambda *_, **__: None)
    monkeypatch.setattr(ar, "ThresholdService", lambda *_, **__: mock.Mock())
    monkeypatch.setattr(ar, "internalize_coding_bot", lambda *_, **__: broker_manager)
    monkeypatch.setattr(ar, "prepare_pipeline_for_bootstrap", mock.Mock())

    manager = ar._ensure_self_coding_manager()

    assert manager is broker_manager
    assert ar._pipeline is broker_pipeline
    assert resolve_calls["count"] >= 3

