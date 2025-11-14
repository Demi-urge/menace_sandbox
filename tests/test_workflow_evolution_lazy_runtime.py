from __future__ import annotations

import importlib
import logging
import sys
from collections import Counter
from types import ModuleType, SimpleNamespace


def _prepare_runtime_stubs(monkeypatch):
    counters: Counter[str] = Counter()

    def _module(name: str) -> ModuleType:
        mod = ModuleType(name)
        return mod

    bot_registry = _module("menace_sandbox.bot_registry")

    class BotRegistry:
        def __init__(self) -> None:
            counters["BotRegistry"] += 1
            self.graph = SimpleNamespace(nodes={})
            self.modules: dict[str, str] = {}

        def register_bot(self, *args, **kwargs):  # pragma: no cover - stub
            counters["register_bot"] += 1

        def update_bot(self, *args, **kwargs):  # pragma: no cover - stub
            counters["update_bot"] += 1

        def hot_swap_active(self) -> bool:  # pragma: no cover - stub
            counters["hot_swap_active"] += 1
            return False

    bot_registry.BotRegistry = BotRegistry
    monkeypatch.setitem(sys.modules, "menace_sandbox.bot_registry", bot_registry)

    data_bot = _module("menace_sandbox.data_bot")

    class DataBot:
        def __init__(self, *_, **__):
            counters["DataBot"] += 1

        def reload_thresholds(self, *_args, **_kwargs):  # pragma: no cover - stub
            counters["reload_thresholds"] += 1
            return SimpleNamespace()

    def persist_sc_thresholds(*_args, **_kwargs):  # pragma: no cover - stub
        counters["persist_sc_thresholds"] += 1

    data_bot.DataBot = DataBot
    data_bot.persist_sc_thresholds = persist_sc_thresholds
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", data_bot)

    context_builder_util = _module("menace_sandbox.context_builder_util")

    def create_context_builder():  # pragma: no cover - stub
        counters["create_context_builder"] += 1
        return SimpleNamespace()

    context_builder_util.create_context_builder = create_context_builder
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.context_builder_util", context_builder_util
    )

    self_coding_engine = _module("menace_sandbox.self_coding_engine")

    class SelfCodingEngine:
        def __init__(self, *_args, **_kwargs):
            counters["SelfCodingEngine"] += 1

    self_coding_engine.SelfCodingEngine = SelfCodingEngine
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.self_coding_engine", self_coding_engine
    )

    threshold_service = _module("menace_sandbox.threshold_service")

    class ThresholdService:
        def __init__(self):
            counters["ThresholdService"] += 1

    threshold_service.ThresholdService = ThresholdService
    monkeypatch.setitem(sys.modules, "menace_sandbox.threshold_service", threshold_service)

    code_database = _module("menace_sandbox.code_database")

    class CodeDB:
        def __init__(self):
            counters["CodeDB"] += 1

    code_database.CodeDB = CodeDB
    monkeypatch.setitem(sys.modules, "menace_sandbox.code_database", code_database)

    gpt_memory = _module("menace_sandbox.gpt_memory")

    class GPTMemoryManager:
        def __init__(self):
            counters["GPTMemoryManager"] += 1

    gpt_memory.GPTMemoryManager = GPTMemoryManager
    monkeypatch.setitem(sys.modules, "menace_sandbox.gpt_memory", gpt_memory)

    self_coding_thresholds = _module("menace_sandbox.self_coding_thresholds")

    def get_thresholds(*_args, **_kwargs):  # pragma: no cover - stub
        counters["get_thresholds"] += 1
        return SimpleNamespace(
            roi_drop=1.0,
            error_increase=0.5,
            test_failure_increase=0.25,
        )

    self_coding_thresholds.get_thresholds = get_thresholds
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.self_coding_thresholds", self_coding_thresholds
    )

    self_coding_manager = _module("menace_sandbox.self_coding_manager")

    class SelfCodingManager:  # pragma: no cover - stub
        pass

    def internalize_coding_bot(*_args, **_kwargs):  # pragma: no cover - stub
        counters["internalize_coding_bot"] += 1
        return SimpleNamespace()

    self_coding_manager.SelfCodingManager = SelfCodingManager
    self_coding_manager.internalize_coding_bot = internalize_coding_bot
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.self_coding_manager", self_coding_manager
    )

    orchestrator_mod = _module("menace_sandbox.shared_evolution_orchestrator")

    def get_orchestrator(*_args, **_kwargs):  # pragma: no cover - stub
        counters["get_orchestrator"] += 1
        return SimpleNamespace()

    orchestrator_mod.get_orchestrator = get_orchestrator
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.shared_evolution_orchestrator", orchestrator_mod
    )

    model_pipeline = _module("menace_sandbox.model_automation_pipeline")

    class ModelAutomationPipeline:
        def __init__(self, *, context_builder):
            counters["ModelAutomationPipeline"] += 1
            self.context_builder = context_builder

    model_pipeline.ModelAutomationPipeline = ModelAutomationPipeline
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.model_automation_pipeline", model_pipeline
    )

    universal_retriever = _module("menace_sandbox.universal_retriever")

    class UniversalRetriever:
        def __init__(self, *_, **__):
            counters["UniversalRetriever"] += 1

        def retrieve(self, *_args, **_kwargs):  # pragma: no cover - stub
            return []

    universal_retriever.UniversalRetriever = UniversalRetriever
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.universal_retriever", universal_retriever
    )

    intent_clusterer = _module("menace_sandbox.intent_clusterer")

    class IntentClusterer:
        def __init__(self, *_args, **_kwargs):
            counters["IntentClusterer"] += 1

        def find_modules_related_to(self, *_args, **_kwargs):  # pragma: no cover - stub
            counters["find_modules_related_to"] += 1
            return []

    intent_clusterer.IntentClusterer = IntentClusterer
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.intent_clusterer", intent_clusterer
    )

    # Ensure a clean import each time.
    sys.modules.pop("menace_sandbox.workflow_evolution_bot", None)

    return counters


def test_import_does_not_bootstrap_runtime(monkeypatch):
    counters = _prepare_runtime_stubs(monkeypatch)
    importlib.import_module("menace_sandbox.workflow_evolution_bot")

    assert counters["BotRegistry"] == 0
    assert counters["DataBot"] == 0
    assert counters["create_context_builder"] == 0
    assert counters["SelfCodingEngine"] == 0
    assert counters["ModelAutomationPipeline"] == 0
    assert counters["get_thresholds"] == 0
    assert counters["internalize_coding_bot"] == 0


def test_helper_bootstraps_runtime(monkeypatch):
    counters = _prepare_runtime_stubs(monkeypatch)
    module = importlib.import_module("menace_sandbox.workflow_evolution_bot")

    runtime = module._ensure_runtime_dependencies()
    assert counters["BotRegistry"] == 1
    assert counters["DataBot"] == 1
    assert counters["create_context_builder"] == 1
    assert counters["SelfCodingEngine"] == 1
    assert counters["ModelAutomationPipeline"] == 1
    assert counters["get_thresholds"] == 1
    assert counters["internalize_coding_bot"] == 1
    assert runtime["manager"] is not None

    # Cached results should avoid re-instantiation.
    module._ensure_runtime_dependencies()
    assert counters["BotRegistry"] == 1
    assert counters["DataBot"] == 1

    bot = module.WorkflowEvolutionBot()
    assert bot.data_bot is runtime["data_bot"]


def test_runtime_promotes_manager_without_reentrancy(monkeypatch, caplog):
    counters = _prepare_runtime_stubs(monkeypatch)
    import menace_sandbox.coding_bot_interface as coding_bot_interface

    helper_calls = 0
    promoted_managers: list[object] = []

    def _spy_prepare_pipeline_for_bootstrap(**_kwargs):
        nonlocal helper_calls
        helper_calls += 1

        def _promote(manager):
            promoted_managers.append(manager)

        dummy_pipeline = SimpleNamespace(name="WorkflowPipeline")
        return dummy_pipeline, _promote

    def _forbidden_bootstrap(*_args, **_kwargs):  # pragma: no cover - guard
        raise AssertionError("_bootstrap_manager should not be re-invoked during runtime bootstrap")

    monkeypatch.setattr(
        coding_bot_interface,
        "prepare_pipeline_for_bootstrap",
        _spy_prepare_pipeline_for_bootstrap,
    )
    monkeypatch.setattr(coding_bot_interface, "_bootstrap_manager", _forbidden_bootstrap)

    module = importlib.import_module("menace_sandbox.workflow_evolution_bot")

    emitted_managers: list[object] = []

    def _spy_internalize_coding_bot(*_args, **_kwargs):
        counters["internalize_coding_bot"] += 1
        manager = SimpleNamespace(marker="manager")
        emitted_managers.append(manager)
        return manager

    monkeypatch.setattr(module, "internalize_coding_bot", _spy_internalize_coding_bot)

    caplog.set_level(logging.WARNING)
    cached_runtime = None
    try:
        runtime = module._ensure_runtime_dependencies()
        cached_runtime = module._ensure_runtime_dependencies()
    finally:
        module._RUNTIME_CACHE = None

    assert helper_calls == 1
    assert len(emitted_managers) == 1
    assert len(promoted_managers) == 1
    assert promoted_managers[0] is emitted_managers[0]
    assert runtime["manager"] is emitted_managers[0]
    assert counters["internalize_coding_bot"] == 1
    assert not any(
        "re-entrant initialisation depth" in record.message for record in caplog.records
    )
    assert runtime["manager"] is not None
    assert cached_runtime is runtime
    assert counters["BotRegistry"] == 1
    assert counters["DataBot"] == 1
