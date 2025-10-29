from __future__ import annotations

import importlib
import sys
import contextlib
from types import ModuleType, SimpleNamespace


def test_research_aggregator_lazy_capital_manager(monkeypatch):
    original_cap_module = sys.modules.get("menace_sandbox.capital_management_bot")
    original_ra_module = sys.modules.get("menace_sandbox.research_aggregator_bot")
    original_plain_ra = sys.modules.get("research_aggregator_bot")
    original_modules: dict[str, ModuleType | None] = {}

    def set_stub(name: str, module: ModuleType) -> None:
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    class DummyRegistry:
        def __init__(self, *args, **kwargs) -> None:
            self.graph = SimpleNamespace(nodes={})

    class DummyDataBot:
        def __init__(self, *args, **kwargs) -> None:
            pass

    def _decorator_factory(**kwargs):
        def _decorator(cls):
            return cls

        return _decorator

    class DummySelfCodingManager:
        pass

    def _internalize(*args, **kwargs) -> DummySelfCodingManager:
        return DummySelfCodingManager()

    class DummySelfCodingEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class DummyThresholdService:
        pass

    class DummyCodeDB:
        pass

    class DummyMemoryManager:
        pass

    def _get_thresholds(name: str) -> SimpleNamespace:
        return SimpleNamespace(roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0)

    class DummyContextBuilder:
        def refresh_db_weights(self) -> None:  # noqa: D401 - simple stub
            return None

        def build(self, query: str) -> str:
            return query

    def _create_context_builder() -> DummyContextBuilder:
        return DummyContextBuilder()

    def _get_orchestrator(*args, **kwargs) -> None:
        return None

    class DummyEnhancementDB:
        pass

    class DummyEnhancementBot:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class DummyEnhancement:
        pass

    class DummyPredictionBot:
        pass

    class DummyIdeaFeatures:
        pass

    class DummyTextBot:
        pass

    class DummyVideoBot:
        pass

    class DummyChatGPTBot:
        pass

    class DummyExchange:
        pass

    def _summarise_text(text: str) -> str:
        return text

    def _get_connection(_path: str):
        dummy = SimpleNamespace(
            execute=lambda *args, **kwargs: SimpleNamespace(
                fetchall=lambda: [], fetchone=lambda: None
            )
        )
        return contextlib.nullcontext(dummy)

    class DummyDBRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_connection(self, *args, **kwargs):  # pragma: no cover - simple stub
            return SimpleNamespace(row_factory=None, execute=lambda *a, **k: SimpleNamespace(fetchall=lambda: []))

    class DummyEmbeddableDBMixin:
        def __init__(self, *args, **kwargs) -> None:
            pass

    def _compress_snippets(snippets: dict[str, str]) -> dict[str, str]:
        return snippets

    def _auto_link(_mapping: dict[str, str]):
        def _decorator(target):
            return target

        return _decorator

    class DummyEventBus:
        pass

    stub_mappings = {
        "menace_sandbox.bot_registry": ModuleType("menace_sandbox.bot_registry"),
        "menace_sandbox.data_bot": ModuleType("menace_sandbox.data_bot"),
        "menace_sandbox.coding_bot_interface": ModuleType(
            "menace_sandbox.coding_bot_interface"
        ),
        "menace_sandbox.self_coding_manager": ModuleType(
            "menace_sandbox.self_coding_manager"
        ),
        "menace_sandbox.self_coding_engine": ModuleType(
            "menace_sandbox.self_coding_engine"
        ),
        "menace_sandbox.threshold_service": ModuleType(
            "menace_sandbox.threshold_service"
        ),
        "menace_sandbox.code_database": ModuleType("menace_sandbox.code_database"),
        "menace_sandbox.gpt_memory": ModuleType("menace_sandbox.gpt_memory"),
        "menace_sandbox.self_coding_thresholds": ModuleType(
            "menace_sandbox.self_coding_thresholds"
        ),
        "vector_service.context_builder": ModuleType("vector_service.context_builder"),
        "menace_sandbox.shared_evolution_orchestrator": ModuleType(
            "menace_sandbox.shared_evolution_orchestrator"
        ),
        "context_builder_util": ModuleType("context_builder_util"),
        "menace_sandbox.chatgpt_enhancement_bot": ModuleType(
            "menace_sandbox.chatgpt_enhancement_bot"
        ),
        "menace_sandbox.chatgpt_prediction_bot": ModuleType(
            "menace_sandbox.chatgpt_prediction_bot"
        ),
        "menace_sandbox.text_research_bot": ModuleType(
            "menace_sandbox.text_research_bot"
        ),
        "menace_sandbox.video_research_bot": ModuleType(
            "menace_sandbox.video_research_bot"
        ),
        "menace_sandbox.chatgpt_research_bot": ModuleType(
            "menace_sandbox.chatgpt_research_bot"
        ),
        "menace_sandbox.database_manager": ModuleType(
            "menace_sandbox.database_manager"
        ),
        "menace_sandbox.db_router": ModuleType("menace_sandbox.db_router"),
        "vector_service": ModuleType("vector_service"),
        "snippet_compressor": ModuleType("snippet_compressor"),
        "menace_sandbox.menace_db": ModuleType("menace_sandbox.menace_db"),
        "menace_sandbox.auto_link": ModuleType("menace_sandbox.auto_link"),
        "menace_sandbox.unified_event_bus": ModuleType(
            "menace_sandbox.unified_event_bus"
        ),
    }

    stub_mappings["menace_sandbox.bot_registry"].BotRegistry = DummyRegistry
    stub_mappings["menace_sandbox.data_bot"].DataBot = DummyDataBot
    stub_mappings["menace_sandbox.data_bot"].persist_sc_thresholds = lambda *a, **k: None
    stub_mappings["menace_sandbox.coding_bot_interface"].self_coding_managed = _decorator_factory
    stub_mappings["menace_sandbox.self_coding_manager"].SelfCodingManager = DummySelfCodingManager
    stub_mappings["menace_sandbox.self_coding_manager"].internalize_coding_bot = _internalize
    stub_mappings["menace_sandbox.self_coding_engine"].SelfCodingEngine = DummySelfCodingEngine
    stub_mappings["menace_sandbox.threshold_service"].ThresholdService = DummyThresholdService
    stub_mappings["menace_sandbox.code_database"].CodeDB = DummyCodeDB
    stub_mappings["menace_sandbox.gpt_memory"].GPTMemoryManager = DummyMemoryManager
    stub_mappings["menace_sandbox.self_coding_thresholds"].get_thresholds = _get_thresholds
    stub_mappings["vector_service.context_builder"].ContextBuilder = DummyContextBuilder
    stub_mappings["menace_sandbox.shared_evolution_orchestrator"].get_orchestrator = _get_orchestrator
    stub_mappings["context_builder_util"].create_context_builder = _create_context_builder
    stub_mappings["menace_sandbox.chatgpt_enhancement_bot"].EnhancementDB = DummyEnhancementDB
    stub_mappings["menace_sandbox.chatgpt_enhancement_bot"].ChatGPTEnhancementBot = DummyEnhancementBot
    stub_mappings["menace_sandbox.chatgpt_enhancement_bot"].Enhancement = DummyEnhancement
    stub_mappings["menace_sandbox.chatgpt_prediction_bot"].ChatGPTPredictionBot = DummyPredictionBot
    stub_mappings["menace_sandbox.chatgpt_prediction_bot"].IdeaFeatures = DummyIdeaFeatures
    stub_mappings["menace_sandbox.text_research_bot"].TextResearchBot = DummyTextBot
    stub_mappings["menace_sandbox.video_research_bot"].VideoResearchBot = DummyVideoBot
    stub_mappings["menace_sandbox.chatgpt_research_bot"].ChatGPTResearchBot = DummyChatGPTBot
    stub_mappings["menace_sandbox.chatgpt_research_bot"].Exchange = DummyExchange
    stub_mappings["menace_sandbox.chatgpt_research_bot"].summarise_text = _summarise_text
    stub_mappings["menace_sandbox.database_manager"].DB_PATH = ""
    stub_mappings["menace_sandbox.database_manager"].get_connection = _get_connection
    stub_mappings["menace_sandbox.db_router"].DBRouter = DummyDBRouter
    stub_mappings["menace_sandbox.db_router"].GLOBAL_ROUTER = None
    stub_mappings["menace_sandbox.db_router"].init_db_router = lambda *a, **k: None
    stub_mappings["vector_service"].EmbeddableDBMixin = DummyEmbeddableDBMixin
    stub_mappings["vector_service"].ContextBuilder = DummyContextBuilder
    stub_mappings["snippet_compressor"].compress_snippets = _compress_snippets
    stub_mappings["menace_sandbox.menace_db"].MenaceDB = None
    stub_mappings["menace_sandbox.auto_link"].auto_link = _auto_link
    stub_mappings["menace_sandbox.unified_event_bus"].UnifiedEventBus = DummyEventBus

    for name, module in stub_mappings.items():
        set_stub(name, module)

    for name in ("menace_sandbox.research_aggregator_bot", "research_aggregator_bot"):
        sys.modules.pop(name, None)

    original_import_module = importlib.import_module
    attempted = False

    def raising_import_module(name: str, package: str | None = None):
        nonlocal attempted
        if name == "menace_sandbox.capital_management_bot":
            attempted = True
            raise RuntimeError("capital import blocked")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", raising_import_module)

    module = None
    try:
        module = importlib.import_module("menace_sandbox.research_aggregator_bot")
        assert attempted is False

        class DummyCapital:
            pass

        def stub_import_module(name: str, package: str | None = None):
            if name == "menace_sandbox.capital_management_bot":
                mod = SimpleNamespace(CapitalManagementBot=DummyCapital)
                sys.modules[name] = mod
                return mod
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", stub_import_module)

        capital_cls = module._get_capital_manager_class()
        assert capital_cls is DummyCapital
        assert isinstance(capital_cls(), DummyCapital)
    finally:
        for name, original in original_modules.items():
            if original is not None:
                sys.modules[name] = original
            else:
                sys.modules.pop(name, None)
        if module is not None:
            module._CapitalManagerCls = None  # type: ignore[attr-defined]
        if original_cap_module is not None:
            sys.modules["menace_sandbox.capital_management_bot"] = original_cap_module
        else:
            sys.modules.pop("menace_sandbox.capital_management_bot", None)
        if original_ra_module is not None:
            sys.modules["menace_sandbox.research_aggregator_bot"] = original_ra_module
        else:
            sys.modules.pop("menace_sandbox.research_aggregator_bot", None)
        if original_plain_ra is not None:
            sys.modules["research_aggregator_bot"] = original_plain_ra
        else:
            sys.modules.pop("research_aggregator_bot", None)
