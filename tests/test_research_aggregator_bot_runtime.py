"""Tests for lazy runtime initialisation in :mod:`research_aggregator_bot`."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


MODULE_NAME = "menace_sandbox.research_aggregator_bot"


def _import_fresh() -> object:
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _stub_module(monkeypatch, name: str, attrs: dict[str, object]) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(autouse=True)
def stub_heavy_dependencies(monkeypatch):
    class _EnhancementDB:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self) -> list[object]:
            return []

    class _ChatGPTEnhancementBot:
        def __init__(self, *args, **kwargs):
            pass

    class _ChatGPTPredictionBot:
        def __init__(self, *args, **kwargs):
            pass

    class _IdeaFeatures:
        pass

    class _TextResearchBot:
        def __init__(self, *args, **kwargs):
            pass

    class _VideoResearchBot:
        def __init__(self, *args, **kwargs):
            pass

    class _ChatGPTResearchBot:
        def __init__(self, *args, **kwargs):
            pass

    class _Exchange:
        pass

    class _CapitalManagementBot:
        def info_ratio(self, value: float) -> float:
            return float(value)

    class _DBRouter:
        def __init__(self, *args, **kwargs):
            pass

        def insert_info(self, *_args, **_kwargs) -> None:
            pass

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            pass

        def build(self, query: str) -> list[str]:
            return [query]

    def _init_db_router(*_args, **_kwargs) -> _DBRouter:
        return _DBRouter()

    def _summarise_text(text: str) -> str:
        return text

    def _compress_snippets(snippets: dict[str, str]) -> dict[str, str]:
        return snippets

    def _record_failed_tags(*_args, **_kwargs) -> None:
        return None

    def _load_failed_tags(*_args, **_kwargs) -> list[str]:
        return []

    stub_vs = _stub_module(
        monkeypatch,
        "vector_service",
        {
            "EmbeddableDBMixin": object,
            "ContextBuilder": _ContextBuilder,
        },
    )
    _stub_module(
        monkeypatch,
        "vector_service.context_builder",
        {
            "ContextBuilder": _ContextBuilder,
            "record_failed_tags": _record_failed_tags,
            "load_failed_tags": _load_failed_tags,
        },
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.chatgpt_enhancement_bot",
        {
            "EnhancementDB": _EnhancementDB,
            "ChatGPTEnhancementBot": _ChatGPTEnhancementBot,
            "Enhancement": object,
        },
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.chatgpt_prediction_bot",
        {
            "ChatGPTPredictionBot": _ChatGPTPredictionBot,
            "IdeaFeatures": _IdeaFeatures,
        },
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.text_research_bot",
        {"TextResearchBot": _TextResearchBot},
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.video_research_bot",
        {"VideoResearchBot": _VideoResearchBot},
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.chatgpt_research_bot",
        {
            "ChatGPTResearchBot": _ChatGPTResearchBot,
            "Exchange": _Exchange,
            "summarise_text": _summarise_text,
        },
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.capital_management_bot",
        {"CapitalManagementBot": _CapitalManagementBot},
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.db_router",
        {
            "DBRouter": _DBRouter,
            "GLOBAL_ROUTER": None,
            "init_db_router": _init_db_router,
        },
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.snippet_compressor",
        {"compress_snippets": _compress_snippets},
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.run_autonomous",
        {"LOCAL_KNOWLEDGE_MODULE": object()},
    )
    return stub_vs


def test_module_import_is_lightweight():
    """Ensure heavy constructors are not triggered during import."""

    with mock.patch("menace_sandbox.bot_registry.BotRegistry") as mock_registry, \
        mock.patch("menace_sandbox.data_bot.DataBot") as mock_data_bot:
        module = _import_fresh()

    assert mock_registry.call_count == 0
    assert mock_data_bot.call_count == 0
    assert module.registry is None
    assert module.data_bot is None
    assert module.manager is None


def test_ensure_runtime_dependencies_bootstraps_once(monkeypatch):
    """``_ensure_runtime_dependencies`` instantiates helpers lazily and caches."""

    with mock.patch("menace_sandbox.bot_registry.BotRegistry") as mock_registry, \
        mock.patch("menace_sandbox.data_bot.DataBot") as mock_data_bot:
        registry_instance = mock.Mock(name="registry")
        data_bot_instance = mock.Mock(name="data_bot")
        mock_registry.return_value = registry_instance
        mock_data_bot.return_value = data_bot_instance
        module = _import_fresh()

    fake_builder = mock.Mock(name="context_builder")
    fake_engine = mock.Mock(name="engine")
    fake_orchestrator = mock.Mock(name="orchestrator")
    fake_pipeline_instances: list[object] = []

    class FakePipeline:
        def __init__(self, *, context_builder: object) -> None:  # pragma: no cover - simple
            fake_pipeline_instances.append(context_builder)

    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )
    fake_manager = mock.Mock(name="manager")
    fake_threshold_service = mock.Mock(name="threshold_service")

    def fake_self_coding_managed(*, bot_registry, data_bot, manager=None):
        call_args.append((bot_registry, data_bot, manager))

        def decorator(cls):
            decorated_classes.append(cls)
            return cls

        return decorator

    call_args: list[tuple[object, object, object | None]] = []
    decorated_classes: list[type] = []

    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=FakePipeline))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    persist_mock = mock.Mock()
    monkeypatch.setattr(module, "persist_sc_thresholds", persist_mock)
    monkeypatch.setattr(module, "internalize_coding_bot", mock.Mock(return_value=fake_manager))
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=fake_threshold_service))
    monkeypatch.setattr(module, "self_coding_managed", fake_self_coding_managed)

    state = module._ensure_runtime_dependencies()

    assert state.registry is registry_instance
    assert state.data_bot is data_bot_instance
    assert state.context_builder is fake_builder
    assert state.engine is fake_engine
    assert state.pipeline is not None
    assert fake_pipeline_instances == [fake_builder]
    assert state.evolution_orchestrator is fake_orchestrator
    assert state.manager is fake_manager

    assert module.registry is registry_instance
    assert module.data_bot is data_bot_instance
    assert module.manager is fake_manager
    assert module._runtime_state is state

    persist_mock.assert_called_once_with(
        "ResearchAggregatorBot",
        roi_drop=fake_thresholds.roi_drop,
        error_increase=fake_thresholds.error_increase,
        test_failure_increase=fake_thresholds.test_failure_increase,
    )
    assert module.internalize_coding_bot.call_count == 1
    assert call_args == [(registry_instance, data_bot_instance, fake_manager)]
    assert decorated_classes and decorated_classes[0].__name__ == "ResearchAggregatorBot"

    module._ensure_runtime_dependencies()
    assert mock_registry.call_count == 1
    assert mock_data_bot.call_count == 1
    assert module.internalize_coding_bot.call_count == 1
    assert len(call_args) == 1
    assert len(decorated_classes) == 1
