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


def test_bootstrap_owner_reuses_injected_pipeline(monkeypatch):
    """A bootstrap sentinel lets callers reuse an inflight pipeline."""

    owner = object()
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
    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )
    fake_manager = mock.Mock(name="manager")
    fake_pipeline = mock.Mock(name="pipeline")
    promote = mock.Mock(name="promote")

    _stub_module(
        monkeypatch,
        "menace_sandbox.coding_bot_interface",
        {
            "get_structural_bootstrap_owner": mock.Mock(return_value=owner),
            "get_active_bootstrap_pipeline": mock.Mock(return_value=(None, None)),
        },
    )
    module._bootstrap_pipeline_cache[owner] = (fake_pipeline, promote, None)
    module.prepare_pipeline_for_bootstrap = mock.Mock()

    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=object))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "internalize_coding_bot", mock.Mock(return_value=fake_manager))
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(module, "self_coding_managed", lambda **_k: (lambda c: c))

    state = module._ensure_runtime_dependencies(bootstrap_owner=owner)

    assert state.pipeline is fake_pipeline
    assert module.prepare_pipeline_for_bootstrap.call_count == 0
    promote.assert_called_once_with(fake_manager)


def test_bootstrap_sentinel_reuses_existing_pipeline(monkeypatch):
    """A bootstrap sentinel prevents nested pipeline creation."""

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
    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )
    fake_pipeline = mock.Mock(name="pipeline")
    sentinel_manager = SimpleNamespace(pipeline=fake_pipeline, bootstrap_mode=True)

    get_owner = mock.Mock(name="get_structural_bootstrap_owner")
    _stub_module(
        monkeypatch,
        "menace_sandbox.coding_bot_interface",
        {
            "get_structural_bootstrap_owner": get_owner,
            "get_active_bootstrap_pipeline": mock.Mock(return_value=(None, None)),
        },
    )

    module.prepare_pipeline_for_bootstrap = mock.Mock()
    monkeypatch.setattr(module, "_using_bootstrap_sentinel", lambda *_a, **_k: True)
    monkeypatch.setattr(
        module, "_looks_like_pipeline_candidate", lambda value: value is fake_pipeline
    )
    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=object))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "internalize_coding_bot", mock.Mock(return_value=sentinel_manager))
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(module, "self_coding_managed", lambda **_k: (lambda c: c))

    state = module._ensure_runtime_dependencies(manager_override=sentinel_manager)

    assert state.pipeline is fake_pipeline
    module.prepare_pipeline_for_bootstrap.assert_not_called()
    get_owner.assert_not_called()


def test_bootstrap_guard_reuses_pipeline_and_manager(monkeypatch):
    """An active guard-provided pipeline suppresses fresh bootstrap calls."""

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
    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )
    fake_pipeline = mock.Mock(name="pipeline")
    guard_manager = SimpleNamespace(pipeline=fake_pipeline)

    _stub_module(
        monkeypatch,
        "menace_sandbox.coding_bot_interface",
        {
            "get_structural_bootstrap_owner": mock.Mock(return_value=None),
            "get_active_bootstrap_pipeline": mock.Mock(
                return_value=(fake_pipeline, guard_manager)
            ),
        },
    )

    module.prepare_pipeline_for_bootstrap = mock.Mock()
    monkeypatch.setattr(
        module, "_looks_like_pipeline_candidate", lambda value: value is fake_pipeline
    )
    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=object))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    internalizer = mock.Mock(name="internalize_coding_bot")
    monkeypatch.setattr(module, "internalize_coding_bot", internalizer)
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(module, "self_coding_managed", lambda **_k: (lambda c: c))

    state = module._ensure_runtime_dependencies()

    assert state.pipeline is fake_pipeline
    assert state.manager is guard_manager
    module.prepare_pipeline_for_bootstrap.assert_not_called()
    internalizer.assert_not_called()


def test_bootstrap_owner_caches_pipeline_for_followups(monkeypatch):
    """The cached promotion callback is reused while the sentinel is active."""

    owner = object()
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
    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )
    promote = mock.Mock(name="promote")

    _stub_module(
        monkeypatch,
        "menace_sandbox.coding_bot_interface",
        {
            "get_structural_bootstrap_owner": mock.Mock(return_value=owner),
            "get_active_bootstrap_pipeline": mock.Mock(return_value=(None, None)),
        },
    )

    fake_pipeline = mock.Mock(name="pipeline")
    module.prepare_pipeline_for_bootstrap = mock.Mock(return_value=(fake_pipeline, promote))

    managers = [mock.Mock(name="manager_one"), mock.Mock(name="manager_two")]
    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=object))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "internalize_coding_bot", mock.Mock(side_effect=managers))
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(module, "self_coding_managed", lambda **_k: (lambda c: c))

    state_one = module._ensure_runtime_dependencies(bootstrap_owner=owner)
    assert state_one.pipeline is fake_pipeline
    assert module.prepare_pipeline_for_bootstrap.call_count == 1

    # Reset runtime globals but keep the bootstrap cache entry seeded above.
    module._runtime_state = None
    module._runtime_placeholder = None
    module._runtime_initializing = False
    module.pipeline = None
    module.manager = None
    module._self_coding_configured = False

    state_two = module._ensure_runtime_dependencies(bootstrap_owner=owner)

    assert state_two.pipeline is fake_pipeline
    assert module.prepare_pipeline_for_bootstrap.call_count == 1
    assert promote.call_count == 2
    assert promote.call_args_list[0][0][0] is managers[0]
    assert promote.call_args_list[1][0][0] is managers[1]


def test_nested_bootstrap_guard_reuses_active_pipeline(monkeypatch):
    """Guard-protected dependents must reuse the active bootstrap pipeline."""

    guard_manager = None
    guard_pipeline = mock.Mock(name="guard_pipeline")
    guard_promote = mock.Mock(name="guard_promote")

    with mock.patch("menace_sandbox.bot_registry.BotRegistry") as mock_registry, \
        mock.patch("menace_sandbox.data_bot.DataBot") as mock_data_bot:
        registry_instance = mock.Mock(name="registry")
        data_bot_instance = mock.Mock(name="data_bot")
        mock_registry.return_value = registry_instance
        mock_data_bot.return_value = data_bot_instance
        module = _import_fresh()

    module._BOOTSTRAP_STATE = SimpleNamespace(
        helper_promotion_callbacks=[guard_promote]
    )
    monkeypatch.setattr(
        module,
        "_bootstrap_dependency_broker",
        mock.Mock(return_value=SimpleNamespace(resolve=lambda: (None, None))),
    )
    monkeypatch.setattr(
        module,
        "get_active_bootstrap_pipeline",
        mock.Mock(return_value=(guard_pipeline, guard_manager)),
    )
    monkeypatch.setattr(
        module, "_looks_like_pipeline_candidate", lambda value: bool(value)
    )
    monkeypatch.setattr(module, "_using_bootstrap_sentinel", lambda *_a, **_k: False)

    fake_builder = mock.Mock(name="context_builder")
    fake_engine = mock.Mock(name="engine")
    fake_orchestrator = mock.Mock(name="orchestrator")
    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )
    manager_result = mock.Mock(name="manager_result")

    module.prepare_pipeline_for_bootstrap = mock.Mock(
        side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")
    )
    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=object))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "internalize_coding_bot", mock.Mock(return_value=manager_result))
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(module, "self_coding_managed", lambda **_k: (lambda c: c))

    state = module._ensure_runtime_dependencies()

    assert state.pipeline is guard_pipeline
    guard_promote.assert_called_once_with(manager_result)
    module.prepare_pipeline_for_bootstrap.assert_not_called()


def test_dependency_broker_reuses_inflight_pipeline(monkeypatch):
    """Dependents fall back to the advertised broker pipeline during bootstrap."""

    broker_manager = mock.Mock(name="broker_manager")
    broker_pipeline = mock.Mock(name="broker_pipeline")
    broker_promote = mock.Mock(name="broker_promote")

    with mock.patch("menace_sandbox.bot_registry.BotRegistry") as mock_registry, \
        mock.patch("menace_sandbox.data_bot.DataBot") as mock_data_bot:
        registry_instance = mock.Mock(name="registry")
        data_bot_instance = mock.Mock(name="data_bot")
        mock_registry.return_value = registry_instance
        mock_data_bot.return_value = data_bot_instance
        module = _import_fresh()

    broker = SimpleNamespace(resolve=lambda: (broker_pipeline, broker_manager))
    module._BOOTSTRAP_STATE = SimpleNamespace(helper_promotion_callbacks=[broker_promote])
    monkeypatch.setattr(
        module, "_bootstrap_dependency_broker", mock.Mock(return_value=broker)
    )
    monkeypatch.setattr(
        module,
        "get_active_bootstrap_pipeline",
        mock.Mock(return_value=(None, None)),
    )
    monkeypatch.setattr(
        module, "_looks_like_pipeline_candidate", lambda value: bool(value)
    )
    monkeypatch.setattr(module, "_using_bootstrap_sentinel", lambda *_a, **_k: False)

    fake_builder = mock.Mock(name="context_builder")
    fake_engine = mock.Mock(name="engine")
    fake_orchestrator = mock.Mock(name="orchestrator")
    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )

    module.prepare_pipeline_for_bootstrap = mock.Mock(
        side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")
    )
    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=object))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "internalize_coding_bot", mock.Mock(return_value=broker_manager))
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(module, "self_coding_managed", lambda **_k: (lambda c: c))

    state = module._ensure_runtime_dependencies()

    assert state.pipeline is broker_pipeline
    broker_promote.assert_called_once_with(broker_manager)
    module.prepare_pipeline_for_bootstrap.assert_not_called()


def test_bootstrap_guard_promise_waits_for_pipeline(monkeypatch):
    """A guard promise blocks fresh bootstrap attempts until the pipeline appears."""

    guard_token = object()
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
    fake_thresholds = SimpleNamespace(
        roi_drop=1.0,
        error_increase=2.0,
        test_failure_increase=3.0,
    )

    class FakeModelAutomationPipeline:
        def __init__(self) -> None:
            self.context_builder = fake_builder
            self.manager = mock.Mock(name="manager")
            self._bot_attribute_order: tuple[str, ...] = ()

    fake_pipeline = FakeModelAutomationPipeline()
    broker_calls: list[int] = []

    class _Broker:
        def resolve(self) -> tuple[object | None, object | None]:
            broker_calls.append(len(broker_calls))
            if len(broker_calls) > 1:
                return fake_pipeline, fake_pipeline.manager
            return None, None

    module._BOOTSTRAP_STATE = SimpleNamespace(active_bootstrap_guard=guard_token)
    module._peek_owner_promise = lambda *_args, **_kwargs: object()
    module._resolve_bootstrap_wait_timeout = lambda *_args, **_kwargs: 0.05
    module._bootstrap_dependency_broker = lambda: _Broker()
    module._current_bootstrap_context = lambda: None
    module.prepare_pipeline_for_bootstrap = mock.Mock(
        side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")
    )

    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=fake_builder))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=fake_engine))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(name="CodeDB"))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(name="GPTMemoryManager"))
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=FakeModelAutomationPipeline))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=fake_orchestrator))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=fake_thresholds))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "internalize_coding_bot", mock.Mock(return_value=fake_pipeline.manager))
    monkeypatch.setattr(module, "ThresholdService", mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(module, "self_coding_managed", lambda **_k: (lambda c: c))

    state = module._ensure_runtime_dependencies()

    assert len(broker_calls) >= 2
    assert state.pipeline is fake_pipeline
    assert state.manager is fake_pipeline.manager
    module.prepare_pipeline_for_bootstrap.assert_not_called()
