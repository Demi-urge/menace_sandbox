from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import menace_sandbox.bot_registry as bot_registry
import menace_sandbox.coding_bot_interface as coding_bot_interface


@pytest.fixture(autouse=True)
def disable_unmanaged_scan(monkeypatch):
    """Prevent background scanner threads during tests."""

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "schedule_unmanaged_scan",
        lambda self, interval=3600.0: None,
    )
    yield


def _make_registry() -> bot_registry.BotRegistry:
    return bot_registry.BotRegistry(event_bus=None)


def test_internal_self_coding_modules_not_transient():
    exc = ModuleNotFoundError(
        "No module named 'menace_sandbox.quick_fix_engine'",
        name="menace_sandbox.quick_fix_engine",
    )
    assert not bot_registry._is_transient_internalization_error(exc)

    exc_top_level = ModuleNotFoundError(
        "No module named 'quick_fix_engine'",
        name="quick_fix_engine",
    )
    assert not bot_registry._is_transient_internalization_error(exc_top_level)


def test_register_bot_records_module_path_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "schedule_unmanaged_scan",
        lambda self, interval=3600.0: None,
    )
    registry = bot_registry.BotRegistry(event_bus=None)

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    def _raise(*_args, **_kwargs):
        raise bot_registry.SelfCodingUnavailableError(
            "self-coding bootstrap failed",
            missing=("quick_fix_engine",),
        )

    monkeypatch.setattr(registry, "_internalize_missing_coding_bot", _raise)

    scheduled: list[tuple[str, float | None]] = []

    def _schedule(
        name: str,
        *,
        delay: float | None = None,
        force: bool = False,
    ) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    module_path = tmp_path / "example_bot.py"
    module_path.write_text("# stub\n", encoding="utf-8")

    registry.register_bot(
        "ExampleBot",
        module_path=module_path,
        is_coding_bot=True,
    )

    assert scheduled, "internalisation retry should be scheduled"
    registry._retry_internalization("ExampleBot")

    node = registry.graph.nodes["ExampleBot"]
    assert node["module"] == str(module_path)
    assert registry.modules["ExampleBot"] == str(module_path)
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["missing_dependencies"] == ["quick_fix_engine"]


def test_register_bot_marks_self_coding_disabled_when_dependencies_missing(monkeypatch):
    monkeypatch.setattr(
        bot_registry,
        "ensure_self_coding_ready",
        lambda modules=None: (False, ("sklearn", "pydantic")),
    )

    registry = _make_registry()
    registry.register_bot("TaskValidationBot", is_coding_bot=True)

    node = registry.graph.nodes["TaskValidationBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert sorted(disabled["missing_dependencies"]) == ["pydantic", "sklearn"]
    assert disabled["reason"].startswith("self-coding dependencies unavailable")
    assert node.get("pending_internalization") is False


def test_manual_registration_clears_stale_self_coding_state(monkeypatch):
    registry = _make_registry()

    name = "TaskValidationBot"
    registry.graph.add_node(name)
    node = registry.graph.nodes[name]
    node.update(
        {
            "pending_internalization": True,
            "internalization_errors": ["boom"],
            "internalization_blocked": {"error": "boom"},
            "selfcoding_manager": SimpleNamespace(),
            "manager": SimpleNamespace(),
            "data_bot": SimpleNamespace(),
        }
    )
    registry._internalization_retry_attempts[name] = 3

    class _Handle:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    handle = _Handle()
    registry._internalization_retry_handles[name] = handle

    registry.register_bot(
        name,
        module_path="C:/bots/task_validation_bot.py",
        is_coding_bot=False,
    )

    assert registry._internalization_retry_attempts.get(name) is None
    assert name not in registry._internalization_retry_handles
    assert handle.cancelled is True
    assert node.get("pending_internalization") is None
    assert node.get("internalization_errors") is None
    assert node.get("internalization_blocked") is None
    assert node.get("selfcoding_manager") is None
    assert node.get("manager") is None
    assert node.get("data_bot") is None
    assert node.get("is_coding_bot") is False


def test_register_bot_handles_bootstrap_import_failures(monkeypatch):
    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    def _raise_components():
        raise bot_registry.SelfCodingUnavailableError(
            "self-coding bootstrap failed",
            missing=("torch", "numpy"),
        )

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_components",
        lambda: _raise_components(),
    )

    registry = _make_registry()
    scheduled: list[tuple[str, float | None]] = []

    def _schedule(
        name: str,
        *,
        delay: float | None = None,
        force: bool = False,
    ) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    registry.register_bot("FutureProfitabilityBot", is_coding_bot=True)

    assert scheduled, "internalisation retry should be scheduled"
    registry._retry_internalization("FutureProfitabilityBot")

    node = registry.graph.nodes["FutureProfitabilityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["missing_dependencies"] == ["numpy", "torch"]


def test_pipeline_manager_promotion_finalizes_decorated_bots(monkeypatch):
    monkeypatch.setattr(coding_bot_interface, "_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(
        coding_bot_interface,
        "ensure_self_coding_ready",
        lambda modules=None: (True, ()),
    )

    class _BrokenManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise TypeError("fallback bootstrap")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _BrokenManager,
    )

    pipeline_bots: list[Any] = []

    class _DummyCodeDB:
        def __init__(self) -> None:
            pass

    class _DummyMemory:
        def __init__(self) -> None:
            pass

    class _DummyEngine:
        def __init__(self, *_args: Any, context_builder: Any = None) -> None:
            self.context_builder = context_builder

    class _DummyPipeline:
        def __init__(
            self,
            *,
            context_builder: Any,
            bot_registry: Any,
            manager: Any,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.manager = manager
            self._bots = list(pipeline_bots)
            for bot in self._bots:
                try:
                    setattr(bot, "manager", manager)
                except Exception:
                    pass
            self.finalized = False
            self.registry_seen = None
            self.data_bot_seen = None

        def _finalize_self_coding_bootstrap(
            self,
            promoted_manager: Any,
            *,
            registry: Any | None = None,
            data_bot: Any | None = None,
        ) -> None:
            self.manager = promoted_manager
            self.finalized = True
            self.registry_seen = registry
            self.data_bot_seen = data_bot

    class _DummyManager:
        def __init__(
            self,
            engine: Any,
            pipeline: Any,
            *,
            bot_name: str,
            data_bot: Any,
            bot_registry: Any,
        ) -> None:
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    original_loader = coding_bot_interface._load_optional_module

    def _fake_loader(name: str, *, fallback: str | None = None) -> Any:
        mapping = {
            "code_database": SimpleNamespace(CodeDB=_DummyCodeDB),
            "gpt_memory": SimpleNamespace(GPTMemoryManager=_DummyMemory),
            "self_coding_engine": SimpleNamespace(SelfCodingEngine=_DummyEngine),
            "model_automation_pipeline": SimpleNamespace(
                ModelAutomationPipeline=_DummyPipeline
            ),
            "self_coding_manager": SimpleNamespace(SelfCodingManager=_DummyManager),
        }
        if name in mapping:
            return mapping[name]
        return original_loader(name, fallback=fallback)

    monkeypatch.setattr(coding_bot_interface, "_load_optional_module", _fake_loader)
    monkeypatch.setattr(
        coding_bot_interface, "create_context_builder", lambda: SimpleNamespace()
    )

    original_bootstrap = coding_bot_interface._bootstrap_manager
    call_counter = {"count": 0}

    def _guarded_bootstrap(name: str, registry: Any, data_bot: Any) -> Any:
        call_counter["count"] += 1
        if call_counter["count"] > 1:
            raise AssertionError("_bootstrap_manager invoked multiple times")
        return original_bootstrap(name, registry, data_bot)

    monkeypatch.setattr(coding_bot_interface, "_bootstrap_manager", _guarded_bootstrap)

    class _DummyRegistry:
        def register_bot(self, name: str, module_path: str = "", **kwargs: Any) -> None:
            pass

        def update_bot(self, name: str, module_path: str = "", **kwargs: Any) -> None:
            pass

    class _Thresholds:
        roi_drop = 0
        error_threshold = 0
        test_failure_threshold = 0

    class _DummyDataBot:
        def reload_thresholds(self, name: str) -> Any:
            return _Thresholds

    registry = _DummyRegistry()
    data_bot = _DummyDataBot()

    pipeline_bots.clear()

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class SentinelBot:
        name = "SentinelBot"

        def __init__(self) -> None:
            pass

    pipeline_bots.append(SentinelBot)

    bot = SentinelBot()
    assert isinstance(bot.manager, _DummyManager)
    assert SentinelBot.manager is bot.manager
    assert SentinelBot.bot_registry is registry
    assert SentinelBot.data_bot is data_bot
    assert SentinelBot._self_coding_manual_mode is False
    assert callable(getattr(SentinelBot, "_finalize_self_coding_bootstrap"))

    pipeline = bot.manager.pipeline
    assert pipeline.manager is bot.manager
    assert pipeline.finalized is True
    assert pipeline.registry_seen is registry
    assert pipeline.data_bot_seen is data_bot

    second = SentinelBot()
    assert second.manager is bot.manager
    assert call_counter["count"] == 1
    assert SentinelBot._self_coding_manual_mode is False


def test_bootstrap_manager_promotes_pipeline_manager(monkeypatch):
    registry = _make_registry()
    data_bot = SimpleNamespace()

    class _LegacyManager:
        def __init__(self, engine, pipeline):
            self.engine = engine
            self.pipeline = pipeline

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _LegacyManager,
    )

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder

    class _StubCommsBot:
        manager = None

        def __init__(self, manager):
            self.manager = manager
            type(self).manager = manager

    class _StubPipeline:
        def __init__(self, *, context_builder, bot_registry, manager):
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.manager = manager
            self.initial_manager = manager
            self.comms_bot = _StubCommsBot(manager)
            self._bots = [self.comms_bot]
            self.reattach_calls = 0

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

    class _StubManager:
        def __init__(
            self,
            engine,
            pipeline,
            *,
            bot_name,
            data_bot,
            bot_registry,
        ) -> None:
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    stubs = {
        "code_database": SimpleNamespace(CodeDB=_StubCodeDB),
        "gpt_memory": SimpleNamespace(GPTMemoryManager=_StubMemory),
        "self_coding_engine": SimpleNamespace(SelfCodingEngine=_StubEngine),
        "model_automation_pipeline": SimpleNamespace(
            ModelAutomationPipeline=_StubPipeline
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubManager),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        return stubs[name]

    monkeypatch.setattr(
        coding_bot_interface, "_load_optional_module", _load_optional_module
    )

    manager = coding_bot_interface._bootstrap_manager("ExampleBot", registry, data_bot)

    assert isinstance(manager, _StubManager)
    pipeline = manager.pipeline
    assert isinstance(pipeline.initial_manager, coding_bot_interface._DisabledSelfCodingManager)
    assert pipeline.initial_manager is not manager
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert type(pipeline.comms_bot).manager is manager
    assert pipeline._bots[0].manager is manager
    assert pipeline.reattach_calls >= 1


def test_lazy_pipeline_bootstrap_does_not_trigger_reentrant_warning(monkeypatch, caplog):
    from menace_sandbox import bot_testing_bot

    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = SimpleNamespace()
    data_bot = SimpleNamespace()
    created: dict[str, Any] = {}

    class _StubCodeDB:
        pass

    class _StubMemoryManager:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None):
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    class _StubPipeline:
        def __init__(self, *, context_builder, bot_registry=None, data_bot=None, manager=None):
            created.setdefault("pipelines", []).append(self)
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            sentinel_bot = SimpleNamespace(manager=manager)
            self._bots = [sentinel_bot, SimpleNamespace(manager=manager)]
            self.nested = SimpleNamespace(manager=manager)

    original_get_module_attr = bot_testing_bot._get_module_attr

    def _get_module_attr(module, attr, module_name):
        if attr == "ModelAutomationPipeline":
            return _StubPipeline
        return original_get_module_attr(module, attr, module_name)

    monkeypatch.setattr(bot_testing_bot, "_get_module_attr", _get_module_attr, raising=False)
    monkeypatch.setattr(bot_testing_bot, "_get_context_builder", lambda: SimpleNamespace(), raising=False)
    monkeypatch.setattr(bot_testing_bot, "_get_registry", lambda: registry, raising=False)
    monkeypatch.setattr(bot_testing_bot, "_get_data_bot", lambda: data_bot, raising=False)
    monkeypatch.setattr(bot_testing_bot, "_pipeline_promoter", None, raising=False)
    bot_testing_bot._get_pipeline.cache_clear()

    def _fake_loader(name: str, *, fallback=None):
        if name == "code_database":
            return SimpleNamespace(CodeDB=_StubCodeDB)
        if name == "gpt_memory":
            return SimpleNamespace(GPTMemoryManager=_StubMemoryManager)
        if name == "self_coding_engine":
            return SimpleNamespace(SelfCodingEngine=_StubEngine)
        if name == "model_automation_pipeline":
            bot_testing_bot._get_pipeline.cache_clear()
            created["lazy_helper_pipeline"] = bot_testing_bot._get_pipeline()
            return SimpleNamespace(ModelAutomationPipeline=_StubPipeline)
        if name == "self_coding_manager":
            return SimpleNamespace(SelfCodingManager=_StubManager)
        raise AssertionError(f"unexpected optional module {name}")

    class _FailingManager:
        def __init__(self, *, bot_registry, data_bot):
            raise TypeError("use fallback")

    monkeypatch.setattr(coding_bot_interface, "_load_optional_module", _fake_loader)
    monkeypatch.setattr(coding_bot_interface, "create_context_builder", lambda: SimpleNamespace())
    monkeypatch.setattr(coding_bot_interface, "_resolve_self_coding_manager_cls", lambda: _FailingManager)

    manager = coding_bot_interface._bootstrap_manager("LazyHelperBot", registry, data_bot)

    assert "re-entrant" not in caplog.text
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.manager is manager
    assert all(getattr(bot, "manager", None) is manager for bot in getattr(pipeline, "_bots", []))
    assert created["pipelines"][-1] is pipeline


def test_bootstrap_manager_manager_cls_promotes_pipeline(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = _make_registry()
    data_bot = SimpleNamespace()

    created: dict[str, Any] = {}

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _PipelineBot:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            created.setdefault("bots", []).append(self)

    class _SentinelPipeline:
        def __init__(self, *, bot_registry, data_bot, manager):
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            bot = _PipelineBot()
            self._bots = [bot]
            self.initial_bot_manager = bot.manager
            self.reattach_calls = 0

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

    class _StubManager:
        def __init__(self, *, bot_registry, data_bot):
            context = coding_bot_interface._current_bootstrap_context()
            if context is None:
                raise AssertionError("bootstrap context should be available")
            sentinel = context.manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.pipeline = _SentinelPipeline(
                bot_registry=bot_registry,
                data_bot=data_bot,
                manager=sentinel,
            )

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _StubManager,
    )

    manager = coding_bot_interface._bootstrap_manager("PipelineBot", registry, data_bot)

    assert isinstance(manager, _StubManager)
    assert not any("re-entrant" in record.message.lower() for record in caplog.records)

    pipeline = manager.pipeline
    assert isinstance(pipeline.initial_manager, coding_bot_interface._BootstrapManagerSentinel)
    assert pipeline.initial_manager is pipeline.initial_bot_manager
    assert pipeline.manager is manager
    assert all(bot.manager is manager for bot in pipeline._bots)
    assert created["bots"]


def test_bootstrap_manager_handles_communication_bot_in_legacy_pipeline(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = _make_registry()
    data_bot = SimpleNamespace()

    class _LegacyManager:
        def __init__(self, engine, pipeline):
            self.engine = engine
            self.pipeline = pipeline

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _LegacyManager,
    )

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _PatchedCommunicationMaintenanceBot:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.error_bot = SimpleNamespace(manager=manager)

    comms_module = SimpleNamespace(
        CommunicationMaintenanceBot=_PatchedCommunicationMaintenanceBot
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.communication_maintenance_bot",
        comms_module,
    )

    class _StubPipeline:
        def __init__(self, *, context_builder, bot_registry, manager):
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.manager = manager
            self.initial_manager = manager
            self.comms_bot = comms_module.CommunicationMaintenanceBot(
                manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            bots = [self.comms_bot, getattr(self.comms_bot, "error_bot", None)]
            self._bots = [bot for bot in bots if bot is not None]
            self.reattach_calls = 0

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

    modules = {
        "code_database": SimpleNamespace(CodeDB=_StubCodeDB),
        "gpt_memory": SimpleNamespace(GPTMemoryManager=_StubMemory),
        "self_coding_engine": SimpleNamespace(SelfCodingEngine=_StubEngine),
        "model_automation_pipeline": SimpleNamespace(
            ModelAutomationPipeline=_StubPipeline
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubManager),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        return modules[name]

    monkeypatch.setattr(
        coding_bot_interface, "_load_optional_module", _load_optional_module
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: SimpleNamespace(),
    )

    manager = coding_bot_interface._bootstrap_manager(
        "CommunicationMaintenanceBot",
        registry,
        data_bot,
    )

    assert isinstance(manager, _StubManager)
    assert not any("re-entrant" in record.message.lower() for record in caplog.records)

    pipeline = manager.pipeline
    assert pipeline.initial_manager is not manager
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.error_bot.manager is manager
    assert all(getattr(bot, "manager", None) is manager for bot in pipeline._bots)


def test_bootstrap_manager_promotes_eager_pipeline_helpers(monkeypatch, caplog):
    """Pipeline bootstrap should promote nested helpers instantiated without a manager."""

    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = _make_registry()
    thresholds = SimpleNamespace(
        roi_drop=0.0,
        error_threshold=0.0,
        test_failure_threshold=0.0,
    )
    data_bot = SimpleNamespace(
        reload_thresholds=lambda name: thresholds,
        check_degradation=lambda *args, **kwargs: None,
        subscribe_degradation=lambda callback: None,
        event_bus=None,
    )

    monkeypatch.setattr(
        coding_bot_interface,
        "_DEFERRED_SENTINEL_CALLBACKS",
        set(),
        raising=False,
    )

    helper_name = "StubCommunicationMaintenanceBot"
    registry.graph.add_node(helper_name, is_coding_bot=True)

    attach_counter = {"count": 0}
    original_attach = coding_bot_interface._BootstrapManagerSentinel.attach_delegate

    def _tracking_attach(self, real_manager):
        attach_counter["count"] += 1
        return original_attach(self, real_manager)

    monkeypatch.setattr(
        coding_bot_interface._BootstrapManagerSentinel,
        "attach_delegate",
        _tracking_attach,
        raising=False,
    )

    class _FailingManager:
        def __init__(self, *, bot_registry, data_bot):
            raise TypeError("use fallback pipeline bootstrap")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "ensure_self_coding_ready",
        lambda modules=None: (True, ()),
    )

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _StubCommunicationMaintenanceBot:
        name = helper_name

        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    comms_module = SimpleNamespace(
        CommunicationMaintenanceBot=_StubCommunicationMaintenanceBot
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.communication_maintenance_bot",
        comms_module,
    )

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    class _UserSpacePipeline:
        def __init__(self, *, context_builder, bot_registry, data_bot, manager):
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            self.comms_bot = comms_module.CommunicationMaintenanceBot(
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            self.comms_initial_manager = self.comms_bot.manager
            self._bots = [self.comms_bot]

    modules = {
        "code_database": SimpleNamespace(CodeDB=_StubCodeDB),
        "gpt_memory": SimpleNamespace(GPTMemoryManager=_StubMemory),
        "self_coding_engine": SimpleNamespace(SelfCodingEngine=_StubEngine),
        "model_automation_pipeline": SimpleNamespace(
            ModelAutomationPipeline=_UserSpacePipeline
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubManager),
        "quick_fix_engine": SimpleNamespace(
            QuickFixEngine=lambda *args, **kwargs: SimpleNamespace()
        ),
        "error_bot": SimpleNamespace(ErrorDB=lambda: SimpleNamespace()),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        if name in modules:
            return modules[name]
        raise AssertionError(f"unexpected optional module {name}")

    monkeypatch.setattr(
        coding_bot_interface,
        "_load_optional_module",
        _load_optional_module,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: SimpleNamespace(),
    )

    manager = coding_bot_interface._bootstrap_manager(
        "UserSpacePipelineBot", registry, data_bot
    )

    assert attach_counter["count"] == 1
    assert not any("re-entrant" in record.message.lower() for record in caplog.records)
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.manager is manager
    assert isinstance(
        pipeline.initial_manager, coding_bot_interface._BootstrapManagerSentinel
    )
    assert pipeline.initial_manager is pipeline.comms_initial_manager
    assert pipeline.comms_bot.manager is manager

    helper_node = registry.graph.nodes[helper_name]
    assert helper_node["selfcoding_manager"] is manager
    assert helper_node["manager"] is manager
    assert helper_node["selfcoding_manager"] is not pipeline.initial_manager


def test_bootstrap_manager_fallback_injects_bootstrap_owner(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = _make_registry()
    thresholds = SimpleNamespace(
        roi_drop=0.0,
        error_threshold=0.0,
        test_failure_threshold=0.0,
    )
    data_bot = SimpleNamespace(
        reload_thresholds=lambda name: thresholds,
        check_degradation=lambda *args, **kwargs: None,
        subscribe_degradation=lambda callback: None,
        event_bus=None,
    )

    class _FailingManager:
        def __init__(self, *, bot_registry, data_bot):
            raise TypeError("use fallback bootstrap")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "ensure_self_coding_ready",
        lambda modules=None: (True, ()),
    )

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _InlineHelper:
        name = "InlineHelper"

        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.initial_manager = manager
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    class _InlinePipeline:
        def __init__(self, *, context_builder, bot_registry, data_bot, manager=None):
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            self.helper = _InlineHelper(
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            self._bots = [self.helper]

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    modules = {
        "code_database": SimpleNamespace(CodeDB=_StubCodeDB),
        "gpt_memory": SimpleNamespace(GPTMemoryManager=_StubMemory),
        "self_coding_engine": SimpleNamespace(SelfCodingEngine=_StubEngine),
        "model_automation_pipeline": SimpleNamespace(
            ModelAutomationPipeline=_InlinePipeline
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubManager),
        "quick_fix_engine": SimpleNamespace(
            QuickFixEngine=lambda *args, **kwargs: SimpleNamespace()
        ),
        "error_bot": SimpleNamespace(ErrorDB=lambda: SimpleNamespace()),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        if name in modules:
            return modules[name]
        raise AssertionError(f"unexpected optional module {name}")

    monkeypatch.setattr(
        coding_bot_interface,
        "_load_optional_module",
        _load_optional_module,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: SimpleNamespace(),
    )

    manager = coding_bot_interface._bootstrap_manager(
        "InlineHelperBot", registry, data_bot
    )

    assert not any(
        "re-entrant" in record.message.lower() for record in caplog.records
    )
    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    helper = pipeline.helper
    assert helper.manager is manager
    assert getattr(pipeline.initial_manager, "_bootstrap_owner_marker", False) is True

def test_bootstrap_manager_fallback_honors_sentinel_during_nested_pipeline_init(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = _make_registry()
    data_bot = SimpleNamespace()

    class _FailingManager:
        def __init__(self, *, bot_registry, data_bot):
            raise TypeError("use legacy fallback")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _SentinelCommunicationBot:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.initial_manager = manager
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.error_bot = SimpleNamespace(manager=manager)

    comms_module = SimpleNamespace(
        CommunicationMaintenanceBot=_SentinelCommunicationBot
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.communication_maintenance_bot",
        comms_module,
    )

    class _StubPipeline:
        def __init__(self, *, context_builder, bot_registry, data_bot, manager):
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            self.comms_bot = comms_module.CommunicationMaintenanceBot(
                manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            self.initial_comms_manager = self.comms_bot.initial_manager
            bots = [self.comms_bot, getattr(self.comms_bot, "error_bot", None)]
            self._bots = [bot for bot in bots if bot is not None]
            self.reattach_calls = 0

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

    modules = {
        "code_database": SimpleNamespace(CodeDB=_StubCodeDB),
        "gpt_memory": SimpleNamespace(GPTMemoryManager=_StubMemory),
        "self_coding_engine": SimpleNamespace(SelfCodingEngine=_StubEngine),
        "model_automation_pipeline": SimpleNamespace(
            ModelAutomationPipeline=_StubPipeline
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubManager),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        return modules[name]

    monkeypatch.setattr(
        coding_bot_interface, "_load_optional_module", _load_optional_module
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: SimpleNamespace(),
    )

    original_bootstrap = coding_bot_interface._bootstrap_manager
    call_count = 0

    def _spy_bootstrap_manager(name: str, bot_registry, data_bot):
        nonlocal call_count
        call_count += 1
        return original_bootstrap(name, bot_registry, data_bot)

    monkeypatch.setattr(
        coding_bot_interface,
        "_bootstrap_manager",
        _spy_bootstrap_manager,
    )

    manager = coding_bot_interface._bootstrap_manager(
        "CommunicationMaintenanceBot",
        registry,
        data_bot,
    )

    assert call_count == 1
    assert isinstance(manager, _StubManager)
    assert not any("re-entrant" in record.message.lower() for record in caplog.records)

    pipeline = manager.pipeline
    assert isinstance(
        pipeline.initial_manager, coding_bot_interface._BootstrapManagerSentinel
    )
    assert isinstance(
        pipeline.initial_comms_manager, coding_bot_interface._BootstrapManagerSentinel
    )
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.error_bot.manager is manager
    assert all(getattr(bot, "manager", None) is manager for bot in pipeline._bots)


def test_bootstrap_manager_fallback_avoids_reentrant_disabled_manager(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    registry = _make_registry()
    data_bot = SimpleNamespace(
        reload_thresholds=lambda name: SimpleNamespace(
            roi_drop=0.1, error_threshold=0.05
        )
    )

    class _LegacySelfCodingManager:
        def __init__(self, engine, *, bot_registry, data_bot):  # pragma: no cover - stub
            self.engine = engine
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _LegacySelfCodingManager,
    )
    monkeypatch.setattr(coding_bot_interface, "_self_coding_runtime_available", lambda: True)
    monkeypatch.setattr(
        coding_bot_interface,
        "ensure_self_coding_ready",
        lambda: (True, ()),
    )

    created: dict[str, object] = {}

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _NestedBot:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    class _StubCodeDB:
        pass

    class _StubMemoryManager:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None):
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    class _StubPipeline:
        def __init__(self, *, context_builder, bot_registry, manager):
            created["pipeline"] = self
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.manager = manager
            sentinel_bot = SimpleNamespace(manager=manager)
            self.nested = _NestedBot(manager=manager)
            self._bots = [sentinel_bot, self.nested]

    def _fake_loader(name: str, *, fallback=None):
        if name == "code_database":
            return SimpleNamespace(CodeDB=_StubCodeDB)
        if name == "gpt_memory":
            return SimpleNamespace(GPTMemoryManager=_StubMemoryManager)
        if name == "self_coding_engine":
            return SimpleNamespace(SelfCodingEngine=_StubEngine)
        if name == "model_automation_pipeline":
            return SimpleNamespace(ModelAutomationPipeline=_StubPipeline)
        if name == "self_coding_manager":
            return SimpleNamespace(SelfCodingManager=_StubManager)
        raise AssertionError(f"unexpected optional module {name}")

    monkeypatch.setattr(coding_bot_interface, "_load_optional_module", _fake_loader)
    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: SimpleNamespace(),
    )

    manager = coding_bot_interface._bootstrap_manager("NestedBot", registry, data_bot)

    assert not any("re-entrant" in record.message.lower() for record in caplog.records)
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is created.get("pipeline")
    assert pipeline is not None
    assert pipeline.manager is manager
    assert all(getattr(bot, "manager", None) is manager for bot in pipeline._bots)


def test_bootstrap_manager_fallback_promotes_registry_entries(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = _make_registry()
    thresholds = SimpleNamespace(roi_drop=0.0, error_threshold=0.0, test_failure_threshold=0.0)
    data_bot = SimpleNamespace(
        reload_thresholds=lambda name: thresholds,
        check_degradation=lambda name, roi, errors, test_failures: None,
        subscribe_degradation=lambda callback: None,
        event_bus=None,
    )

    class _LegacySelfCodingManager:
        def __init__(self, engine, *, bot_registry, data_bot):
            self.engine = engine
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _LegacySelfCodingManager,
    )
    monkeypatch.setattr(coding_bot_interface, "_self_coding_runtime_available", lambda: True)
    monkeypatch.setattr(
        coding_bot_interface,
        "ensure_self_coding_ready",
        lambda: (True, ()),
    )

    sentinel = coding_bot_interface._create_bootstrap_manager_sentinel(
        bot_registry=registry,
        data_bot=data_bot,
    )
    registry.register_bot(
        "NestedBot",
        manager=sentinel,
        data_bot=data_bot,
        is_coding_bot=True,
    )
    registry.register_bot(
        "ChildBot",
        manager=sentinel,
        data_bot=data_bot,
        is_coding_bot=True,
    )

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _RootBot:
        name = "NestedBot"

        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    @coding_bot_interface.self_coding_managed(
        bot_registry=lambda: registry,
        data_bot=lambda: data_bot,
    )
    class _NestedBot:
        name = "ChildBot"

        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None):
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    def _manager_helper(*_args, **_kwargs):
        return None

    class _StubPipeline:
        def __init__(self, *, context_builder, bot_registry, data_bot, manager):
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.root = _RootBot(manager=manager)
            self.child = _NestedBot(manager=manager)
            self._bots = [self.root, self.child]

    modules = {
        "code_database": SimpleNamespace(CodeDB=_StubCodeDB),
        "gpt_memory": SimpleNamespace(GPTMemoryManager=_StubMemory),
        "self_coding_engine": SimpleNamespace(SelfCodingEngine=_StubEngine),
        "model_automation_pipeline": SimpleNamespace(ModelAutomationPipeline=_StubPipeline),
        "self_coding_manager": SimpleNamespace(
            SelfCodingManager=_StubManager,
            _manager_generate_helper_with_builder=_manager_helper,
        ),
        "quick_fix_engine": SimpleNamespace(QuickFixEngine=lambda *a, **k: SimpleNamespace()),
        "error_bot": SimpleNamespace(ErrorDB=lambda: SimpleNamespace()),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        if name in modules:
            return modules[name]
        raise AssertionError(f"unexpected optional module {name}")

    monkeypatch.setattr(
        coding_bot_interface,
        "_load_optional_module",
        _load_optional_module,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: SimpleNamespace(),
    )

    manager = coding_bot_interface._bootstrap_manager("NestedBot", registry, data_bot)

    assert isinstance(manager, _StubManager)
    assert not any("re-entrant" in record.message.lower() for record in caplog.records)

    node = registry.graph.nodes["NestedBot"]
    assert node["selfcoding_manager"] is manager
    assert node["selfcoding_manager"] is not sentinel
    assert not isinstance(node["selfcoding_manager"], coding_bot_interface._BootstrapManagerSentinel)

    child_node = registry.graph.nodes["ChildBot"]
    assert child_node["selfcoding_manager"] is manager
    assert child_node["selfcoding_manager"] is not sentinel
    assert not isinstance(
        child_node["selfcoding_manager"],
        coding_bot_interface._BootstrapManagerSentinel,
    )


def test_bootstrap_manager_logs_reentrant_warning(monkeypatch, caplog):
    monkeypatch.setattr(coding_bot_interface._BOOTSTRAP_STATE, "depth", 1, raising=False)

    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    manager = coding_bot_interface._bootstrap_manager(
        "ExampleBot",
        bot_registry=SimpleNamespace(),
        data_bot=SimpleNamespace(),
    )

    assert not manager
    expected = (
        "SelfCodingManager bootstrap skipped for ExampleBot: "
        "re-entrant initialisation depth=1. "
        "Re-entrant bootstrap detected; returning disabled manager temporarily"
        "—internalisation will retry after ExampleBot completes."
    )
    assert expected in caplog.text


def test_transient_import_errors_eventually_disable_self_coding(monkeypatch):
    registry = _make_registry()

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )
    monkeypatch.setattr(bot_registry, "_collect_missing_modules", lambda exc: set())

    def _boom(*_a, **_k):  # pragma: no cover - monkeypatched in test
        raise ImportError(
            "cannot import name 'Helper' from partially initialized module 'menace.foo'"
        )

    scheduled: list[tuple[str, float | None]] = []

    monkeypatch.setattr(
        registry,
        "_internalize_missing_coding_bot",
        lambda *a, **k: _boom(),
    )
    def _schedule(
        name: str,
        *,
        delay: float | None = None,
        force: bool = False,
    ) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    registry.register_bot("FutureLucrativityBot", is_coding_bot=True)

    # simulate background retries until the registry disables self-coding
    for _ in range(5):
        if scheduled:
            registry._retry_internalization("FutureLucrativityBot")
        node = registry.graph.nodes["FutureLucrativityBot"]
        if node.get("self_coding_disabled"):
            break

    node = registry.graph.nodes["FutureLucrativityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["transient_error"]["repeat_count"] == registry._max_transient_error_signature_repeats
    assert (
        disabled["transient_error"]["total_repeat_count"]
        == registry._max_transient_error_signature_repeats
    )
    assert disabled["transient_error"]["unique_signatures"] == 1
    assert node.get("pending_internalization") is False
    assert "internalization_blocked" in node


def test_missing_modules_abort_transient_retry(monkeypatch):
    registry = _make_registry()

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    name = "TaskValidationBot"
    registry.graph.add_node(name)
    node = registry.graph.nodes[name]
    node["pending_internalization"] = True
    registry._internalization_retry_attempts[name] = 0

    def _raise_import_error(*_args, **_kwargs):
        raise ImportError(
            "cannot import name 'Helper' from partially initialized module "
            "'menace_sandbox.quick_fix_engine' (most likely due to a circular import)"
        )

    monkeypatch.setattr(registry, "_internalize_missing_coding_bot", _raise_import_error)

    registry._retry_internalization(name)

    node = registry.graph.nodes[name]
    assert node.get("pending_internalization") is False
    blocked = node.get("internalization_blocked")
    assert blocked is not None
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    deps = disabled.get("missing_dependencies", [])
    assert any(dep.endswith("quick_fix_engine") for dep in deps)
    assert name not in registry._internalization_retry_attempts


def test_internalize_missing_coding_bot_handles_runtime_dependency(monkeypatch):
    registry = _make_registry()

    registry.graph.add_node("TaskValidationBot")

    components = SimpleNamespace(
        context_builder_factory=lambda: object(),
        engine_cls=lambda *a, **k: object(),
        pipeline_cls=lambda *a, **k: object(),
        data_bot_cls=lambda *a, **k: object(),
        code_db_cls=lambda: object(),
        memory_manager_cls=lambda: object(),
    )

    def _raise_runtime(*_a, **_k):
        raise RuntimeError(
            "context_builder_util helpers are required for quick_fix_engine"
        )

    components.internalize_coding_bot = _raise_runtime

    monkeypatch.setattr(
        bot_registry, "_load_self_coding_components", lambda: components
    )
    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    with pytest.raises(bot_registry.SelfCodingUnavailableError) as excinfo:
        registry._internalize_missing_coding_bot(
            "TaskValidationBot",
            manager=object(),
            data_bot=object(),
        )

    missing = set(excinfo.value.missing_modules)
    assert "quick_fix_engine" in missing
    assert "context_builder_util" in missing


def test_retry_internalization_records_missing_resources(monkeypatch):
    registry = _make_registry()

    registry.graph.add_node("TaskValidationBot")
    node = registry.graph.nodes["TaskValidationBot"]
    node["pending_internalization"] = True
    registry._internalization_retry_attempts["TaskValidationBot"] = 0

    def _raise_resources(*_args, **_kwargs):
        raise bot_registry.SelfCodingUnavailableError(
            "self-coding bootstrap failed: missing runtime resources",
            missing_resources={"bots.db", "errors.db"},
        )

    monkeypatch.setattr(
        registry,
        "_internalize_missing_coding_bot",
        _raise_resources,
    )

    registry._retry_internalization("TaskValidationBot")

    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert set(disabled.get("missing_dependencies", ())) == {"bots.db", "errors.db"}
    assert set(disabled.get("missing_resources", ())) == {"bots.db", "errors.db"}
    assert node.get("pending_internalization") is False


def test_internalization_blocked_without_missing_disables(monkeypatch):
    registry = _make_registry()

    registry.graph.add_node("TaskValidationBot")
    node = registry.graph.nodes["TaskValidationBot"]
    node["pending_internalization"] = True
    registry._internalization_retry_attempts["TaskValidationBot"] = 0

    monkeypatch.setattr(
        bot_registry,
        "_collect_missing_modules",
        lambda exc: set(),
    )
    monkeypatch.setattr(
        bot_registry,
        "_collect_missing_resources",
        lambda exc: set(),
    )
    monkeypatch.setattr(
        bot_registry,
        "_derive_import_error_hints",
        lambda exc: {"mysterious.module"},
    )

    def _boom(*_args, **_kwargs):
        raise ImportError("bootstrap failed unexpectedly")

    monkeypatch.setattr(
        registry,
        "_internalize_missing_coding_bot",
        _boom,
    )

    registry._retry_internalization("TaskValidationBot")

    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled.get("source") == "internalization_blocked"
    assert "mysterious.module" in disabled.get("missing_dependencies", [])
    assert node.get("pending_internalization") is False


def test_transient_import_error_purges_partial_modules(monkeypatch):
    registry = _make_registry()

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    registry.graph.add_node("TaskValidationBot")
    node = registry.graph.nodes["TaskValidationBot"]
    node["pending_internalization"] = True
    registry._internalization_retry_attempts["TaskValidationBot"] = 0

    partial_msg = (
        "cannot import name 'TaskValidationBot' from partially initialized module "
        "'menace_sandbox.task_validation_bot' (most likely due to a circular import)"
    )

    def _boom(*_args, **_kwargs):
        raise ImportError(partial_msg)

    monkeypatch.setattr(
        registry,
        "_internalize_missing_coding_bot",
        _boom,
    )

    import sys
    import types

    sys.modules.setdefault("menace_sandbox", types.ModuleType("menace_sandbox"))
    partial = types.ModuleType("menace_sandbox.task_validation_bot")
    sys.modules["menace_sandbox.task_validation_bot"] = partial
    sys.modules["task_validation_bot"] = partial

    registry._retry_internalization("TaskValidationBot")

    assert "task_validation_bot" not in sys.modules
    assert "menace_sandbox.task_validation_bot" not in sys.modules
    assert "menace_sandbox" in sys.modules
    assert "TaskValidationBot" not in registry._internalization_retry_attempts
    assert node.get("pending_internalization") is False
    assert "internalization_blocked" in node


def test_transient_import_errors_with_varying_signatures(monkeypatch):
    registry = _make_registry()
    registry._max_transient_error_signature_repeats = 99
    registry._max_transient_error_total_repeats = 3
    registry._transient_error_grace_period = 0.0

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )
    monkeypatch.setattr(bot_registry, "_collect_missing_modules", lambda exc: set())

    attempts = {"count": 0}

    def _boom():
        attempts["count"] += 1
        raise ImportError(
            "cannot import name 'Helper' from partially initialized module 'menace.foo' "
            f"(attempt {attempts['count']})"
        )

    scheduled: list[tuple[str, float | None]] = []

    monkeypatch.setattr(
        registry,
        "_internalize_missing_coding_bot",
        lambda *a, **k: _boom(),
    )
    def _schedule(
        name: str,
        *,
        delay: float | None = None,
        force: bool = False,
    ) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    registry.register_bot("FutureProfitabilityBot", is_coding_bot=True)

    for _ in range(10):
        if not scheduled:
            break
        name, _delay = scheduled.pop(0)
        registry._retry_internalization(name)
        node = registry.graph.nodes[name]
        if node.get("self_coding_disabled"):
            break

    node = registry.graph.nodes["FutureProfitabilityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None, "self-coding should be disabled after repeated failures"
    error_meta = disabled["transient_error"]
    assert error_meta["total_repeat_count"] == attempts["count"]
    assert error_meta["unique_signatures"] == attempts["count"]
    assert error_meta["repeat_count"] == 1
    assert node.get("pending_internalization") is False


def test_manual_registration_clears_pending_state(tmp_path):
    registry = _make_registry()

    registry.graph.add_node("TaskValidationBot")
    node = registry.graph.nodes["TaskValidationBot"]
    node["pending_internalization"] = True
    node["self_coding_disabled"] = {
        "reason": "previous failure",
        "missing_dependencies": ["quick_fix_engine"],
    }
    registry._internalization_retry_attempts["TaskValidationBot"] = 3

    cancelled: list[str] = []

    class _Handle:
        def cancel(self) -> None:  # pragma: no cover - invoked by test
            cancelled.append("cancelled")

    registry._internalization_retry_handles["TaskValidationBot"] = _Handle()

    module_file = tmp_path / "task_validation_bot.py"
    module_file.write_text("# stub\n", encoding="utf-8")

    registry.register_bot(
        "TaskValidationBot",
        module_path=module_file,
        is_coding_bot=False,
    )

    node = registry.graph.nodes["TaskValidationBot"]
    assert "pending_internalization" not in node
    assert "internalization_blocked" not in node
    assert "internalization_errors" not in node
    assert registry._internalization_retry_attempts.get("TaskValidationBot") is None
    assert cancelled == ["cancelled"]

    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["manual_override"] is True
    assert disabled["source"] == "manual_registration"
    assert disabled["module"].endswith("task_validation_bot.py")
    assert disabled.get("previous_reason") == "previous failure"
    assert disabled.get("missing_dependencies") == ["quick_fix_engine"]


def test_load_self_coding_components_uses_import_helper(monkeypatch):
    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    loaded: list[str] = []

    def _fake_loader(name: str):
        loaded.append(name)
        if name == "self_coding_manager":
            return SimpleNamespace(
                internalize_coding_bot=lambda *a, **k: None,
                SelfCodingManager=object,
            )
        if name == "self_coding_engine":
            return SimpleNamespace(SelfCodingEngine=type("_Engine", (), {}))
        if name == "model_automation_pipeline":
            return SimpleNamespace(
                ModelAutomationPipeline=type("_Pipeline", (), {})
            )
        if name == "data_bot":
            return SimpleNamespace(DataBot=type("_DataBot", (), {}))
        if name == "code_database":
            return SimpleNamespace(CodeDB=type("_CodeDB", (), {}))
        if name == "gpt_memory":
            return SimpleNamespace(GPTMemoryManager=type("_Memory", (), {}))
        if name == "context_builder_util":
            return SimpleNamespace(create_context_builder=lambda: object())
        raise AssertionError(f"unexpected module request: {name}")

    monkeypatch.setattr(bot_registry, "_load_internal_module", _fake_loader)

    components = bot_registry._load_self_coding_components()

    assert components.engine_cls is not None
    assert components.pipeline_cls is not None
    assert components.data_bot_cls is not None
    assert components.memory_manager_cls is not None
    assert components.context_builder_factory() is not None
    assert sorted(loaded) == [
        "code_database",
        "context_builder_util",
        "data_bot",
        "gpt_memory",
        "model_automation_pipeline",
        "self_coding_engine",
        "self_coding_manager",
    ]


def test_load_self_coding_components_handles_flat_import_error(monkeypatch):
    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    calls: list[str] = []

    def _loader(name: str):
        calls.append(name)
        raise ImportError(
            "attempted relative import with no known parent package", name=None
        )

    monkeypatch.setattr(bot_registry, "_load_internal_module", _loader)

    with pytest.raises(bot_registry.SelfCodingUnavailableError) as excinfo:
        bot_registry._load_self_coding_components()

    assert calls == ["self_coding_manager"]
    assert "self_coding_manager" in excinfo.value.missing_dependencies

