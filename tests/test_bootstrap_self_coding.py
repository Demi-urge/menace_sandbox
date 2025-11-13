"""Regression tests for the bootstrap_self_coding helper script."""

from __future__ import annotations

import logging
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

import menace_sandbox.coding_bot_interface as coding_bot_interface


@pytest.fixture(autouse=True)
def _preserve_runtime_flags(monkeypatch):
    """Ensure runtime availability probes are restored after each test."""

    original = coding_bot_interface._self_coding_runtime_available
    monkeypatch.setattr(
        coding_bot_interface,
        "_self_coding_runtime_available",
        original,
    )
    yield


def test_script_bootstrap_promotes_real_manager(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    class _Registry:
        def register_bot(self, *_args, **_kwargs):
            return None

        def update_bot(self, *_args, **_kwargs):
            return None

    registry = _Registry()

    class _Thresholds:
        roi_drop = 0.0
        error_threshold = 0.0
        test_failure_threshold = 0.0

    data_bot = SimpleNamespace(
        reload_thresholds=lambda _name: _Thresholds,
        schedule_monitoring=lambda _name: None,
    )

    pipeline_state: dict[str, object] = {}

    def _registry_factory() -> _Registry:
        return registry

    def _data_bot_factory() -> SimpleNamespace:
        return data_bot

    _registry_factory.__self_coding_lazy__ = True  # type: ignore[attr-defined]
    _data_bot_factory.__self_coding_lazy__ = True  # type: ignore[attr-defined]

    @coding_bot_interface.self_coding_managed(
        bot_registry=_registry_factory,
        data_bot=_data_bot_factory,
    )
    class _PipelineBot:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            pipeline_state.setdefault("bots", []).append(self)

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = registry
            self.data_bot = data_bot

    class _StubPipeline:
        def __init__(self, *, context_builder):
            self.context_builder = context_builder
            self.manager = None
            self.initial_manager = None
            self.reattach_calls = 0
            self._bots = [_PipelineBot(manager=None)]
            self.finalized = False
            self.registry_seen = None
            self.data_bot_seen = None
            pipeline_state["pipeline"] = self

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.finalized = True
            self.registry_seen = registry
            self.data_bot_seen = data_bot

    monkeypatch.setattr(
        coding_bot_interface,
        "_self_coding_runtime_available",
        lambda: False,
    )

    builder = SimpleNamespace()
    pipeline = _StubPipeline(context_builder=builder)
    initial_manager = pipeline_state["bots"][0].manager
    assert isinstance(initial_manager, coding_bot_interface._DisabledSelfCodingManager)

    class _StubManager:
        def __init__(self, *, bot_registry, data_bot):
            context = coding_bot_interface._current_bootstrap_context()
            assert context is not None
            sentinel = context.manager
            pipeline_ref = pipeline_state["pipeline"]
            pipeline_ref.manager = sentinel
            pipeline_ref.initial_manager = sentinel
            for bot in pipeline_ref._bots:
                bot.manager = sentinel
            pipeline_ref._attach_information_synthesis_manager()
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.pipeline = pipeline_ref
            self.engine = SimpleNamespace()

    monkeypatch.setattr(
        coding_bot_interface,
        "_self_coding_runtime_available",
        lambda: True,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _StubManager,
    )

    manager = coding_bot_interface._bootstrap_manager("ScriptedBot", registry, data_bot)

    assert manager
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)
    assert "re-entrant initialisation depth" not in caplog.text

    assert manager.pipeline is pipeline
    assert pipeline.manager is manager
    assert isinstance(
        pipeline.initial_manager, coding_bot_interface._BootstrapManagerSentinel
    )
    assert pipeline.reattach_calls >= 1
    assert pipeline.finalized is True
    assert pipeline.registry_seen is registry
    assert pipeline.data_bot_seen is data_bot
    assert all(bot.manager is manager for bot in pipeline._bots)


def test_bootstrap_entrypoint_uses_prepare_helper(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    import menace_sandbox.bootstrap_self_coding as bootstrap_self_coding

    bootstrap_self_coding = importlib.reload(bootstrap_self_coding)

    builder = SimpleNamespace()
    monkeypatch.setattr(
        "menace_sandbox.context_builder_util.create_context_builder",
        lambda: builder,
    )

    class _Registry:
        def __init__(self) -> None:
            self.registered = []

    class _DataBot:
        def __init__(self, *_, **__):
            self.thresholds = {}

    monkeypatch.setattr("menace_sandbox.bot_registry.BotRegistry", _Registry)
    monkeypatch.setattr("menace_sandbox.data_bot.DataBot", _DataBot)
    monkeypatch.setattr("menace_sandbox.code_database.CodeDB", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "menace_sandbox.menace_memory_manager.MenaceMemoryManager",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "menace_sandbox.self_coding_engine.SelfCodingEngine",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )

    thresholds = SimpleNamespace(
        roi_drop=0.1, error_increase=0.2, test_failure_increase=0.3
    )
    monkeypatch.setattr(
        "menace_sandbox.self_coding_thresholds.get_thresholds",
        lambda _name: thresholds,
    )
    monkeypatch.setattr(
        "menace_sandbox.data_bot.persist_sc_thresholds",
        lambda *_, **__: None,
    )

    helper_called = False
    pipeline_box: dict[str, SimpleNamespace] = {}
    manager_box: dict[str, SimpleNamespace] = {}

    def _prepare_pipeline_for_bootstrap(*, pipeline_cls, context_builder, bot_registry, data_bot):
        nonlocal helper_called
        helper_called = True
        sentinel = SimpleNamespace(kind="sentinel", pipeline_cls=pipeline_cls)
        pipeline = SimpleNamespace(
            manager=sentinel,
            initial_manager=sentinel,
            context_builder=context_builder,
            bot_registry=bot_registry,
            data_bot=data_bot,
            promotions=[],
            sentinel=sentinel,
        )
        pipeline_box["pipeline"] = pipeline

        def _promote(manager):
            pipeline.promotions.append(manager)
            pipeline.manager = manager

        return pipeline, _promote

    monkeypatch.setattr(
        "menace_sandbox.coding_bot_interface.prepare_pipeline_for_bootstrap",
        _prepare_pipeline_for_bootstrap,
    )

    def _internalize_coding_bot(*, pipeline, **_kwargs):
        assert pipeline is pipeline_box["pipeline"]
        assert pipeline.manager is pipeline.sentinel
        manager = SimpleNamespace(kind="manager", pipeline=pipeline)
        manager_box["manager"] = manager
        return manager

    monkeypatch.setattr(
        "menace_sandbox.self_coding_manager.internalize_coding_bot",
        _internalize_coding_bot,
    )

    bootstrap_self_coding.bootstrap_self_coding("ExampleBot")

    assert helper_called is True
    pipeline = pipeline_box["pipeline"]
    manager = manager_box["manager"]
    assert pipeline.manager is manager
    assert pipeline.promotions == [manager]
    assert "re-entrant initialisation depth" not in caplog.text


def test_error_bot_legacy_bootstrap_promotes_manager(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    import menace_sandbox.error_bot as error_bot

    error_bot = importlib.reload(error_bot)

    builder = SimpleNamespace()
    monkeypatch.setattr(error_bot, "create_context_builder", lambda: builder)
    monkeypatch.setattr(error_bot, "CodeDB", lambda: SimpleNamespace())
    monkeypatch.setattr(error_bot, "GPTMemoryManager", lambda: SimpleNamespace())
    monkeypatch.setattr(error_bot, "ThresholdService", lambda: SimpleNamespace())
    monkeypatch.setattr(
        error_bot,
        "get_thresholds",
        lambda _name: SimpleNamespace(
            roi_drop=0.1, error_increase=0.2, test_failure_increase=0.3
        ),
    )
    monkeypatch.setattr(error_bot, "persist_sc_thresholds", lambda *args, **kwargs: None)
    monkeypatch.setattr(error_bot, "get_orchestrator", lambda *args, **kwargs: SimpleNamespace())

    registry = SimpleNamespace()
    data_bot = SimpleNamespace()
    error_bot.registry = registry
    error_bot.data_bot = data_bot

    class _StubCommunicationMaintenanceBot:
        def __init__(self, manager):
            self.manager = manager
            self.finalized_with = None

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.manager = manager
            self.finalized_with = (registry, data_bot)

    class _StubPipeline:
        instantiation_count = 0

        def __init__(self, *, context_builder, manager=None, bot_registry=None, data_bot=None):
            type(self).instantiation_count += 1
            self.context_builder = context_builder
            self.manager = manager
            self.initial_manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.communication_bot = _StubCommunicationMaintenanceBot(manager)
            self.comms_bot = self.communication_bot
            self._bots = [self.communication_bot]
            self.finalized_with = None

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.finalized_with = (manager, registry, data_bot)

    pipeline_module = types.ModuleType("menace_sandbox.model_automation_pipeline")
    pipeline_module.ModelAutomationPipeline = _StubPipeline
    monkeypatch.setitem(sys.modules, "menace_sandbox.model_automation_pipeline", pipeline_module)

    class _StubEngine:
        def __init__(self, *_args, **_kwargs):
            pass

    engine_module = types.ModuleType("menace_sandbox.self_coding_engine")
    engine_module.SelfCodingEngine = _StubEngine
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", engine_module)

    class _StubManager:
        def __init__(self, *, engine, pipeline, bot_registry, data_bot):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    def _internalize(
        _bot_name,
        engine,
        pipeline,
        *,
        data_bot,
        bot_registry,
        **_kwargs,
    ):
        return _StubManager(
            engine=engine,
            pipeline=pipeline,
            bot_registry=bot_registry,
            data_bot=data_bot,
        )

    manager_module = types.ModuleType("menace_sandbox.self_coding_manager")
    manager_module.internalize_coding_bot = _internalize
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", manager_module)

    manager = error_bot.get_manager()

    assert "re-entrant initialisation depth" not in caplog.text

    pipeline = error_bot.get_pipeline()
    assert isinstance(pipeline.initial_manager, coding_bot_interface._BootstrapManagerSentinel)
    assert pipeline.manager is manager
    assert pipeline.finalized_with == (manager, registry, data_bot)
    assert _StubPipeline.instantiation_count == 1

    communication_bot = pipeline.communication_bot
    assert communication_bot.manager is manager
    assert communication_bot.finalized_with == (registry, data_bot)

    assert error_bot.get_manager() is manager
    assert _StubPipeline.instantiation_count == 1


def test_bootstrap_entrypoint_promotes_manager(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    import menace_sandbox.bootstrap_self_coding as bootstrap_self_coding

    bootstrap_self_coding = importlib.reload(bootstrap_self_coding)

    builder = SimpleNamespace()
    monkeypatch.setattr(
        "menace_sandbox.context_builder_util.create_context_builder",
        lambda: builder,
    )

    monkeypatch.setattr(
        "menace_sandbox.code_database.CodeDB", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(
        "menace_sandbox.menace_memory_manager.MenaceMemoryManager",
        lambda: SimpleNamespace(),
    )

    class _Engine:
        def __init__(self, *_args, **_kwargs) -> None:
            self.args = _args
            self.kwargs = _kwargs

    monkeypatch.setattr(
        "menace_sandbox.self_coding_engine.SelfCodingEngine", _Engine
    )

    class _Registry:
        def __init__(self) -> None:
            self.bots: list[str] = []

    class _DataBot:
        def __init__(self, *_, **__) -> None:
            self.persisted: list[tuple[str, object]] = []

    monkeypatch.setattr("menace_sandbox.bot_registry.BotRegistry", _Registry)
    monkeypatch.setattr("menace_sandbox.data_bot.DataBot", _DataBot)

    thresholds = SimpleNamespace(
        roi_drop=0.1, error_increase=0.2, test_failure_increase=0.3
    )

    monkeypatch.setattr(
        "menace_sandbox.self_coding_thresholds.get_thresholds",
        lambda _name: thresholds,
    )
    monkeypatch.setattr(
        "menace_sandbox.data_bot.persist_sc_thresholds", lambda *_, **__: None
    )

    manager_box: dict[str, SimpleNamespace] = {}
    pipeline_box: dict[str, SimpleNamespace] = {}
    promote_calls: list[SimpleNamespace] = []

    def _prepare_pipeline_for_bootstrap(
        *,
        pipeline_cls,
        context_builder,
        bot_registry,
        data_bot,
    ):
        sentinel = SimpleNamespace(kind="sentinel", pipeline_cls=pipeline_cls)
        pipeline = SimpleNamespace(
            context_builder=context_builder,
            manager=sentinel,
            initial_manager=sentinel,
            bot_registry=bot_registry,
            data_bot=data_bot,
            sentinel_manager=sentinel,
        )
        pipeline_box["pipeline"] = pipeline

        def _promote(manager):
            promote_calls.append(manager)
            pipeline.manager = manager

        return pipeline, _promote

    monkeypatch.setattr(
        coding_bot_interface,
        "prepare_pipeline_for_bootstrap",
        _prepare_pipeline_for_bootstrap,
    )

    def _internalize_coding_bot(
        *,
        bot_name: str,
        engine: object,
        pipeline: SimpleNamespace,
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert pipeline is pipeline_box["pipeline"]
        sentinel = pipeline.sentinel_manager
        assert pipeline.manager is sentinel
        manager = SimpleNamespace(bot_name=bot_name, pipeline=pipeline, engine=engine)
        manager_box["manager"] = manager
        return manager

    monkeypatch.setattr(
        "menace_sandbox.self_coding_manager.internalize_coding_bot",
        _internalize_coding_bot,
    )

    bootstrap_self_coding.bootstrap_self_coding("ExampleBot")
    pipeline = pipeline_box["pipeline"]
    manager = manager_box["manager"]
    assert pipeline.manager is manager
    assert "re-entrant initialisation depth" not in caplog.text
    assert promote_calls == [manager]


def test_bootstrap_entrypoint_promotes_all_bots(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    import menace_sandbox.bootstrap_self_coding as bootstrap_self_coding

    bootstrap_self_coding = importlib.reload(bootstrap_self_coding)

    builder = SimpleNamespace()
    monkeypatch.setattr(
        "menace_sandbox.context_builder_util.create_context_builder",
        lambda: builder,
    )

    class _Registry:
        def __init__(self) -> None:
            self.registered: list[str] = []

    class _DataBot:
        def __init__(self, *_, **__):
            self.thresholds: dict[str, object] = {}

        def reload_thresholds(self, *_args, **_kwargs):
            return SimpleNamespace(
                roi_drop=0.1, error_threshold=0.2, test_failure_threshold=0.3
            )

    class _CodeDB:
        def __init__(self) -> None:
            self.initialized = True

    class _MemoryManager:
        def __init__(self) -> None:
            self.context = None

    class _Engine:
        def __init__(self, *_args, **_kwargs) -> None:
            self.args = _args
            self.kwargs = _kwargs

    monkeypatch.setattr("menace_sandbox.bot_registry.BotRegistry", _Registry)
    monkeypatch.setattr("menace_sandbox.data_bot.DataBot", _DataBot)
    monkeypatch.setattr(
        "menace_sandbox.data_bot.persist_sc_thresholds", lambda *_, **__: None
    )
    monkeypatch.setattr("menace_sandbox.code_database.CodeDB", _CodeDB)
    monkeypatch.setattr(
        "menace_sandbox.menace_memory_manager.MenaceMemoryManager", _MemoryManager
    )
    monkeypatch.setattr("menace_sandbox.self_coding_engine.SelfCodingEngine", _Engine)

    thresholds = SimpleNamespace(
        roi_drop=0.1, error_increase=0.2, test_failure_increase=0.3
    )
    monkeypatch.setattr(
        "menace_sandbox.self_coding_thresholds.get_thresholds",
        lambda _name: thresholds,
    )

    pipeline_box: dict[str, SimpleNamespace] = {}
    manager_box: dict[str, SimpleNamespace] = {}

    def _prepare_pipeline_for_bootstrap(
        *,
        pipeline_cls,
        context_builder,
        bot_registry,
        data_bot,
    ):
        sentinel = coding_bot_interface._create_bootstrap_manager_sentinel(
            bot_registry=bot_registry,
            data_bot=data_bot,
        )
        bots = [SimpleNamespace(name=f"bot-{idx}", manager=sentinel) for idx in range(3)]
        pipeline = SimpleNamespace(
            context_builder=context_builder,
            manager=sentinel,
            initial_manager=sentinel,
            bot_registry=bot_registry,
            data_bot=data_bot,
            sentinel_manager=sentinel,
            _bots=bots,
            promotions=[],
        )
        pipeline_box["pipeline"] = pipeline

        def _promote(manager):
            pipeline.promotions.append(manager)
            pipeline.manager = manager
            for bot in pipeline._bots:
                bot.manager = manager

        return pipeline, _promote

    monkeypatch.setattr(
        "menace_sandbox.coding_bot_interface.prepare_pipeline_for_bootstrap",
        _prepare_pipeline_for_bootstrap,
    )

    def _internalize_coding_bot(
        *,
        bot_name: str,
        engine: object,
        pipeline: SimpleNamespace,
        data_bot: object,
        bot_registry: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert pipeline is pipeline_box["pipeline"]
        manager = SimpleNamespace(
            bot_name=bot_name,
            engine=engine,
            pipeline=pipeline,
            data_bot=data_bot,
            bot_registry=bot_registry,
        )
        manager_box["manager"] = manager
        return manager

    monkeypatch.setattr(
        "menace_sandbox.self_coding_manager.internalize_coding_bot",
        _internalize_coding_bot,
    )

    bootstrap_self_coding.bootstrap_self_coding("ExampleBot")

    pipeline = pipeline_box["pipeline"]
    manager = manager_box["manager"]
    assert manager
    assert pipeline.manager is manager
    assert pipeline.promotions == [manager]
    assert all(bot.manager is manager for bot in pipeline._bots)
    assert "re-entrant initialisation depth" not in caplog.text
