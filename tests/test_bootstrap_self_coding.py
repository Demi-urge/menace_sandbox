"""Regression tests for the bootstrap_self_coding helper script."""

from __future__ import annotations

import logging
import importlib
import os
import sys
import types
from types import SimpleNamespace
import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import pytest

import menace_sandbox.coding_bot_interface as coding_bot_interface
import bootstrap_self_coding
import tests.test_bootstrap_manager_self_coding as manager_tests

pytest_plugins = ("tests.test_bootstrap_manager_self_coding",)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


@pytest.fixture
def eager_helper_stub_env(monkeypatch, tmp_path):
    stub_env = manager_tests.stub_bootstrap_env.__wrapped__(monkeypatch)
    return manager_tests.eager_helper_pipeline_env.__wrapped__(
        stub_env, monkeypatch, tmp_path
    )


def test_prepare_pipeline_promotes_eager_helper_stub(
    eager_helper_stub_env, caplog
):
    env = eager_helper_stub_env

    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)
    pipeline, promoter = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=env.pipeline_cls,
        context_builder=env.builder,
        bot_registry=env.registry,
        data_bot=env.data_bot,
    )

    helper = env.helper_instances[-1]
    assert helper is pipeline.comms_bot
    assert env.helper_init_managers
    runtime_manager = env.helper_init_managers[-1]
    assert isinstance(runtime_manager, coding_bot_interface._DisabledSelfCodingManager)
    assert "re-entrant initialisation depth" not in caplog.text

    manager = coding_bot_interface._bootstrap_manager(
        "PreparedEagerHelperStub",
        env.registry,
        env.data_bot,
        pipeline=pipeline,
        pipeline_manager=pipeline.manager,
        pipeline_promoter=promoter,
    )

    assert manager
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)
    assert helper.manager is manager
    assert env.helper_final_managers and env.helper_final_managers[-1] is manager
    assert "re-entrant initialisation depth" not in caplog.text


def test_bootstrap_manager_promotes_eager_helper_stub(
    eager_helper_stub_env, monkeypatch, caplog
):
    env = eager_helper_stub_env

    _install_managerless_bootstrap(monkeypatch, env.pipeline_cls)
    monkeypatch.setattr(
        coding_bot_interface, "create_context_builder", lambda: env.builder
    )

    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    manager = coding_bot_interface._bootstrap_manager(
        "FallbackEagerHelperStub",
        env.registry,
        env.data_bot,
    )

    assert manager
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)
    helper = env.helper_instances[-1]
    assert helper.manager is manager
    assert env.helper_final_managers and env.helper_final_managers[-1] is manager
    assert "re-entrant initialisation depth" not in caplog.text


@dataclass(frozen=True)
class _EntryPointCase:
    """Describe a bootstrap entry point for regression coverage."""

    name: str
    module_path: str
    invoker: Callable[[Any, dict[str, Any]], Any]
    setup: Callable[[Any, Any], dict[str, Any]]
    pre_reload: Callable[[Any], None] | None = None
    reload: bool = True


def _install_managerless_bootstrap(monkeypatch, pipeline_cls):
    builder = SimpleNamespace(name="managerless-builder")
    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: builder,
    )

    code_db_module = types.ModuleType("menace_sandbox.code_database")
    code_db_module.CodeDB = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "menace_sandbox.code_database", code_db_module)
    monkeypatch.setitem(sys.modules, "code_database", code_db_module)

    memory_module = types.ModuleType("menace_sandbox.gpt_memory")
    memory_module.GPTMemoryManager = lambda *_args, **_kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "menace_sandbox.gpt_memory", memory_module)
    monkeypatch.setitem(sys.modules, "gpt_memory", memory_module)

    class _Engine:
        def __init__(self, *_args, **_kwargs):
            self.context_builder = _kwargs.get("context_builder")

    engine_module = types.ModuleType("menace_sandbox.self_coding_engine")
    engine_module.SelfCodingEngine = _Engine
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.self_coding_engine", engine_module
    )
    monkeypatch.setitem(sys.modules, "self_coding_engine", engine_module)

    class _SelfCodingManager:
        def __init__(self, engine, pipeline, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    manager_module = types.ModuleType("menace_sandbox.self_coding_manager")
    manager_module.SelfCodingManager = _SelfCodingManager
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.self_coding_manager", manager_module
    )
    monkeypatch.setitem(sys.modules, "self_coding_manager", manager_module)

    pipeline_module = types.ModuleType("menace_sandbox.model_automation_pipeline")
    pipeline_module.ModelAutomationPipeline = pipeline_cls
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.model_automation_pipeline", pipeline_module
    )
    monkeypatch.setitem(sys.modules, "model_automation_pipeline", pipeline_module)

    def _fake_load_internal(name: str):
        module = sys.modules.get(f"menace_sandbox.{name}")
        if module is None:
            raise ModuleNotFoundError(name)
        return module

    monkeypatch.setattr(coding_bot_interface, "load_internal", _fake_load_internal)

    class _PrimaryManager:
        def __init__(self, *, bot_registry, data_bot):
            raise TypeError("legacy bootstrap rejecting manager keyword")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _PrimaryManager,
    )

    monkeypatch.setattr(
        coding_bot_interface,
        "_self_coding_runtime_available",
        lambda: True,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "ensure_self_coding_ready",
        lambda modules=None: (True, ()),
    )


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


def test_fallback_bootstrap_nested_helpers_use_owner_sentinel(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    thresholds = SimpleNamespace(
        roi_drop=0.0,
        error_threshold=0.0,
        test_failure_threshold=0.0,
    )

    class _Registry:
        def register_bot(self, *_args, **_kwargs):
            return None

        def update_bot(self, *_args, **_kwargs):
            return None

    class _DataBot:
        def reload_thresholds(self, *_args, **_kwargs):
            return thresholds

        def schedule_monitoring(self, *_args, **_kwargs):
            return None

    registry = _Registry()
    data_bot = _DataBot()

    monkeypatch.setattr(
        coding_bot_interface,
        "ensure_self_coding_ready",
        lambda modules=None: (True, ()),
    )

    policy = SimpleNamespace(
        allowlist=None,
        denylist=frozenset(),
        is_enabled=lambda _name: True,
    )
    monkeypatch.setattr(
        coding_bot_interface, "get_self_coding_policy", lambda: policy
    )

    monkeypatch.setattr(
        coding_bot_interface,
        "create_context_builder",
        lambda: SimpleNamespace(name="context"),
    )

    class _FailingManager:
        def __init__(self, *, bot_registry, data_bot):  # pragma: no cover - stub
            raise TypeError("legacy path")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    helper_instances: list[SimpleNamespace] = []
    nested_helper_instances: list[SimpleNamespace] = []

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class _NestedHelper:
        name = "FallbackNestedHelper"

        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.initial_manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            nested_helper_instances.append(self)

        def _finalize_self_coding_bootstrap(
            self, manager, *, registry=None, data_bot=None
        ):
            self.manager = manager
            if registry is not None:
                self.bot_registry = registry
            if data_bot is not None:
                self.data_bot = data_bot

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class _OuterHelper:
        name = "FallbackOuterHelper"

        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.initial_manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.nested_helper = _NestedHelper(
                manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            helper_instances.append(self)

        def _finalize_self_coding_bootstrap(
            self, manager, *, registry=None, data_bot=None
        ):
            self.manager = manager
            if registry is not None:
                self.bot_registry = registry
            if data_bot is not None:
                self.data_bot = data_bot

    class _StubPipeline:
        def __init__(self, *, context_builder, bot_registry, data_bot, manager):
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            self.finalized = False
            self.finalizer_args = None
            self._bots = []
            self.helper = _OuterHelper(
                manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            self._bots.extend([self.helper, self.helper.nested_helper])
            self.reattach_calls = 0

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

        def _finalize_self_coding_bootstrap(
            self, manager, *, registry=None, data_bot=None
        ):
            self.finalized = True
            self.finalizer_args = (manager, registry, data_bot)

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _StubSelfCodingManager:
        def __init__(self, engine, pipeline, *, bot_name, data_bot, bot_registry):
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry

    stub_modules = {
        "code_database": SimpleNamespace(CodeDB=_StubCodeDB),
        "gpt_memory": SimpleNamespace(GPTMemoryManager=_StubMemory),
        "self_coding_engine": SimpleNamespace(SelfCodingEngine=_StubEngine),
        "model_automation_pipeline": SimpleNamespace(
            ModelAutomationPipeline=_StubPipeline
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubSelfCodingManager),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        if name not in stub_modules:
            raise AssertionError(f"unexpected optional module {name}")
        return stub_modules[name]

    monkeypatch.setattr(
        coding_bot_interface,
        "_load_optional_module",
        _load_optional_module,
    )

    manager = coding_bot_interface._bootstrap_manager(
        "NestedFallbackBot", registry, data_bot
    )

    assert manager
    assert isinstance(manager, _StubSelfCodingManager)
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)
    assert "re-entrant initialisation depth" not in caplog.text

    pipeline = manager.pipeline
    assert isinstance(pipeline, _StubPipeline)
    assert pipeline.manager is manager
    assert pipeline.reattach_calls >= 1

    assert helper_instances and nested_helper_instances
    for helper in (*helper_instances, *nested_helper_instances):
        assert helper.manager is manager
        assert helper.initial_manager is not manager
        assert helper.bot_registry is registry
        assert helper.data_bot is data_bot

    assert all(bot.manager is manager for bot in pipeline._bots)


def test_bootstrap_manager_handles_prebuilt_pipeline(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    registry = SimpleNamespace()
    data_bot = SimpleNamespace()

    class _FailingManager:
        def __init__(self, *, bot_registry, data_bot):
            raise TypeError("use fallback")

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

    helper = SimpleNamespace(manager=None)
    pipeline = SimpleNamespace(
        manager=None,
        initial_manager=None,
        helper=helper,
        _bots=[helper],
        comms_bot=None,
        synthesis_bot=None,
        diagnostic_manager=None,
        planner=None,
        aggregator=None,
        reattach_calls=0,
        finalized=False,
        finalizer_args=None,
    )

    def _attach_information_synthesis_manager():
        pipeline.reattach_calls += 1

    def _finalize_self_coding_bootstrap(manager, *, registry=None, data_bot=None):
        pipeline.finalized = True
        pipeline.finalizer_args = (manager, registry, data_bot)

    pipeline._attach_information_synthesis_manager = _attach_information_synthesis_manager
    pipeline._finalize_self_coding_bootstrap = _finalize_self_coding_bootstrap

    def _pipeline_factory(**kwargs):
        manager = kwargs.get("manager")
        pipeline.manager = manager
        pipeline.initial_manager = manager
        helper.manager = manager
        helper.initial_manager = manager
        pipeline.context = kwargs
        return pipeline

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
            ModelAutomationPipeline=_pipeline_factory
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

    # Script builds the pipeline before self-coding bootstrap runs.
    _pipeline_factory(
        context_builder=SimpleNamespace(),
        bot_registry=registry,
        data_bot=data_bot,
        manager=None,
    )

    manager = coding_bot_interface._bootstrap_manager(
        "OneOffScriptBot", registry, data_bot
    )

    assert manager.pipeline is pipeline
    assert pipeline.manager is manager
    assert helper.manager is manager
    assert helper.initial_manager is not manager
    assert getattr(helper.initial_manager, "_bootstrap_owner_marker", False)
    assert pipeline._finalize_self_coding_bootstrap is _finalize_self_coding_bootstrap
    assert pipeline.reattach_calls >= 1
    assert "re-entrant" not in caplog.text


def test_fallback_prebuilt_pipeline_helpers_receive_promoted_manager(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    builder = SimpleNamespace()
    monkeypatch.setattr(coding_bot_interface, "create_context_builder", lambda: builder)

    runtime_enabled = {"value": False}

    def _runtime_available() -> bool:
        return runtime_enabled["value"]

    monkeypatch.setattr(
        coding_bot_interface,
        "_self_coding_runtime_available",
        _runtime_available,
    )

    registry = SimpleNamespace()
    data_bot = SimpleNamespace()

    helper_instances: list[SimpleNamespace] = []

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class _ManagedHelper:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.created_during_fallback = runtime_enabled["value"]
            helper_instances.append(self)

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.manager = manager
            if registry is not None:
                self.bot_registry = registry
            if data_bot is not None:
                self.data_bot = data_bot

    pipeline_box: dict[str, SimpleNamespace] = {}

    def _pipeline_factory(*, context_builder, manager=None, bot_registry=None, data_bot=None):
        helper = _ManagedHelper(
            manager=manager,
            bot_registry=bot_registry,
            data_bot=data_bot,
        )
        pipeline = pipeline_box.get("pipeline")
        if pipeline is None:
            pipeline = SimpleNamespace(
                context_builder=context_builder,
                helper_instances=[helper],
                _bots=[helper],
                helper=helper,
                manager=manager,
                initial_manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
                reattach_calls=0,
                finalized=False,
                finalizer_args=None,
            )

            def _attach_information_synthesis_manager():
                pipeline.reattach_calls += 1

            def _finalize_self_coding_bootstrap(manager_obj, *, registry=None, data_bot=None):
                pipeline.finalized = True
                pipeline.finalizer_args = (manager_obj, registry, data_bot)

            pipeline._attach_information_synthesis_manager = (
                _attach_information_synthesis_manager
            )
            pipeline._finalize_self_coding_bootstrap = _finalize_self_coding_bootstrap
            pipeline_box["pipeline"] = pipeline
        else:
            pipeline.helper_instances.append(helper)
            pipeline._bots.append(helper)
            pipeline.helper = helper
            pipeline.manager = manager
            pipeline.initial_manager = manager
            pipeline.bot_registry = bot_registry
            pipeline.data_bot = data_bot
        return pipeline

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder
            self.cognition_layer = SimpleNamespace(context_builder=context_builder)

    class _FallbackManager:
        def __init__(self, *, bot_registry, data_bot):
            raise TypeError("force fallback pipeline path")

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
            ModelAutomationPipeline=_pipeline_factory
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubManager),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        if name in modules:
            return modules[name]
        raise AssertionError(f"unexpected optional module {name}")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _FallbackManager,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "_load_optional_module",
        _load_optional_module,
    )

    # Script builds the pipeline before bootstrap kicks in.
    _pipeline_factory(
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
        manager=None,
    )

    runtime_enabled["value"] = True

    manager = coding_bot_interface._bootstrap_manager(
        "PrebuiltFallbackBot", registry, data_bot
    )

    assert manager.pipeline is pipeline_box["pipeline"]
    assert "re-entrant initialisation depth" not in caplog.text
    promoted_helpers = [
        helper for helper in helper_instances if helper.created_during_fallback
    ]
    assert promoted_helpers, "expected helpers created during fallback bootstrap"
    for helper in promoted_helpers:
        assert helper.manager is manager
        assert not isinstance(
            helper.manager, coding_bot_interface._DisabledSelfCodingManager
        )


def test_bootstrap_manager_handles_nested_helper_bootstrap(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    builder = SimpleNamespace()
    monkeypatch.setattr(coding_bot_interface, "create_context_builder", lambda: builder)
    monkeypatch.setattr(
        coding_bot_interface,
        "_self_coding_runtime_available",
        lambda: True,
    )

    class _Registry:
        def register_bot(self, *_args, **_kwargs):
            return None

        def update_bot(self, *_args, **_kwargs):
            return None

    registry = _Registry()

    class _DataBot:
        def __init__(self) -> None:
            self.thresholds: dict[str, float] = {}

        def reload_thresholds(self, *_args, **_kwargs):
            return SimpleNamespace(
                roi_drop=0.1,
                error_threshold=0.2,
                test_failure_threshold=0.3,
            )

    data_bot = _DataBot()

    helper_instances: list[SimpleNamespace] = []

    def _registry_factory() -> SimpleNamespace:
        return registry

    def _data_bot_factory() -> _DataBot:
        return data_bot

    _registry_factory.__self_coding_lazy__ = True  # type: ignore[attr-defined]
    _data_bot_factory.__self_coding_lazy__ = True  # type: ignore[attr-defined]

    @coding_bot_interface.self_coding_managed(
        bot_registry=_registry_factory,
        data_bot=_data_bot_factory,
    )
    class _ManagedHelper:
        name = "NestedBootstrapHelper"

        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.initial_manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.finalized_with = None
            helper_instances.append(self)

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = registry or self.bot_registry
            self.data_bot = data_bot or self.data_bot
            self.finalized_with = (registry, data_bot)

    pipeline_box: dict[str, SimpleNamespace] = {}

    def _pipeline_factory(*, context_builder, manager=None, bot_registry=None, data_bot=None):
        pipeline = pipeline_box.get("pipeline")
        if pipeline is None:
            pipeline = SimpleNamespace(
                context_builder=context_builder,
                manager=manager,
                initial_manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
                helper=None,
                _bots=[],
                finalized=False,
                finalizer_args=None,
                reattach_calls=0,
            )

            def _attach_information_synthesis_manager():
                pipeline.reattach_calls += 1

            def _finalize_self_coding_bootstrap(manager_obj, *, registry=None, data_bot=None):
                pipeline.finalized = True
                pipeline.finalizer_args = (manager_obj, registry, data_bot)

            pipeline._attach_information_synthesis_manager = (
                _attach_information_synthesis_manager
            )
            pipeline._finalize_self_coding_bootstrap = _finalize_self_coding_bootstrap
            helper = _ManagedHelper(
                manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            pipeline.helper = helper
            pipeline._bots = [helper]
            pipeline_box["pipeline"] = pipeline
            return pipeline

        pipeline.manager = manager
        pipeline.initial_manager = manager
        pipeline.bot_registry = bot_registry
        pipeline.data_bot = data_bot
        helper = pipeline.helper
        if helper is not None and manager is not None:
            helper.manager = manager
        return pipeline

    bootstrap_attempts = {"count": 0}

    class _FailingManager:
        def __init__(self, *, bot_registry, data_bot):
            bootstrap_attempts["count"] += 1
            raise TypeError("preferred bootstrap path unavailable")

    class _StubCodeDB:
        pass

    class _StubMemory:
        pass

    class _StubEngine:
        def __init__(self, *_args, context_builder=None, **_kwargs):
            self.context_builder = context_builder

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
            ModelAutomationPipeline=_pipeline_factory
        ),
        "self_coding_manager": SimpleNamespace(SelfCodingManager=_StubManager),
    }

    def _load_optional_module(name: str, *, fallback: str | None = None):
        if name in modules:
            return modules[name]
        raise AssertionError(f"unexpected optional module {name}")

    monkeypatch.setattr(
        coding_bot_interface,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "_load_optional_module",
        _load_optional_module,
    )

    pipeline, _promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=_pipeline_factory,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
    )

    assert pipeline_box["pipeline"] is pipeline
    assert bootstrap_attempts["count"] == 0

    manager = coding_bot_interface._bootstrap_manager(
        "NestedPipelineBot", registry, data_bot
    )

    assert bootstrap_attempts["count"] == 1
    assert manager
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)
    assert "re-entrant initialisation depth" not in caplog.text

    helper = pipeline.helper
    assert isinstance(helper.initial_manager, coding_bot_interface._BootstrapManagerSentinel)
    assert manager.pipeline is pipeline
    assert pipeline.manager is manager
    assert pipeline.reattach_calls >= 1
    assert helper.manager is manager
    assert len(helper_instances) == 1


def _install_prepare_stub(module, monkeypatch):
    states: list[dict[str, bool]] = []

    def _prepare_pipeline_for_bootstrap(**_kwargs):
        state = {"promoted": False}
        pipeline = SimpleNamespace(manager=None)

        def _promote(_manager):
            state["promoted"] = True

        states.append(state)
        return pipeline, _promote

    monkeypatch.setattr(
        module, "prepare_pipeline_for_bootstrap", _prepare_pipeline_for_bootstrap, raising=False
    )
    return states


def _preload_watchdog_module(monkeypatch):
    import sys
    import types

    def _install_module(name: str, attrs: dict[str, object]) -> None:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    _install_module(
        "menace_sandbox.data_bot",
        {
            "DataBot": lambda *a, **k: SimpleNamespace(),
            "MetricsDB": lambda *a, **k: SimpleNamespace(
                fetch=lambda *_a, **_k: SimpleNamespace(empty=True),
                log_eval=lambda *_a, **_k: None,
            ),
            "persist_sc_thresholds": lambda *a, **k: None,
        },
    )
    _install_module(
        "menace_sandbox.bot_registry",
        {"BotRegistry": lambda *a, **k: SimpleNamespace(record_heartbeat=lambda *_: None)},
    )
    _install_module(
        "menace_sandbox.auto_escalation_manager",
        {"AutoEscalationManager": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.error_bot",
        {"ErrorDB": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.resource_allocation_optimizer",
        {"ROIDB": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.chaos_tester",
        {"ChaosTester": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.knowledge_graph",
        {"KnowledgeGraph": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.advanced_error_management",
        {
            "SelfHealingOrchestrator": lambda *a, **k: SimpleNamespace(heal=lambda *_: None),
            "PlaybookGenerator": lambda *a, **k: SimpleNamespace(),
        },
    )
    _install_module(
        "menace_sandbox.escalation_protocol",
        {
            "EscalationProtocol": lambda *a, **k: SimpleNamespace(),
            "EscalationLevel": lambda *a, **k: SimpleNamespace(),
        },
    )
    _install_module(
        "menace_sandbox.unified_event_bus",
        {"UnifiedEventBus": lambda *a, **k: SimpleNamespace(publish=lambda *_: None)},
    )
    _install_module(
        "menace_sandbox.model_automation_pipeline",
        {"ModelAutomationPipeline": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.code_database",
        {"CodeDB": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.menace_memory_manager",
        {"MenaceMemoryManager": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.self_coding_manager",
        {"internalize_coding_bot": lambda *a, **k: SimpleNamespace()},
    )
    _install_module(
        "menace_sandbox.self_coding_engine",
        {"SelfCodingEngine": lambda *a, **k: SimpleNamespace(context_builder=None)},
    )
    _install_module(
        "menace_sandbox.retry_utils",
        {"retry": lambda *a, **k: (lambda func: func)},
    )
    scope_utils = types.ModuleType("menace_sandbox.scope_utils")
    scope_utils.Scope = lambda value: value
    scope_utils.build_scope_clause = lambda *_a, **_k: ("", [])
    scope_utils.apply_scope = lambda query, clause: query
    monkeypatch.setitem(sys.modules, "menace_sandbox.scope_utils", scope_utils)
    dyn_router = types.ModuleType("menace_sandbox.dynamic_path_router")
    dyn_router.resolve_path = lambda path: path
    monkeypatch.setitem(sys.modules, "menace_sandbox.dynamic_path_router", dyn_router)
    db_router_mod = types.ModuleType("db_router")
    db_router_mod.GLOBAL_ROUTER = SimpleNamespace(
        get_connection=lambda *_: SimpleNamespace(execute=lambda *_: SimpleNamespace(fetchall=lambda: []))
    )
    monkeypatch.setitem(sys.modules, "db_router", db_router_mod)
    vec_pkg = types.ModuleType("vector_service")
    ctx_module = types.ModuleType("vector_service.context_builder")

    class _CtxBuilder:
        def refresh_db_weights(self) -> None:
            return None

    ctx_module.ContextBuilder = _CtxBuilder
    vec_pkg.context_builder = ctx_module
    monkeypatch.setitem(sys.modules, "vector_service", vec_pkg)
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", ctx_module)


def _preload_sandbox_module(monkeypatch):
    import db_router
    import dynamic_path_router
    import importlib.util
    import pathlib
    import types

    def _conn_factory(*_args, **_kwargs):
        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def execute(self, *_sql):
                return SimpleNamespace(fetchall=lambda: [], fetchone=lambda: None)

        return _Conn()

    router = SimpleNamespace(get_connection=_conn_factory)
    monkeypatch.setattr(db_router, "init_db_router", lambda *_a, **_k: router)
    monkeypatch.setattr(dynamic_path_router, "resolve_path", lambda path: pathlib.Path(path))
    monkeypatch.setattr(dynamic_path_router, "repo_root", lambda: pathlib.Path.cwd())
    monkeypatch.setattr(dynamic_path_router, "path_for_prompt", lambda *_a, **_k: pathlib.Path.cwd())

    def _install_module(name: str, attrs: dict[str, object]) -> None:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    _install_module(
        "dependency_hints",
        {"format_system_package_instructions": lambda *_a, **_k: ""},
    )
    _install_module(
        "log_tags",
        {
            "INSIGHT": "insight",
            "IMPROVEMENT_PATH": "improvement_path",
            "FEEDBACK": "feedback",
            "ERROR_FIX": "error_fix",
        },
    )

    class _LocalKnowledgeModule:
        def __init__(self):
            self.memory = SimpleNamespace()

    _local_module = _LocalKnowledgeModule()
    _install_module(
        "shared_knowledge_module",
        {
            "LOCAL_KNOWLEDGE_MODULE": _local_module,
            "LocalKnowledgeModule": _LocalKnowledgeModule,
        },
    )

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    class _SharedVectorService:
        def __init__(self, *args, **kwargs):
            self.embedder = kwargs.get("embedder")

        def vectorise_and_store(self, *_a, **_k):
            return []

    vec_module = types.ModuleType("vector_service")
    vec_module.ContextBuilder = _ContextBuilder
    vec_module.FallbackResult = object
    vec_module.ErrorResult = Exception
    vec_module.SharedVectorService = _SharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service", vec_module)

    vec_vectorizer = types.ModuleType("vector_service.vectorizer")
    vec_vectorizer.SharedVectorService = _SharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vec_vectorizer)

    class _Prompt:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    _install_module("prompt_types", {"Prompt": _Prompt})
    _install_module(
        "snippet_compressor",
        {"compress_snippets": lambda *_a, **_k: []},
    )
    _install_module(
        "context_builder_util",
        {
            "ensure_fresh_weights": lambda *_a, **_k: None,
            "create_context_builder": lambda *_a, **_k: SimpleNamespace(
                refresh_db_weights=lambda: None
            ),
        },
    )

    class _PromptBuildError(Exception):
        pass

    _install_module(
        "context_builder",
        {
            "handle_failure": lambda *_a, **_k: None,
            "PromptBuildError": _PromptBuildError,
        },
    )
    code_db_attrs = {
        "CodeDB": lambda *_a, **_k: SimpleNamespace(),
        "PatchRecord": SimpleNamespace,
    }
    _install_module("code_database", code_db_attrs)
    _install_module("menace_sandbox.code_database", code_db_attrs)

    def _install_menace_module(name: str, attrs: dict[str, object]) -> None:
        full_name = f"menace.{name}"
        mod = types.ModuleType(full_name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, full_name, mod)

    env_attrs = {
        "SANDBOX_EXTRA_METRICS": {},
        "SANDBOX_ENV_PRESETS": {},
        "simulate_execution_environment": lambda *_a, **_k: None,
        "simulate_full_environment": lambda *_a, **_k: None,
        "generate_sandbox_report": lambda *_a, **_k: None,
        "run_repo_section_simulations": lambda *_a, **_k: None,
        "run_workflow_simulations": lambda *_a, **_k: None,
        "run_scenarios": lambda *_a, **_k: None,
        "simulate_temporal_trajectory": lambda *_a, **_k: None,
        "_section_worker": lambda *_a, **_k: None,
        "validate_preset": lambda *_a, **_k: None,
        "_cleanup_pools": lambda *_a, **_k: None,
        "_await_cleanup_task": lambda *_a, **_k: None,
    }
    _install_module("sandbox_runner.environment", env_attrs)

    _install_module(
        "sandbox_runner.cli",
        {
            "_run_sandbox": lambda *_a, **_k: None,
            "rank_scenarios": lambda *_a, **_k: None,
            "main": lambda *_a, **_k: None,
        },
    )
    _install_module(
        "meta_workflow_planner",
        {"simulate_meta_workflow": lambda *_a, **_k: None},
    )

    class _Orchestrator:
        def __init__(self, *_, **__):
            self.manager = SimpleNamespace()

    thresholds = SimpleNamespace(
        roi_drop=1.0, error_increase=1.0, test_failure_increase=1.0
    )

    _install_menace_module(
        "unified_event_bus",
        {"UnifiedEventBus": lambda *_a, **_k: SimpleNamespace(publish=lambda *_: None)},
    )
    _install_menace_module("menace_orchestrator", {"MenaceOrchestrator": _Orchestrator})
    _install_menace_module(
        "self_improvement_policy", {"SelfImprovementPolicy": lambda *_a, **_k: None}
    )
    _install_menace_module(
        "self_improvement",
        {"SelfImprovementEngine": lambda *_a, **_k: SimpleNamespace()},
    )
    _install_menace_module(
        "patch_score_backend", {"backend_from_url": lambda *_a, **_k: SimpleNamespace()}
    )
    _install_menace_module(
        "self_test_service", {"SelfTestService": lambda *_a, **_k: SimpleNamespace()}
    )
    _install_menace_module(
        "code_database",
        {
            "PatchHistoryDB": lambda *_a, **_k: SimpleNamespace(),
            "CodeDB": lambda *_a, **_k: SimpleNamespace(),
        },
    )
    _install_menace_module(
        "patch_suggestion_db", {"PatchSuggestionDB": lambda *_a, **_k: SimpleNamespace()}
    )
    _install_menace_module("audit_trail", {"AuditTrail": lambda *_a, **_k: SimpleNamespace()})
    _install_menace_module(
        "error_bot",
        {
            "ErrorBot": lambda *_a, **_k: SimpleNamespace(),
            "ErrorDB": SimpleNamespace,
        },
    )
    _install_menace_module(
        "data_bot",
        {
            "MetricsDB": lambda *_a, **_k: SimpleNamespace(),
            "DataBot": lambda *_a, **_k: SimpleNamespace(),
            "persist_sc_thresholds": lambda *_a, **_k: None,
        },
    )
    _install_menace_module(
        "self_coding_thresholds", {"get_thresholds": lambda *_a, **_k: thresholds}
    )
    _install_menace_module(
        "composite_workflow_scorer",
        {"CompositeWorkflowScorer": lambda *_a, **_k: SimpleNamespace()},
    )
    _install_menace_module(
        "neuroplasticity", {"PathwayDB": lambda *_a, **_k: SimpleNamespace()}
    )
    _install_menace_module(
        "discrepancy_detection_bot",
        {"DiscrepancyDetectionBot": lambda *_a, **_k: SimpleNamespace()},
    )
    _install_menace_module(
        "error_logger", {"ErrorLogger": lambda *_a, **_k: SimpleNamespace()}
    )
    _install_menace_module(
        "knowledge_graph", {"KnowledgeGraph": lambda *_a, **_k: SimpleNamespace()}
    )
    _install_menace_module(
        "error_forecaster", {"ErrorForecaster": lambda *_a, **_k: SimpleNamespace()}
    )
    _install_menace_module(
        "quick_fix_engine",
        {
            "QuickFixEngine": lambda *_a, **_k: SimpleNamespace(),
            "QuickFixEngineError": type("QuickFixEngineError", (Exception,), {}),
            "generate_patch": lambda *_a, **_k: None,
        },
    )
    _install_menace_module(
        "coding_bot_interface",
        {
            "prepare_pipeline_for_bootstrap": lambda *_a, **_k: (
                SimpleNamespace(),
                lambda *_args, **_kwargs: None,
            )
        },
    )

    sys.modules.pop("menace_sandbox.sandbox_runner", None)
    runner_path = pathlib.Path(__file__).resolve().parents[1] / "sandbox_runner.py"
    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.sandbox_runner", runner_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_runner", module)
    assert spec and spec.loader is not None
    spec.loader.exec_module(module)


def _setup_genetic_case(module, monkeypatch):
    builder = SimpleNamespace()
    monkeypatch.setattr(
        "context_builder_util.create_context_builder", lambda: builder
    )
    monkeypatch.setattr(
        "menace_sandbox.self_coding_engine.SelfCodingEngine",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr("menace_sandbox.code_database.CodeDB", lambda: SimpleNamespace())
    def _gpt_memory_manager(*_a, **_k):
        # Accept the same keyword arguments as the production class, including
        # optional ``embedder`` hooks used when local knowledge bootstraps the
        # manager.
        return SimpleNamespace()

    monkeypatch.setattr(
        "menace_sandbox.gpt_memory.GPTMemoryManager", _gpt_memory_manager
    )
    monkeypatch.setattr(module, "persist_sc_thresholds", lambda *a, **k: None)
    thresholds = SimpleNamespace(roi_drop=1.0, error_increase=1.0, test_failure_increase=1.0)
    monkeypatch.setattr(
        "menace_sandbox.self_coding_thresholds.get_thresholds",
        lambda *_args: thresholds,
    )
    monkeypatch.setattr(
        "menace_sandbox.shared_evolution_orchestrator.get_orchestrator",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "menace_sandbox.threshold_service.ThresholdService",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(module, "_get_data_bot", lambda: SimpleNamespace())
    monkeypatch.setattr(module, "_get_registry", lambda: SimpleNamespace())

    def _internalize(*_a, **_k):
        return SimpleNamespace(evolution_orchestrator=None, quick_fix=None)

    monkeypatch.setattr(
        "menace_sandbox.self_coding_manager.internalize_coding_bot",
        _internalize,
    )
    return {}


def _setup_watchdog_case(module, monkeypatch):
    builder = SimpleNamespace(refresh_db_weights=lambda: None)
    module.DATA_BOT = SimpleNamespace()
    module.REGISTRY = SimpleNamespace(record_heartbeat=lambda *a, **k: None)
    module.UnifiedEventBus = lambda: SimpleNamespace(publish=lambda *a, **k: None)
    module.AutoEscalationManager = lambda *a, **k: SimpleNamespace()
    module.KnowledgeGraph = lambda *a, **k: SimpleNamespace()
    module.SelfHealingOrchestrator = lambda *a, **k: SimpleNamespace(heal=lambda *_a, **_k: None)
    module.EscalationLevel = lambda *a, **k: SimpleNamespace()
    module.EscalationProtocol = lambda *a, **k: SimpleNamespace()
    module.SelfCodingEngine = (
        lambda *a, **k: SimpleNamespace(context_builder=k.get("context_builder"))
    )
    module.CodeDB = lambda: SimpleNamespace()
    module.MenaceMemoryManager = lambda: SimpleNamespace()
    module.persist_sc_thresholds = lambda *a, **k: None
    thresholds = SimpleNamespace(roi_drop=1.0, error_increase=1.0, test_failure_increase=1.0)
    module._default_auto_handler = lambda *_: None

    def _internalize(*args, **_k):
        engine = args[1] if len(args) > 1 else SimpleNamespace()
        return SimpleNamespace(
            engine=engine, evolution_orchestrator=None, quick_fix=None
        )

    monkeypatch.setattr(module, "internalize_coding_bot", _internalize)

    metrics_db = SimpleNamespace(
        fetch=lambda *_a, **_k: SimpleNamespace(empty=True),
        log_eval=lambda *_a, **_k: None,
    )
    error_db = SimpleNamespace(
        _menace_id=lambda *_: "",
        conn=SimpleNamespace(
            execute=lambda *_a, **_k: SimpleNamespace(fetchall=lambda: [])
        ),
    )
    roi_db = SimpleNamespace()
    return {
        "builder": builder,
        "metrics_db": metrics_db,
        "error_db": error_db,
        "roi_db": roi_db,
        "bus": SimpleNamespace(publish=lambda *a, **k: None),
        "registry": module.REGISTRY,
    }


def _setup_sandbox_case(module, monkeypatch):
    builder = SimpleNamespace(refresh_db_weights=lambda: None)
    data_bot = SimpleNamespace()
    registry = SimpleNamespace()
    bus = SimpleNamespace(publish=lambda *a, **k: None)
    thresholds = SimpleNamespace(roi_drop=1.0, error_increase=1.0, test_failure_increase=1.0)
    import importlib

    sc_engine_mod = importlib.import_module("menace.self_coding_engine")
    monkeypatch.setattr(
        sc_engine_mod,
        "SelfCodingEngine",
        lambda *a, **k: SimpleNamespace(context_builder=k.get("context_builder")),
    )
    memory_mod = importlib.import_module("menace.menace_memory_manager")
    monkeypatch.setattr(memory_mod, "MenaceMemoryManager", lambda: SimpleNamespace())
    code_db_mod = importlib.import_module("menace.code_database")
    monkeypatch.setattr(code_db_mod, "CodeDB", lambda: SimpleNamespace(), raising=False)
    manager_mod = importlib.import_module("menace.self_coding_manager")
    monkeypatch.setattr(
        manager_mod,
        "internalize_coding_bot",
        lambda *a, **k: SimpleNamespace(evolution_orchestrator=None, quick_fix=None),
    )

    def _fake_thresholds(*_a, **_k):
        return thresholds

    monkeypatch.setattr(module, "get_thresholds", _fake_thresholds)
    monkeypatch.setattr(module, "persist_sc_thresholds", lambda *a, **k: None)
    return {
        "builder": builder,
        "data_bot": data_bot,
        "registry": registry,
        "bus": bus,
    }


def _invoke_genetic(module, _context):
    module._build_manager()


def _invoke_watchdog(module, context):
    module.Watchdog(
        context["error_db"],
        context["roi_db"],
        context["metrics_db"],
        context_builder=context["builder"],
        event_bus=context["bus"],
        registry=context["registry"],
    )


def _invoke_sandbox(module, context):
    module._bootstrap_sandbox_manager(
        context_builder=context["builder"],
        gpt_memory=object(),
        data_bot=context["data_bot"],
        registry=context["registry"],
        event_bus=context["bus"],
    )


ENTRYPOINT_CASES = (
    _EntryPointCase(
        name="genetic_manager",
        module_path="menace_sandbox.genetic_algorithm_bot",
        invoker=_invoke_genetic,
        setup=_setup_genetic_case,
    ),
    _EntryPointCase(
        name="watchdog_manager",
        module_path="menace_sandbox.watchdog",
        invoker=_invoke_watchdog,
        setup=_setup_watchdog_case,
        pre_reload=_preload_watchdog_module,
    ),
    _EntryPointCase(
        name="sandbox_manager",
        module_path="menace_sandbox.sandbox_runner",
        invoker=_invoke_sandbox,
        setup=_setup_sandbox_case,
        pre_reload=_preload_sandbox_module,
        reload=True,
    ),
)


@pytest.mark.parametrize("case", ENTRYPOINT_CASES, ids=lambda c: c.name)
def test_pipeline_entrypoints_promote_sentinel(monkeypatch, caplog, case):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    if case.pre_reload is not None:
        case.pre_reload(monkeypatch)
    module = importlib.import_module(case.module_path)
    if case.reload:
        module = importlib.reload(module)
    states = _install_prepare_stub(module, monkeypatch)
    context = case.setup(module, monkeypatch)
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)
    case.invoker(module, context)
    for state in states:
        if not state["promoted"]:
            coding_bot_interface.logger.warning(
                "re-entrant initialisation depth stub detected for %s", case.name
            )
            break
    assert "re-entrant initialisation depth" not in caplog.text

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

    def _prepare_pipeline_for_bootstrap(
        *,
        pipeline_cls,
        context_builder,
        bot_registry,
        data_bot,
        manager_override=None,
        manager_sentinel=None,
        sentinel_factory=None,
        **_,
    ):
        nonlocal helper_called
        helper_called = True
        sentinel = manager_sentinel or manager_override
        if sentinel is None:
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


@pytest.mark.parametrize("accepts_manager", [True, False])
def test_prepare_pipeline_promotes_placeholder_for_scripts(
    accepts_manager: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    class _Registry:
        def __init__(self) -> None:
            self.registered: list[tuple[str, object]] = []

        def register_bot(self, name: str, manager: object, **_kwargs: object) -> None:
            self.registered.append((name, manager))

        def update_bot(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _Thresholds:
        roi_drop = 0.0
        error_threshold = 0.0
        test_failure_threshold = 0.0

    class _DataBot:
        def reload_thresholds(self, _name: str) -> _Thresholds:
            return _Thresholds()

        def schedule_monitoring(self, _name: str) -> None:
            return None

    registry = _Registry()
    data_bot = _DataBot()

    def _registry_factory() -> _Registry:
        return registry

    def _data_bot_factory() -> _DataBot:
        return data_bot

    _registry_factory.__self_coding_lazy__ = True  # type: ignore[attr-defined]
    _data_bot_factory.__self_coding_lazy__ = True  # type: ignore[attr-defined]

    helper_instances: list[SimpleNamespace] = []

    @coding_bot_interface.self_coding_managed(
        bot_registry=_registry_factory,
        data_bot=_data_bot_factory,
    )
    class _HelperBot:
        def __init__(self, *, manager: object | None = None) -> None:
            self.manager = manager
            self.bot_registry = None
            self.data_bot = None
            helper_instances.append(self)

        def _finalize_self_coding_bootstrap(
            self,
            manager: object,
            *,
            registry: object | None = None,
            data_bot: object | None = None,
        ) -> None:
            self.manager = manager
            self.bot_registry = registry
            self.data_bot = data_bot

    class _PipelineAcceptsManager:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object | None = None,
            data_bot: object | None = None,
            manager: object | None = None,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            self.helpers = [_HelperBot(manager=manager), _HelperBot()]
            self._bots = list(self.helpers)
            self.comms_bot = None
            self.synthesis_bot = None
            self.diagnostic_manager = None
            self.planner = None
            self.aggregator = None
            self._attach_information_synthesis_manager = lambda: None

    class _PipelineWithoutManager:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object | None = None,
            data_bot: object | None = None,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = None
            self.initial_manager = None
            self.helpers = [_HelperBot(), _HelperBot()]
            self._bots = list(self.helpers)
            self.comms_bot = None
            self.synthesis_bot = None
            self.diagnostic_manager = None
            self.planner = None
            self.aggregator = None
            self._attach_information_synthesis_manager = lambda: None

    pipeline_cls = _PipelineAcceptsManager if accepts_manager else _PipelineWithoutManager
    builder = SimpleNamespace()

    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=pipeline_cls,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
    )

    placeholder = pipeline.manager
    assert isinstance(placeholder, coding_bot_interface._DisabledSelfCodingManager)
    assert all(helper.manager is placeholder for helper in pipeline.helpers)
    assert "re-entrant initialisation depth" not in caplog.text

    real_manager = SimpleNamespace(bot_registry=registry, data_bot=data_bot)
    promote(real_manager)

    assert pipeline.manager is real_manager
    assert all(helper.manager is real_manager for helper in pipeline.helpers)


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
        manager_override=None,
        manager_sentinel=None,
        sentinel_factory=None,
        **_,
    ):
        sentinel = manager_sentinel or manager_override
        if sentinel is None:
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
        manager_override=None,
        manager_sentinel=None,
        sentinel_factory=None,
        **_,
    ):
        sentinel = manager_sentinel or manager_override
        if sentinel is None:
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


def test_internalize_coding_bot_handles_prebuilt_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path,
) -> None:
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    class _Registry:
        def __init__(self) -> None:
            self.graph = SimpleNamespace(nodes={})
            self.modules: dict[str, str] = {}
            self.registered: list[tuple[str, object | None]] = []

        def register_bot(self, name: str, **kwargs: object) -> None:
            self.registered.append((name, kwargs.get("manager")))

        def update_bot(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _Thresholds:
        roi_drop = 0.1
        error_threshold = 0.2
        test_failure_threshold = 0.0

    class _DataBot:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(bot_thresholds={"ExampleBot": _Thresholds()})
            self.event_bus = SimpleNamespace(publish=lambda *_: None)

        def reload_thresholds(self, _name: str) -> _Thresholds:
            return _Thresholds()

        def schedule_monitoring(self, _name: str) -> None:
            return None

        def collect(self, *_args: object, **_kwargs: object) -> None:
            return None

    registry = _Registry()
    data_bot = _DataBot()

    helper_instances: list[SimpleNamespace] = []

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class _HelperBot:
        def __init__(self, *, manager: object | None = None, **_kwargs: object) -> None:
            self.manager = manager
            helper_instances.append(self)

        def _finalize_self_coding_bootstrap(
            self,
            manager: object,
            *,
            registry: object | None = None,
            data_bot: object | None = None,
        ) -> None:
            self.manager = manager

    class _Pipeline:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object,
            data_bot: object,
            manager: object,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            self._bots = [
                _HelperBot(
                    manager=manager,
                    bot_registry=bot_registry,
                    data_bot=data_bot,
                )
            ]

    builder = SimpleNamespace(name="builder")
    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=_Pipeline,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
    )

    module_path = tmp_path / "example_bot.py"
    module_path.write_text("print('stub bot')\n", encoding="utf-8")
    registry.graph.nodes["ExampleBot"] = {"module": str(module_path)}
    registry.modules["ExampleBot"] = str(module_path)

    import menace_sandbox.self_coding_manager as scm

    class _StubSelfCodingManager:
        def __init__(
            self,
            engine: object,
            pipeline_obj: object,
            *,
            bot_name: str,
            data_bot: object,
            bot_registry: object,
            **_kwargs: object,
        ) -> None:
            self.engine = engine
            self.pipeline = pipeline_obj
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry
            self.quick_fix = object()
            self.logger = logging.getLogger("test.internalize")
            self.event_bus = SimpleNamespace(publish=lambda *_: None)
            self.evolution_orchestrator = None

        def run_post_patch_cycle(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"self_tests": {"failed": 0}}

        def scan_repo(self) -> None:  # pragma: no cover - stubbed
            return None

    monkeypatch.setattr(scm, "SelfCodingManager", _StubSelfCodingManager)
    monkeypatch.setattr(scm, "persist_sc_thresholds", lambda *a, **k: None)

    engine = SimpleNamespace(name="engine")
    orchestrator = SimpleNamespace(
        provenance_token="token",
        register_bot=lambda *_: None,
        register_patch_cycle=lambda *_: None,
        event_bus=SimpleNamespace(publish=lambda *_: None),
    )

    manager = scm.internalize_coding_bot(
        bot_name="ExampleBot",
        engine=engine,
        pipeline=pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        evolution_orchestrator=orchestrator,
        roi_threshold=0.1,
        error_threshold=0.2,
        test_failure_threshold=0.0,
    )

    promote(manager)

    assert isinstance(manager, _StubSelfCodingManager)
    assert manager in [entry[1] for entry in registry.registered]
    assert all(
        isinstance(helper.manager, _StubSelfCodingManager) for helper in helper_instances
    )
    assert "re-entrant initialisation depth" not in caplog.text


def test_managerless_pipeline_bootstrap_injects_placeholder(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    class _Registry:
        def __init__(self):
            self.graph = SimpleNamespace(nodes={})

        def register_bot(self, name, module_path, **kwargs):
            node = self.graph.nodes.setdefault(name, {})
            node["module_path"] = module_path
            node["selfcoding_manager"] = kwargs.get("manager")

        def update_bot(self, *_args, **_kwargs):  # pragma: no cover - helper stub
            return None

        def mark_bot_patch_in_progress(self, *_args, **_kwargs):  # pragma: no cover
            return None

    class _DataBot:
        def reload_thresholds(self, *_args, **_kwargs):
            return SimpleNamespace(
                roi_drop=0.1,
                error_threshold=0.2,
                test_failure_threshold=0.3,
            )

        def schedule_monitoring(self, *_args, **_kwargs):  # pragma: no cover
            return None

    registry = _Registry()
    data_bot = _DataBot()
    policy = SimpleNamespace(
        allowlist=None,
        denylist=frozenset(),
        is_enabled=lambda _name: True,
    )
    monkeypatch.setattr(coding_bot_interface, "get_self_coding_policy", lambda: policy)

    helper_manager_states: list[object] = []
    pipeline_manager_states: list[object] = []

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class _PipelineHelper:
        name = "ManagerlessPipelineHelper"

        def __init__(self, *, manager=None, **_kwargs):
            helper_manager_states.append(manager)
            assert manager is not None, "helper should never see manager=None"
            self.manager = manager
            self.initial_manager = manager
            self.finalized_with = None

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.manager = manager
            self.initial_manager = manager
            self.finalized_with = (registry, data_bot)

    class _ManagerlessPipeline:
        def __init__(self, *, context_builder, bot_registry=None, data_bot=None):
            snapshot = getattr(self, "manager", None)
            pipeline_manager_states.append(snapshot)
            assert snapshot is not None, "pipeline should receive a placeholder"
            self.context_builder = context_builder
            self.manager = snapshot
            self.initial_manager = getattr(self, "initial_manager", snapshot)
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.communication_bot = _PipelineHelper(manager=self.manager)
            self.comms_bot = self.communication_bot
            self._bots = [self.communication_bot]
            self.reattach_calls = 0
            self.finalized_with = None

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

        def _finalize_self_coding_bootstrap(self, manager, *, registry=None, data_bot=None):
            self.finalized_with = (manager, registry, data_bot)

    _install_managerless_bootstrap(monkeypatch, _ManagerlessPipeline)

    manager = coding_bot_interface._bootstrap_manager(
        "ManagerlessBot", registry, data_bot
    )

    assert manager
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)
    assert "re-entrant initialisation depth" not in caplog.text
    assert helper_manager_states
    assert all(state is not None for state in helper_manager_states)
    assert pipeline_manager_states
    assert all(
        isinstance(state, coding_bot_interface._DisabledSelfCodingManager)
        for state in pipeline_manager_states
    )

    pipeline = manager.pipeline
    helper = pipeline.communication_bot
    assert helper.manager is manager
    assert helper.initial_manager is manager
    assert pipeline.manager is manager
    assert pipeline.initial_manager is manager
    node = registry.graph.nodes.get(_PipelineHelper.name)
    assert node["selfcoding_manager"] is manager


def test_resolve_helpers_promotes_managerless_registry(monkeypatch):
    class _Registry:
        def __init__(self):
            self.graph = SimpleNamespace(nodes={})
            self.promotions: list[tuple[str, object]] = []

        def register_bot(self, name, module_path, **kwargs):
            node = self.graph.nodes.setdefault(name, {})
            node["module_path"] = module_path
            node["selfcoding_manager"] = kwargs.get("manager")

        def promote_self_coding_manager(self, name, manager, *_args, **_kwargs):
            self.promotions.append((name, manager))

        def update_bot(self, *_args, **_kwargs):  # pragma: no cover - helper stub
            return None

    class _DataBot:
        def reload_thresholds(self, *_args, **_kwargs):
            return SimpleNamespace(
                roi_drop=0.1,
                error_threshold=0.2,
                test_failure_threshold=0.3,
            )

        def schedule_monitoring(self, *_args, **_kwargs):  # pragma: no cover
            return None

    registry = _Registry()
    data_bot = _DataBot()
    policy = SimpleNamespace(
        allowlist=None,
        denylist=frozenset(),
        is_enabled=lambda _name: True,
    )
    monkeypatch.setattr(coding_bot_interface, "get_self_coding_policy", lambda: policy)

    pipeline_helper_states: list[object] = []

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class _PipelineHelper:
        name = "PipelineCommsHelper"

        def __init__(self, *, manager=None, **_kwargs):
            pipeline_helper_states.append(manager)
            assert manager is not None
            self.manager = manager
            self.initial_manager = manager

    class _ManagerlessPipeline:
        def __init__(self, *, context_builder, bot_registry=None, data_bot=None):
            snapshot = getattr(self, "manager", None)
            assert snapshot is not None
            self.context_builder = context_builder
            self.manager = snapshot
            self.initial_manager = getattr(self, "initial_manager", snapshot)
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.communication_bot = _PipelineHelper(manager=self.manager)
            self.comms_bot = self.communication_bot
            self._bots = [self.communication_bot]

        def _attach_information_synthesis_manager(self):  # pragma: no cover - stub
            return None

    _install_managerless_bootstrap(monkeypatch, _ManagerlessPipeline)

    helper_instances: list[object] = []

    @coding_bot_interface.self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
    )
    class _StandaloneHelper:
        name = "StandaloneManagerlessHelper"

        def __init__(self, *, manager=None, **_kwargs):
            helper_instances.append(self)
            self.manager = manager
            self.bot_registry = _kwargs.get("bot_registry") or registry
            self.data_bot = _kwargs.get("data_bot") or data_bot

    helper = _StandaloneHelper()
    assert helper.manager
    assert helper.manager.pipeline.communication_bot.manager is helper.manager
    assert helper.manager.pipeline.initial_manager is helper.manager
    comms_helper = helper.manager.pipeline.communication_bot
    assert comms_helper.manager is helper.manager
    assert comms_helper.initial_manager is helper.manager
    node = registry.graph.nodes.get(_StandaloneHelper.name)
    assert node["selfcoding_manager"] is helper.manager
    assert pipeline_helper_states
    assert all(state is not None for state in pipeline_helper_states)


def test_prepare_pipeline_avoids_nested_bootstrap_manager(monkeypatch):
    registry = SimpleNamespace(name="registry")
    data_bot = SimpleNamespace(name="data")
    builder = SimpleNamespace(name="ctx")

    bootstrap_calls: list[str] = []

    def _forbidden_bootstrap(name, *_args, **_kwargs):
        bootstrap_calls.append(name)
        raise AssertionError("_bootstrap_manager should not be re-invoked")

    monkeypatch.setattr(coding_bot_interface, "_bootstrap_manager", _forbidden_bootstrap)

    class _StubHelper:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            if manager is None:
                coding_bot_interface._bootstrap_manager("NestedHelper", bot_registry, data_bot)
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot

    class _StubPipeline:
        def __init__(
            self,
            *,
            context_builder,
            bot_registry=None,
            data_bot=None,
            manager=None,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            helper = _StubHelper(
                manager=manager, bot_registry=bot_registry, data_bot=data_bot
            )
            self.helper = helper
            self._bots = [helper]

        def _attach_information_synthesis_manager(self):
            return None

    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=_StubPipeline,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
    )
    manager = SimpleNamespace(bot_registry=registry, data_bot=data_bot)
    promote(manager)
    assert pipeline.manager is manager
    assert bootstrap_calls == []


def test_bootstrap_entrypoint_handles_stubbed_communication_bot(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)

    import menace_sandbox.bootstrap_self_coding as bootstrap_self_coding

    bootstrap_self_coding = importlib.reload(bootstrap_self_coding)

    builder = SimpleNamespace(name="context")
    monkeypatch.setattr(
        "menace_sandbox.context_builder_util.create_context_builder",
        lambda: builder,
    )

    class _Registry:
        def register_bot(self, *_args, **_kwargs):
            return None

        def update_bot(self, *_args, **_kwargs):
            return None

    class _DataBot:
        def __init__(self, *_, **__):
            self.thresholds = {}

        def reload_thresholds(self, *_args, **_kwargs):
            return SimpleNamespace(
                roi_drop=0.1, error_threshold=0.2, test_failure_threshold=0.3
            )

        def schedule_monitoring(self, *_args, **_kwargs):
            return None

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
        "menace_sandbox.data_bot.persist_sc_thresholds", lambda *_, **__: None
    )

    reentrant_state = {"triggered": False}

    class _StubCommunicationMaintenanceBot:
        def __init__(self, *, manager=None, bot_registry=None, data_bot=None):
            self.manager = manager
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            if manager is None:
                reentrant_state["triggered"] = True
                coding_bot_interface.logger.warning(
                    "re-entrant initialisation depth=1"
                )
                self.manager = coding_bot_interface._DisabledSelfCodingManager(
                    bot_registry=bot_registry,
                    data_bot=data_bot,
                )

    comms_module = types.ModuleType("menace_sandbox.communication_maintenance_bot")
    comms_module.CommunicationMaintenanceBot = _StubCommunicationMaintenanceBot
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.communication_maintenance_bot", comms_module
    )
    monkeypatch.setitem(sys.modules, "communication_maintenance_bot", comms_module)

    class _StubPipeline:
        def __init__(
            self,
            *,
            context_builder,
            bot_registry=None,
            data_bot=None,
            manager=None,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            helper = comms_module.CommunicationMaintenanceBot(
                manager=manager,
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
            self.communication_helper = helper
            self._bots = [helper]
            self.reattach_calls = 0

        def _attach_information_synthesis_manager(self):
            self.reattach_calls += 1

    pipeline_module = types.ModuleType("menace_sandbox.model_automation_pipeline")
    pipeline_module.ModelAutomationPipeline = _StubPipeline
    pipeline_module.AutomationResult = object
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.model_automation_pipeline", pipeline_module
    )
    monkeypatch.setitem(sys.modules, "model_automation_pipeline", pipeline_module)

    class _StubManager:
        def __init__(self, *, pipeline, bot_registry, data_bot, bot_name):
            self.pipeline = pipeline
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.bot_name = bot_name
            self.quick_fix = object()
            self.logger = logging.getLogger("stub-manager")

        def __bool__(self):
            return True

    manager_box: dict[str, _StubManager] = {}

    def _internalize_coding_bot(
        *,
        bot_name,
        pipeline,
        bot_registry,
        data_bot,
        **_kwargs,
    ):
        if reentrant_state["triggered"]:
            result = coding_bot_interface._DisabledSelfCodingManager(
                bot_registry=bot_registry,
                data_bot=data_bot,
            )
        else:
            result = _StubManager(
                pipeline=pipeline,
                bot_registry=bot_registry,
                data_bot=data_bot,
                bot_name=bot_name,
            )
        manager_box["manager"] = result
        return result

    monkeypatch.setattr(
        "menace_sandbox.self_coding_manager.internalize_coding_bot",
        _internalize_coding_bot,
    )

    bootstrap_self_coding.bootstrap_self_coding("ExampleBot")

    assert reentrant_state["triggered"] is False
    assert "re-entrant initialisation depth=1" not in caplog.text
    manager = manager_box["manager"]
    assert not isinstance(manager, coding_bot_interface._DisabledSelfCodingManager)


def _install_stub_module(monkeypatch, name: str, **entries: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(entries)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize(
    "pipeline_fixture",
    ["prebuilt_managerless_pipeline_env", "reentrant_prebuilt_pipeline_env"],
)
def test_cli_bootstrap_promotes_helper_graphs(
    monkeypatch, caplog, request, pipeline_fixture
):
    env = request.getfixturevalue(pipeline_fixture)

    caplog.set_level(logging.WARNING, logger=coding_bot_interface.logger.name)
    caplog.set_level(logging.WARNING, logger=bootstrap_self_coding.LOGGER.name)

    helpers_before = tuple(getattr(env.pipeline, "_bots", ()))
    threshold_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    prepare_calls: list[dict[str, object]] = []
    internalize_state: dict[str, object] = {}

    _install_stub_module(
        monkeypatch,
        "menace_sandbox.bot_registry",
        BotRegistry=lambda: env.registry,
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.code_database",
        CodeDB=lambda: SimpleNamespace(kind="code-db"),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.context_builder_util",
        create_context_builder=lambda: env.builder,
    )

    def _data_bot_factory(start_server: bool = False) -> object:
        assert start_server is False
        return env.data_bot

    def _persist_sc_thresholds(*args: object, **kwargs: object) -> None:
        threshold_calls.append((args, kwargs))

    _install_stub_module(
        monkeypatch,
        "menace_sandbox.data_bot",
        DataBot=_data_bot_factory,
        persist_sc_thresholds=_persist_sc_thresholds,
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.menace_memory_manager",
        MenaceMemoryManager=lambda: SimpleNamespace(kind="memory"),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.model_automation_pipeline",
        ModelAutomationPipeline=env.pipeline_cls,
    )

    class _StubEngine:
        def __init__(self, *_args: object, context_builder: object, **_kwargs: object):
            self.context_builder = context_builder

    _install_stub_module(
        monkeypatch,
        "menace_sandbox.self_coding_engine",
        SelfCodingEngine=_StubEngine,
    )

    def _thresholds(_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            roi_drop=0.1,
            error_increase=0.2,
            test_failure_increase=0.3,
        )

    _install_stub_module(
        monkeypatch,
        "menace_sandbox.self_coding_thresholds",
        get_thresholds=_thresholds,
    )

    def _prepare_pipeline_for_bootstrap(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        prepare_calls.append(kwargs)
        assert kwargs.get("pipeline_cls") is env.pipeline_cls
        assert kwargs.get("context_builder") is env.builder
        assert kwargs.get("bot_registry") is env.registry
        assert kwargs.get("data_bot") is env.data_bot
        return env.pipeline, env.promoter

    monkeypatch.setattr(
        coding_bot_interface,
        "prepare_pipeline_for_bootstrap",
        _prepare_pipeline_for_bootstrap,
    )

    def _internalize_coding_bot(**kwargs: object) -> SimpleNamespace:
        internalize_state["kwargs"] = kwargs
        pipeline = kwargs["pipeline"]
        internalize_state["pipeline_manager"] = getattr(pipeline, "manager", None)
        manager = SimpleNamespace(
            pipeline=pipeline,
            bot_registry=kwargs["bot_registry"],
            data_bot=kwargs["data_bot"],
            bot_name=kwargs["bot_name"],
        )
        internalize_state["manager"] = manager
        return manager

    _install_stub_module(
        monkeypatch,
        "menace_sandbox.self_coding_manager",
        internalize_coding_bot=_internalize_coding_bot,
    )

    bot_name = f"Cli{pipeline_fixture.title()}"
    bootstrap_self_coding.bootstrap_self_coding(bot_name)

    assert prepare_calls, "expected prepare_pipeline_for_bootstrap to be invoked"
    assert threshold_calls, "persist_sc_thresholds should capture CLI thresholds"
    assert "kwargs" in internalize_state
    assert internalize_state["kwargs"]["bot_name"] == bot_name
    assert internalize_state["kwargs"]["pipeline"] is env.pipeline
    received_manager = internalize_state["pipeline_manager"]
    if received_manager is not None:
        assert isinstance(
            received_manager, coding_bot_interface._BootstrapManagerSentinel
        )

    manager = internalize_state["manager"]
    pipeline_manager = getattr(env.pipeline, "manager", None)
    promote_calls = getattr(env, "promote_calls", None)
    if promote_calls is not None:
        assert promote_calls, "expected prepare_pipeline_for_bootstrap to promote manager"
        assert manager in promote_calls
    if pipeline_manager is not None:
        assert pipeline_manager is manager
    for helper in helpers_before:
        helper_manager = getattr(helper, "manager", None)
        assert helper_manager is manager
        assert not isinstance(
            helper_manager, coding_bot_interface._DisabledSelfCodingManager
        )

    assert "re-entrant initialisation depth" not in caplog.text
