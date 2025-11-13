from __future__ import annotations

import logging
import sys
from typing import Callable
from types import ModuleType, SimpleNamespace

import pytest


class DummyRegistry:
    def __init__(self) -> None:
        self.registered: list[tuple[str, dict[str, object]]] = []
        self.updated: list[tuple[str, dict[str, object]]] = []
        self.promotions: list[tuple[str, object]] = []

    def register_bot(self, name: str, module_path: str, **kwargs: object) -> None:
        self.registered.append((name, dict(kwargs)))

    def update_bot(self, name: str, module_path: str, **kwargs: object) -> None:
        self.updated.append((name, dict(kwargs)))

    def promote_self_coding_manager(
        self, name: str, manager: object, data_bot: object, **_kwargs: object
    ) -> None:
        self.promotions.append((name, manager))


class DummyDataBot:
    def reload_thresholds(self, name: str) -> SimpleNamespace:  # pragma: no cover - simple data holder
        return SimpleNamespace(
            roi_drop=0.0,
            error_threshold=0.0,
            test_failure_threshold=0.0,
        )


@pytest.fixture
def stub_bootstrap_env(monkeypatch) -> dict[str, ModuleType]:
    import menace.coding_bot_interface as cbi

    module_map: dict[str, ModuleType] = {}

    def _install(name: str, module: ModuleType) -> None:
        module_map[name] = module

    code_db_mod = ModuleType("code_database")
    class _CodeDB:  # pragma: no cover - simple stub
        pass
    code_db_mod.CodeDB = _CodeDB  # type: ignore[attr-defined]
    _install("code_database", code_db_mod)

    memory_mod = ModuleType("gpt_memory")
    class _MemoryManager:  # pragma: no cover - simple stub
        pass
    memory_mod.GPTMemoryManager = _MemoryManager  # type: ignore[attr-defined]
    _install("gpt_memory", memory_mod)

    engine_mod = ModuleType("self_coding_engine")

    class _StubEngine:
        def __init__(self, *_args: object, context_builder: object | None = None) -> None:
            self.context_builder = context_builder

    class _ManagerContext:
        def set(self, _value: object) -> object:  # pragma: no cover - simple token
            return object()

        def reset(self, _token: object) -> None:  # pragma: no cover - simple stub
            return None

    engine_mod.SelfCodingEngine = _StubEngine  # type: ignore[attr-defined]
    engine_mod.MANAGER_CONTEXT = _ManagerContext()  # type: ignore[attr-defined]
    _install("self_coding_engine", engine_mod)

    manager_mod = ModuleType("self_coding_manager")

    class _StubManager:
        def __init__(
            self,
            engine: object,
            pipeline: object,
            *,
            bot_name: str,
            data_bot: object,
            bot_registry: object,
        ) -> None:
            self.engine = engine
            self.pipeline = pipeline
            self.bot_name = bot_name
            self.data_bot = data_bot
            self.bot_registry = bot_registry
            self.evolution_orchestrator = SimpleNamespace()

        def register(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
            return None

    manager_mod.SelfCodingManager = _StubManager  # type: ignore[attr-defined]
    _install("self_coding_manager", manager_mod)

    pipeline_mod = ModuleType("model_automation_pipeline")
    _install("model_automation_pipeline", pipeline_mod)

    thresholds_mod = ModuleType("self_coding_thresholds")
    thresholds_mod.update_thresholds = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    thresholds_mod._load_config = lambda *args, **kwargs: {}  # type: ignore[attr-defined]
    sys.modules.setdefault("self_coding_thresholds", thresholds_mod)
    sys.modules.setdefault("menace_sandbox.self_coding_thresholds", thresholds_mod)

    real_load_internal = cbi.load_internal

    def _fake_load_internal(name: str) -> ModuleType:
        module = module_map.get(name)
        if module is not None:
            return module
        return real_load_internal(name)

    monkeypatch.setattr(cbi, "load_internal", _fake_load_internal)

    return module_map


@pytest.fixture
def prebuilt_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Prepare a pipeline instance before ``_bootstrap_manager`` executes."""

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("fallback to manual pipeline")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="prebuilt")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class NestedHelper:
        name = "PrebuiltNestedHelper"

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class CommunicationMaintenanceBot:
        name = "PrebuiltComms"

        def __init__(self) -> None:
            self.helper = NestedHelper()

    class StubPipeline:
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
            self.comms_bot = CommunicationMaintenanceBot()
            self._bots = [self.comms_bot, self.comms_bot.helper]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        StubPipeline
    )

    real_prepare = cbi.prepare_pipeline_for_bootstrap
    pipeline, promoter = real_prepare(
        pipeline_cls=StubPipeline,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
    )

    assert isinstance(pipeline.manager, cbi._BootstrapManagerSentinel)
    assert pipeline.comms_bot.manager is pipeline.manager
    assert pipeline.comms_bot.helper.manager is pipeline.manager

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        pipeline_cls=StubPipeline,
        pipeline=pipeline,
        promoter=promoter,
        helper_names=("PrebuiltComms", "PrebuiltNestedHelper"),
    )




@pytest.fixture
def reentrant_prebuilt_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Build a pipeline whose helper instantiation re-enters ``_bootstrap_manager``.

    The pipeline is constructed *before* the test exercises
    :func:`coding_bot_interface._bootstrap_manager`.  During construction the
    ``CommunicationMaintenanceBot`` immediately instantiates a nested helper that
    is also decorated with :func:`self_coding_managed`.  The nested helper's
    decorator resolves helpers eagerly and re-enters ``_bootstrap_manager`` while
    the first call is still running.  This fixture mirrors the production
    bootstrap regression so the accompanying test can assert the final manager
    promotion remains stable.
    """

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("manager bootstrap fallback")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="reentrant")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_names = ("ReentrantComms", "ReentrantNestedHelper")

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ReentrantNestedHelper:
        name = helper_names[1]

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ReentrantCommunicationBot:
        name = helper_names[0]

        def __init__(self) -> None:
            self.helper = ReentrantNestedHelper()

    class ReentrantPipeline:
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
            self.comms_bot = ReentrantCommunicationBot()
            self._bots = [self.comms_bot, self.comms_bot.helper]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        ReentrantPipeline
    )

    pipeline, promoter = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=ReentrantPipeline,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
    )

    assert isinstance(pipeline.manager, cbi._BootstrapManagerSentinel)
    assert pipeline.comms_bot.manager is pipeline.manager
    assert pipeline.comms_bot.helper.manager is pipeline.manager

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        pipeline_cls=ReentrantPipeline,
        pipeline=pipeline,
        promoter=promoter,
        helper_names=helper_names,
    )


@pytest.fixture
def fallback_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Create a pipeline whose helpers normalise ``manager`` via truthiness checks."""

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("pipeline bootstrap fallback")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="fallback")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_names = ("LegacyCommunicationBot", "LegacyNestedHelper")

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class LegacyNestedHelper:
        name = helper_names[1]

        def __init__(self, manager: object | None = None) -> None:
            self.bot_name = self.name
            self.manager = manager
            self.initial_manager = manager

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class LegacyCommunicationBot:
        name = helper_names[0]

        def __init__(self, manager: object | None = None) -> None:
            manager = manager or None
            self.bot_name = self.name
            self.manager = manager
            self.initial_manager = manager
            self.helper = LegacyNestedHelper(manager=manager)

    class LegacyPipeline:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object,
            data_bot: object,
            manager: object,
            **_kwargs: object,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.comms_bot = LegacyCommunicationBot(manager=manager)
            self._bots = [self.comms_bot, self.comms_bot.helper]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        LegacyPipeline
    )

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        helper_names=helper_names,
    )


def test_pipeline_bootstrap_promotes_sentinel(stub_bootstrap_env: dict[str, ModuleType], caplog: pytest.LogCaptureFixture) -> None:
    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class NestedHelper:
        name = "NestedHelper"

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class CommunicationMaintenanceBot:
        name = "PipelineComms"

        def __init__(self) -> None:
            self.helper = NestedHelper()

    class StubPipeline:
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
            self.comms_bot = CommunicationMaintenanceBot()
            helpers = [self.comms_bot]
            helper_nested = getattr(self.comms_bot, "helper", None)
            if helper_nested is not None:
                helpers.append(helper_nested)
            self._bots = helpers

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = StubPipeline  # type: ignore[attr-defined]

    caplog.set_level(logging.WARNING)
    manager = cbi._bootstrap_manager("PipelineOwner", registry, data_bot)

    assert not any("re-entrant" in record.message for record in caplog.records)

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager

    registered_names = {entry[0] for entry in registry.registered}
    assert registered_names == {"PipelineComms", "NestedHelper"}
    assert {entry[0] for entry in registry.promotions} == {
        "PipelineComms",
        "NestedHelper",
    }
    assert all(entry[1] is manager for entry in registry.promotions)


def test_prebuilt_pipeline_helpers_promoted_without_reentrant_warning(
    prebuilt_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    def _reuse_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        assert kwargs.get("pipeline_cls") is prebuilt_pipeline_env.pipeline_cls
        return prebuilt_pipeline_env.pipeline, prebuilt_pipeline_env.promoter

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_pipeline)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "PrebuiltOwner",
            prebuilt_pipeline_env.registry,
            prebuilt_pipeline_env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert prebuilt_pipeline_env.pipeline.manager is manager
    assert prebuilt_pipeline_env.pipeline.comms_bot.manager is manager
    assert prebuilt_pipeline_env.pipeline.comms_bot.helper.manager is manager
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    registered_names = {entry[0] for entry in prebuilt_pipeline_env.registry.registered}
    assert registered_names == set(prebuilt_pipeline_env.helper_names)
    assert all(
        entry[1].get("manager") is manager
        for entry in prebuilt_pipeline_env.registry.registered
    )
    assert {entry[0] for entry in prebuilt_pipeline_env.registry.promotions} == set(
        prebuilt_pipeline_env.helper_names
    )
    assert all(
        entry[1] is manager for entry in prebuilt_pipeline_env.registry.promotions
    )


def test_prebuilt_reentrant_pipeline_stabilises_manager(
    reentrant_prebuilt_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression test for helper decorators re-entering ``_bootstrap_manager``.

    The pipeline is instantiated before calling ``_bootstrap_manager`` so its
    ``@self_coding_managed`` helpers already kicked off a nested bootstrap while
    the first call was still constructing the fallback pipeline.  The test
    ensures the subsequent promotion replaces every helper's manager with the
    real instance and no re-entrant warnings leak into the logs.
    """

    import menace.coding_bot_interface as cbi

    def _reuse_prebuilt_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        assert kwargs.get("pipeline_cls") is reentrant_prebuilt_pipeline_env.pipeline_cls
        return (
            reentrant_prebuilt_pipeline_env.pipeline,
            reentrant_prebuilt_pipeline_env.promoter,
        )

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_prebuilt_pipeline)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "ReentrantOwner",
            reentrant_prebuilt_pipeline_env.registry,
            reentrant_prebuilt_pipeline_env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert reentrant_prebuilt_pipeline_env.pipeline.manager is manager
    assert reentrant_prebuilt_pipeline_env.pipeline.comms_bot.manager is manager
    assert (
        reentrant_prebuilt_pipeline_env.pipeline.comms_bot.helper.manager is manager
    )
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    helper_names = set(reentrant_prebuilt_pipeline_env.helper_names)
    registered_names = {
        entry[0] for entry in reentrant_prebuilt_pipeline_env.registry.registered
    }
    assert registered_names == helper_names
    assert all(
        entry[1].get("manager") is manager
        for entry in reentrant_prebuilt_pipeline_env.registry.registered
    )
    assert {
        entry[0] for entry in reentrant_prebuilt_pipeline_env.registry.promotions
    } == helper_names
    assert all(
        entry[1] is manager for entry in reentrant_prebuilt_pipeline_env.registry.promotions
    )


def test_bootstrap_manager_handles_truthy_owner_sentinel(
    fallback_pipeline_env: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    caplog.set_level(logging.WARNING, logger=cbi.logger.name)
    manager = cbi._bootstrap_manager(
        "LegacyPipelineOwner",
        fallback_pipeline_env.registry,
        fallback_pipeline_env.data_bot,
    )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager

    registered_names = {entry[0] for entry in fallback_pipeline_env.registry.registered}
    assert registered_names == set(fallback_pipeline_env.helper_names)
    assert {entry[0] for entry in fallback_pipeline_env.registry.promotions} == set(
        fallback_pipeline_env.helper_names
    )
    assert all(
        entry[1] is manager for entry in fallback_pipeline_env.registry.promotions
    )


def test_bootstrap_manager_warns_when_owner_sentinel_is_falsy(
    fallback_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    original_activate = cbi._activate_bootstrap_sentinel

    def _regression_activate(manager: object | None) -> Callable[[], None]:  # pragma: no cover - regression hook
        restore = original_activate(manager)
        try:
            delattr(cbi._BOOTSTRAP_STATE, "sentinel_manager")
        except AttributeError:
            pass
        return restore

    monkeypatch.setattr(cbi, "_activate_bootstrap_sentinel", _regression_activate)

    caplog.set_level(logging.WARNING, logger=cbi.logger.name)
    manager = cbi._bootstrap_manager(
        "LegacyPipelineOwner",
        fallback_pipeline_env.registry,
        fallback_pipeline_env.data_bot,
    )

    assert any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None

    registered_names = {entry[0] for entry in fallback_pipeline_env.registry.registered}
    assert registered_names == set(fallback_pipeline_env.helper_names)
    assert all(
        isinstance(entry[1].get("manager"), cbi._DisabledSelfCodingManager)
        for entry in fallback_pipeline_env.registry.registered
    )
