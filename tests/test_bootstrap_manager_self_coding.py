from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Callable, Iterable
from types import ModuleType, SimpleNamespace

import pytest


class DummyRegistry:
    def __init__(self) -> None:
        self.registered: list[tuple[str, dict[str, object]]] = []
        self.updated: list[tuple[str, dict[str, object]]] = []
        self.promotions: list[tuple[str, object]] = []

    def register_bot(
        self, name: str, module_path: str | None = None, **kwargs: object
    ) -> None:
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
def prebuilt_manager_kwarg_reject_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Pipeline that rejects the ``manager`` kwarg during ``prepare_pipeline``."""

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("typeerror pipeline fallback")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="prebuilt-manager-kwarg-reject")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_names = ("TypeErrorComms", "TypeErrorNestedHelper")

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class TypeErrorNestedHelper:
        name = helper_names[1]

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class TypeErrorCommunicationBot:
        name = helper_names[0]

        def __init__(self) -> None:
            self.bot_name = self.name
            self.helper = TypeErrorNestedHelper()

    class TypeErrorPipeline:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object | None = None,
            data_bot: object | None = None,
            **kwargs: object,
        ) -> None:
            if "manager" in kwargs:
                raise TypeError("__init__() got an unexpected keyword argument 'manager'")
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = None
            self.initial_manager = None
            self.comms_bot = TypeErrorCommunicationBot()
            self._bots = [self.comms_bot, self.comms_bot.helper]

    pipeline_mod = stub_bootstrap_env["model_automation_pipeline"]
    pipeline_mod.ModelAutomationPipeline = TypeErrorPipeline  # type: ignore[attr-defined]

    pipeline, promoter = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=TypeErrorPipeline,
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
        pipeline_cls=TypeErrorPipeline,
        pipeline=pipeline,
        promoter=promoter,
        helper_names=helper_names,
        communication_cls=TypeErrorCommunicationBot,
        nested_cls=TypeErrorNestedHelper,
    )


@pytest.fixture
def prebuilt_managerless_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Build a pipeline whose helpers bootstrap themselves before tests run."""

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

    builder = SimpleNamespace(label="prebuilt-managerless")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_names = (
        "PrebuiltManagerlessComms",
        "PrebuiltManagerlessNested",
        "PrebuiltManagerlessPredictor",
        "PrebuiltManagerlessHandoff",
        "PrebuiltManagerlessDB",
    )
    placeholder_manager = cbi._DisabledSelfCodingManager(
        bot_registry=registry,
        data_bot=data_bot,
    )
    bootstrap_attempts: list[str] = []

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class PrebuiltManagerlessNestedHelper:
        name = helper_names[1]

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class PrebuiltManagerlessCommunicationBot:
        name = helper_names[0]

        def __init__(self) -> None:
            self.bot_name = self.name
            self.helper = PrebuiltManagerlessNestedHelper()

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class PrebuiltManagerlessPredictor:
        name = helper_names[2]

        def __init__(self) -> None:
            self.bot_name = self.name
            self.data_bot = None
            self.capital_bot = None

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class PrebuiltManagerlessHandoff:
        name = helper_names[3]

        def __init__(self) -> None:
            self.bot_name = self.name
            self.event_bus = None

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class PrebuiltManagerlessDatabaseBot:
        name = helper_names[4]

        def __init__(self) -> None:
            self.bot_name = self.name

    class PrebuiltManagerlessPipeline:
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
            self._comms_builder = PrebuiltManagerlessCommunicationBot
            self._predictor_builder = PrebuiltManagerlessPredictor
            self._handoff_builder = PrebuiltManagerlessHandoff
            self._db_builder = PrebuiltManagerlessDatabaseBot
            self.comms_bot: PrebuiltManagerlessCommunicationBot | None = None
            self.predictor: PrebuiltManagerlessPredictor | None = None
            self.handoff: PrebuiltManagerlessHandoff | None = None
            self.db_bot: PrebuiltManagerlessDatabaseBot | None = None
            self._bots: list[object] = []
            self._maybe_build_helpers(manager)

        def _maybe_build_helpers(self, manager: object | None) -> None:
            if manager is None:
                return

            def _register_helper(helper: object | None) -> None:
                if helper is None or self.bot_registry is None:
                    return
                name = getattr(helper, "bot_name", getattr(helper, "name", None))
                if not name:
                    return
                self.bot_registry.register_bot(
                    name,
                    module_path=None,
                    manager=manager,
                    data_bot=self.data_bot,
                )

            if self.comms_bot is None:
                comms_bot = self._comms_builder()
                comms_bot.manager = manager
                comms_bot.helper.manager = manager
                self.comms_bot = comms_bot
                _register_helper(comms_bot)
                _register_helper(getattr(comms_bot, "helper", None))
            if self.predictor is None:
                predictor = self._predictor_builder()
                predictor.manager = manager
                self.predictor = predictor
                _register_helper(predictor)
            if self.handoff is None:
                handoff = self._handoff_builder()
                handoff.manager = manager
                self.handoff = handoff
                _register_helper(handoff)
            if self.db_bot is None:
                db_bot = self._db_builder()
                db_bot.manager = manager
                self.db_bot = db_bot
                _register_helper(db_bot)
            self._bots = [
                helper
                for helper in (
                    self.comms_bot,
                    getattr(self.comms_bot, "helper", None),
                    self.predictor,
                    self.handoff,
                    self.db_bot,
                )
                if helper is not None
            ]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        PrebuiltManagerlessPipeline
    )

    with monkeypatch.context() as patch_context:
        def _capture_reentrant_bootstrap(
            name: str, bot_registry: object, data_bot_obj: object
        ) -> object:
            bootstrap_attempts.append(name)
            return placeholder_manager

        patch_context.setattr(cbi, "_bootstrap_manager", _capture_reentrant_bootstrap)
        pipeline = PrebuiltManagerlessPipeline(
            context_builder=builder,
            bot_registry=registry,
            data_bot=data_bot,
            manager=None,
        )

    assert not bootstrap_attempts  # helpers should defer until manager exists

    promote_calls: list[object] = []

    def _promote(manager: object) -> None:
        promote_calls.append(manager)
        sentinel_types: tuple[type[object], ...] = ()
        sentinel_cls = getattr(cbi, "_BootstrapOwnerSentinel", None)
        if sentinel_cls is not None:
            sentinel_types = (sentinel_cls, cbi._DisabledSelfCodingManager)
        current_manager = pipeline.manager
        is_placeholder = (
            current_manager in (None, placeholder_manager)
            or isinstance(current_manager, sentinel_types)
        )
        if not is_placeholder and manager is not current_manager:
            return
        cbi._promote_pipeline_manager(pipeline, manager, placeholder_manager)
        pipeline.manager = manager
        pipeline.initial_manager = manager
        pipeline._maybe_build_helpers(manager)

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        pipeline_cls=PrebuiltManagerlessPipeline,
        pipeline=pipeline,
        promoter=_promote,
        promote_calls=promote_calls,
        helper_names=helper_names,
        bootstrap_attempts=tuple(bootstrap_attempts),
        placeholder=placeholder_manager,
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
def script_fallback_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Minimal fallback pipeline whose helpers re-enter ``_bootstrap_manager``.

    The pipeline class is intentionally lightweight so tests can simulate the
    user's "one-off script" where helper bootstrap re-enters
    :func:`coding_bot_interface._bootstrap_manager` while the fallback pipeline
    is still being assembled.  ``ModelAutomationPipeline`` is registered on the
    stub module but **not** instantiated via ``prepare_pipeline_for_bootstrap``
    so the tests exercise the full bootstrap flow end-to-end.
    """

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("fallback to sentinel pipeline")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="script-fallback")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_names = ("ScriptFallbackComms", "ScriptFallbackNestedHelper")

    bootstrap_calls: list[str] = []
    real_bootstrap = cbi._bootstrap_manager

    def _tracking_bootstrap(name: str, *args: object, **kwargs: object) -> object:
        bootstrap_calls.append(name)
        return real_bootstrap(name, *args, **kwargs)

    monkeypatch.setattr(cbi, "_bootstrap_manager", _tracking_bootstrap)

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ScriptFallbackNestedHelper:
        name = helper_names[1]

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ScriptFallbackCommunicationBot:
        name = helper_names[0]

        def __init__(self) -> None:
            self.bot_name = self.name
            self.helper = ScriptFallbackNestedHelper()

    class ScriptFallbackPipeline:
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
            self.comms_bot = ScriptFallbackCommunicationBot()
            self._bots = [self.comms_bot, self.comms_bot.helper]

    pipeline_mod = stub_bootstrap_env["model_automation_pipeline"]
    pipeline_mod.ModelAutomationPipeline = ScriptFallbackPipeline  # type: ignore[attr-defined]

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        helper_names=helper_names,
        pipeline_cls=ScriptFallbackPipeline,
        pipeline_module=pipeline_mod,
        communication_cls=ScriptFallbackCommunicationBot,
        nested_cls=ScriptFallbackNestedHelper,
        bootstrap_calls=bootstrap_calls,
    )


@pytest.fixture
def direct_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Build a pipeline directly so helpers bootstrap themselves mid-build."""

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("direct pipeline fallback")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="direct-prebuilt")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_names = ("DirectComms", "DirectNestedHelper")
    placeholder_manager = cbi._BootstrapManagerSentinel(
        bot_registry=registry,
        data_bot=data_bot,
    )

    bootstrap_attempts: list[str] = []

    with monkeypatch.context() as patch_context:
        def _placeholder_bootstrap(name: str, *_args: object, **_kwargs: object) -> object:
            bootstrap_attempts.append(name)
            return placeholder_manager

        patch_context.setattr(cbi, "_bootstrap_manager", _placeholder_bootstrap)

        @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
        class DirectNestedHelper:
            name = helper_names[1]

            def __init__(self) -> None:
                self.bot_name = self.name

        @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
        class DirectCommunicationMaintenanceBot:
            name = helper_names[0]

            def __init__(self) -> None:
                self.bot_name = self.name
                self.helper = DirectNestedHelper()

        class DirectPipeline:
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
                self.comms_bot = DirectCommunicationMaintenanceBot()
                self._bots = [self.comms_bot, self.comms_bot.helper]

        pipeline = DirectPipeline(
            context_builder=builder,
            bot_registry=registry,
            data_bot=data_bot,
            manager=placeholder_manager,
        )

    assert bootstrap_attempts  # ensure helpers attempted bootstrap eagerly

    promotion_calls: list[object] = []

    def _promote(manager: object, *, extra_sentinels: Iterable[object] | None = None) -> None:
        promotion_calls.append(manager)
        cbi._promote_pipeline_manager(
            pipeline,
            manager,
            placeholder_manager,
            extra_sentinels=extra_sentinels,
        )

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = DirectPipeline  # type: ignore[attr-defined]

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        pipeline=pipeline,
        pipeline_cls=DirectPipeline,
        promoter=_promote,
        helper_names=helper_names,
        bootstrap_attempts=tuple(bootstrap_attempts),
        placeholder=placeholder_manager,
    )


@pytest.fixture
def real_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Instantiate the actual ``ModelAutomationPipeline`` with lightweight helpers."""

    import sys
    import menace.coding_bot_interface as cbi

    def _bot_class(label: str) -> type:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.name = label
            self.manager = kwargs.get("manager")
            self.initial_manager = kwargs.get("manager")

        return type(label, (), {"__init__": __init__})

    def _install_module(name: str, entries: dict[str, object]) -> None:
        module = ModuleType(name)
        module.__dict__.update(entries)
        monkeypatch.setitem(sys.modules, name, module)

    def _install_dual(name: str, entries: dict[str, object]) -> None:
        for prefix in ("menace.", "menace_sandbox."):
            _install_module(f"{prefix}{name}", dict(entries))

    module_specs: dict[str, dict[str, object]] = {
        "resource_prediction_bot": {
            "ResourcePredictionBot": _bot_class("ResourcePredictionBot"),
            "ResourceMetrics": type("ResourceMetrics", (), {}),
        },
        "data_interfaces": {
            "DataBotInterface": _bot_class("DataBotInterface"),
        },
        "task_handoff_bot": {
            "TaskHandoffBot": _bot_class("TaskHandoffBot"),
            "TaskInfo": type("TaskInfo", (), {}),
            "TaskPackage": type("TaskPackage", (), {}),
            "WorkflowDB": _bot_class("WorkflowDB"),
        },
        "efficiency_bot": {"EfficiencyBot": _bot_class("EfficiencyBot")},
        "performance_assessment_bot": {
            "PerformanceAssessmentBot": _bot_class("PerformanceAssessmentBot"),
        },
        "operational_monitor_bot": {
            "OperationalMonitoringBot": _bot_class("OperationalMonitoringBot"),
        },
        "central_database_bot": {
            "CentralDatabaseBot": _bot_class("CentralDatabaseBot"),
            "Proposal": type("Proposal", (), {}),
        },
        "sentiment_bot": {"SentimentBot": _bot_class("SentimentBot")},
        "query_bot": {"QueryBot": _bot_class("QueryBot")},
        "memory_bot": {"MemoryBot": _bot_class("MemoryBot")},
        "offer_testing_bot": {"OfferTestingBot": _bot_class("OfferTestingBot")},
        "research_fallback_bot": {
            "ResearchFallbackBot": _bot_class("ResearchFallbackBot"),
        },
        "ai_counter_bot": {"AICounterBot": _bot_class("AICounterBot")},
        "idea_search_bot": {"KeywordBank": _bot_class("KeywordBank")},
        "newsreader_bot": {"NewsDB": _bot_class("NewsDB")},
        "investment_engine": {
            "AutoReinvestmentBot": _bot_class("AutoReinvestmentBot"),
        },
        "revenue_amplifier": {
            "RevenueSpikeEvaluatorBot": _bot_class("RevenueSpikeEvaluatorBot"),
            "CapitalAllocationBot": _bot_class("CapitalAllocationBot"),
            "RevenueEventsDB": _bot_class("RevenueEventsDB"),
        },
        "bot_db_utils": {"wrap_bot_methods": lambda bot, *_a, **_k: bot},
        "database_manager": {"update_model": lambda *args, **kwargs: None},
        "db_router": {
            "DBRouter": type(
                "DBRouter",
                (),
                {
                    "__init__": lambda self, *args, **kwargs: None,
                    "get_connection": lambda self, *_args, **_kwargs: SimpleNamespace(
                        execute=lambda *a, **k: None,
                        close=lambda: None,
                    ),
                },
            ),
            "GLOBAL_ROUTER": SimpleNamespace(
                get_connection=lambda *_a, **_k: SimpleNamespace(
                    execute=lambda *a, **k: None,
                    close=lambda: None,
                ),
                close=lambda: None,
            ),
            "init_db_router": lambda *_a, **_k: SimpleNamespace(
                get_connection=lambda *_args, **_kwargs: SimpleNamespace(
                    execute=lambda *a, **k: None,
                    close=lambda: None,
                ),
                close=lambda: None,
            ),
        },
        "unified_event_bus": {
            "UnifiedEventBus": type(
                "UnifiedEventBus",
                (),
                {
                    "__init__": lambda self, *args, **kwargs: None,
                    "publish": lambda self, *_args, **_kwargs: None,
                },
            )
        },
        "neuroplasticity": {
            "Outcome": type("Outcome", (), {}),
            "PathwayDB": _bot_class("PathwayDB"),
            "PathwayRecord": type("PathwayRecord", (), {}),
        },
        "communication_maintenance_bot": {
            "CommunicationMaintenanceBot": _bot_class("CommunicationMaintenanceBot"),
        },
        "capital_management_bot": {
            "CapitalManagementBot": _bot_class("CapitalManagementBot"),
        },
        "resource_allocation_optimizer": {
            "ResourceAllocationOptimizer": _bot_class("ResourceAllocationOptimizer"),
        },
    }
    for module_name, attributes in module_specs.items():
        _install_dual(module_name, attributes)

    _install_module(
        "vector_service.context_builder",
        {
            "ContextBuilder": _bot_class("ContextBuilder"),
            "record_failed_tags": lambda *_args, **_kwargs: None,
            "load_failed_tags": lambda *_args, **_kwargs: [],
        },
    )
    _install_module(
        "menace.shared.lazy_data_bot",
        {"create_data_bot": lambda *_args, **_kwargs: SimpleNamespace(name="DataBot")},
    )
    _install_module(
        "menace_sandbox.shared.lazy_data_bot",
        {"create_data_bot": lambda *_args, **_kwargs: SimpleNamespace(name="DataBot")},
    )

    from entry_pipeline_loader import load_pipeline_class

    pipeline_cls = load_pipeline_class()
    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = pipeline_cls

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("force legacy bootstrap")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="real-pipeline")
    builder.refresh_db_weights = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    def _patched_init(
        self,
        *args: object,
        context_builder: object,
        bot_registry: object | None = None,
        data_bot: object | None = None,
        manager: object | None = None,
        **_kwargs: object,
    ) -> None:
        self.context_builder = context_builder
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.manager = manager
        self.initial_manager = manager
        helpers: list[SimpleNamespace] = []

        def _make_helper(label: str) -> SimpleNamespace:
            helper = SimpleNamespace(name=label, manager=manager, initial_manager=manager)
            helpers.append(helper)
            return helper

        self.comms_bot = _make_helper("CommunicationMaintenanceBot")
        self.comms_bot.helper = _make_helper("CommsNestedHelper")
        self.db_bot = _make_helper("CentralDatabaseBot")
        self.monitor_bot = _make_helper("OperationalMonitoringBot")
        self._bots = helpers

    monkeypatch.setattr(pipeline_cls, "__init__", _patched_init)

    pipeline, promoter = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=pipeline_cls,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
    )

    helpers = tuple(getattr(pipeline, "_bots", ()))

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        pipeline_cls=pipeline_cls,
        pipeline=pipeline,
        promoter=promoter,
        helpers=helpers,
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


@pytest.fixture
def prebuilt_communication_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    """Build a pipeline whose communication bot instantiates helpers eagerly."""

    import menace.coding_bot_interface as cbi

    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("MENACE_MODE", "development")

    import menace.communication_maintenance_bot as comms_mod

    comms_mod = importlib.reload(comms_mod)

    class _StubContextBuilder:
        def query(self, *_args: object, **_kwargs: object) -> str:
            return ""

        def refresh_db_weights(self) -> None:  # pragma: no cover - trivial stub
            return None

    monkeypatch.setattr(comms_mod, "ContextBuilder", _StubContextBuilder)
    monkeypatch.setattr(comms_mod, "load_templates", lambda: {})
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "flush_queue",
        lambda self: None,
    )
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "fetch_cluster_data",
        lambda self: None,
    )
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "_ensure_repo",
        lambda self, repo_path: SimpleNamespace(path=repo_path),
    )
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "_create_scheduler",
        lambda self, _broker: SimpleNamespace(),
    )

    class _StubMaintenanceDB:
        def __init__(self) -> None:
            self._state: dict[str, object] = {}

        def get_state(self, key: str) -> object | None:
            return self._state.get(key)

        def set_state(self, key: str, value: object) -> None:
            self._state[key] = value

    class _StubDBRouter:
        def query_all(self, _term: str) -> None:  # pragma: no cover - noop
            return None

        def get_connection(self, _name: str) -> SimpleNamespace:  # pragma: no cover - noop
            class _Conn(SimpleNamespace):
                def execute(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
                    return None

                def close(self) -> None:  # pragma: no cover - noop
                    return None

            return _Conn()

    class _StubCommStore:
        def cleanup(self) -> None:  # pragma: no cover - noop
            return None

    class _StubEventBus:
        def subscribe(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
            return None

    class _StubMemoryMgr:
        def subscribe(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
            return None

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("force communication pipeline bootstrap")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = _StubContextBuilder()
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    class PrebuiltCommunicationPipeline:
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
            comm_store = _StubCommStore()
            config = comms_mod.MaintenanceBotConfig(
                comm_log_path=tmp_path / "comm.log",
                message_queue=tmp_path / "queue.jsonl",
                error_db_path=tmp_path / "errors.db",
                log_dir=tmp_path,
                comm_log_max_size_mb=1.0,
            )
            self.comms_bot = comms_mod.CommunicationMaintenanceBot(
                db=_StubMaintenanceDB(),
                error_bot=None,
                repo_path=tmp_path,
                broker=None,
                db_router=_StubDBRouter(),
                event_bus=_StubEventBus(),
                memory_mgr=_StubMemoryMgr(),
                context_builder=context_builder,
                webhook_urls=[],
                comm_store=comm_store,
                config=config,
                admin_tokens=[],
                manager=manager,
            )
            self._bots = [self.comms_bot]
            helper = getattr(self.comms_bot, "error_bot", None)
            if helper is not None:
                self._bots.append(helper)
                self.error_helper = helper

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        PrebuiltCommunicationPipeline
    )

    def _build_pipeline() -> tuple[object, Callable[[object], None]]:
        pipeline, promoter = cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=PrebuiltCommunicationPipeline,
            context_builder=builder,
            bot_registry=registry,
            data_bot=data_bot,
        )
        assert isinstance(pipeline.manager, cbi._BootstrapManagerSentinel)
        assert pipeline.comms_bot.manager is pipeline.manager
        helper = getattr(pipeline.comms_bot, "error_bot", None)
        if helper is not None:
            assert helper.manager is pipeline.manager
        return pipeline, promoter

    pipeline, promoter = _build_pipeline()

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        pipeline_cls=PrebuiltCommunicationPipeline,
        pipeline=pipeline,
        promoter=promoter,
        rebuild_pipeline=_build_pipeline,
    )


def test_reentrant_helper_instantiated_before_manager_assignment(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("force placeholder pipeline bootstrap")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="pre-init placeholder")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_name = "PreInitHelper"

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class PreInitHelper:
        name = helper_name

        def __init__(self, manager: object | None = None) -> None:
            self.bot_name = self.name
            self.initial_manager = manager
            self.manager = manager

    class PreInitPipeline:
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
            placeholder = getattr(self, "manager", None)
            self.initial_manager = placeholder
            self.helper = PreInitHelper(manager=placeholder)
            self.manager = manager
            self._bots = [self.helper]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        PreInitPipeline
    )

    caplog.clear()
    caplog.set_level(logging.WARNING, logger=cbi.logger.name)
    manager = cbi._bootstrap_manager("PreInitOwner", registry, data_bot)

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.manager is manager
    assert pipeline.helper.manager is manager
    assert pipeline.initial_manager is manager
    assert pipeline.helper.initial_manager is manager
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )


@pytest.fixture
def managerless_pipeline_env(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Pipeline whose constructor rejects ``manager`` but still instantiates helpers."""

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("managerless pipeline bootstrap fallback")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="managerless")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_names = ("ManagerlessComms", "ManagerlessNestedHelper")

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ManagerlessNestedHelper:
        name = helper_names[1]

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ManagerlessCommunicationBot:
        name = helper_names[0]

        def __init__(self) -> None:
            self.bot_name = self.name
            self.helper = ManagerlessNestedHelper()

    class ManagerlessPipeline:
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
            self.comms_bot = ManagerlessCommunicationBot()
            self._bots = [self.comms_bot, self.comms_bot.helper]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        ManagerlessPipeline
    )

    return SimpleNamespace(
        registry=registry,
        data_bot=data_bot,
        builder=builder,
        helper_names=helper_names,
        pipeline_cls=ManagerlessPipeline,
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


def test_fallback_pipeline_promotes_real_communication_bot_manager(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    import menace.coding_bot_interface as cbi

    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("MENACE_MODE", "development")

    import menace.communication_maintenance_bot as comms_mod

    comms_mod = importlib.reload(comms_mod)

    class _StubContextBuilder:
        def query(self, *_args: object, **_kwargs: object) -> str:
            return ""

        def refresh_db_weights(self) -> None:  # pragma: no cover - trivial stub
            return None

    monkeypatch.setattr(comms_mod, "ContextBuilder", _StubContextBuilder)
    monkeypatch.setattr(comms_mod, "load_templates", lambda: {})
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "flush_queue",
        lambda self: None,
    )
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "fetch_cluster_data",
        lambda self: None,
    )
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "_ensure_repo",
        lambda self, repo_path: SimpleNamespace(path=repo_path),
    )
    monkeypatch.setattr(
        comms_mod.CommunicationMaintenanceBot,
        "_create_scheduler",
        lambda self, _broker: SimpleNamespace(),
    )

    class _StubMaintenanceDB:
        def __init__(self) -> None:
            self._state: dict[str, object] = {}

        def get_state(self, key: str) -> object | None:
            return self._state.get(key)

        def set_state(self, key: str, value: object) -> None:
            self._state[key] = value

    class _StubDBRouter:
        def query_all(self, _term: str) -> None:  # pragma: no cover - noop
            return None

        def get_connection(self, _name: str) -> SimpleNamespace:  # pragma: no cover - noop
            class _Conn(SimpleNamespace):
                def execute(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
                    return None

                def close(self) -> None:  # pragma: no cover - noop
                    return None

            return _Conn()

    class _StubCommStore:
        def cleanup(self) -> None:  # pragma: no cover - noop
            return None

    class _StubEventBus:
        def subscribe(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
            return None

    class _StubMemoryMgr:
        def subscribe(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
            return None

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("force fallback pipeline bootstrap")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = _StubContextBuilder()
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    class LegacyCommsPipeline:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object | None = None,
            data_bot: object | None = None,
            manager: object | None = None,
            **_kwargs: object,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            comm_store = _StubCommStore()
            config = comms_mod.MaintenanceBotConfig(
                comm_log_path=tmp_path / "comm.log",
                message_queue=tmp_path / "queue.jsonl",
                error_db_path=tmp_path / "errors.db",
                comm_log_max_size_mb=1.0,
            )
            self.comms_bot = comms_mod.CommunicationMaintenanceBot(
                db=_StubMaintenanceDB(),
                error_bot=None,
                repo_path=tmp_path,
                broker=None,
                db_router=_StubDBRouter(),
                event_bus=_StubEventBus(),
                memory_mgr=_StubMemoryMgr(),
                context_builder=context_builder,
                webhook_urls=[],
                comm_store=comm_store,
                config=config,
                admin_tokens=[],
                manager=manager,
            )
            helper = getattr(self.comms_bot, "error_bot", None)
            bots = [self.comms_bot]
            if helper is not None:
                bots.append(helper)
                self.error_helper = helper
            self._bots = bots

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        LegacyCommsPipeline
    )

    caplog.set_level(logging.WARNING)
    manager = cbi._bootstrap_manager("LegacyCommsOwner", registry, data_bot)

    assert not any(
        "re-entrant initialisation depth" in record.message for record in caplog.records
    )
    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    helper = getattr(pipeline.comms_bot, "error_bot", None)
    if helper is not None:
        assert helper.manager is manager
        assert not isinstance(helper.manager, cbi._DisabledSelfCodingManager)


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


def test_prebuilt_managerless_pipeline_promotes_helpers(
    prebuilt_managerless_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    def _reuse_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        assert kwargs.get("pipeline_cls") is prebuilt_managerless_pipeline_env.pipeline_cls
        return (
            prebuilt_managerless_pipeline_env.pipeline,
            prebuilt_managerless_pipeline_env.promoter,
        )

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_pipeline)

    pipeline = prebuilt_managerless_pipeline_env.pipeline
    assert pipeline.manager is None
    assert pipeline.comms_bot is None
    assert pipeline.predictor is None
    assert pipeline.handoff is None
    assert pipeline.db_bot is None
    assert pipeline._bots == []

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "PrebuiltManagerlessOwner",
            prebuilt_managerless_pipeline_env.registry,
            prebuilt_managerless_pipeline_env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    pipeline = prebuilt_managerless_pipeline_env.pipeline
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager
    assert pipeline.predictor.manager is manager
    assert pipeline.handoff.manager is manager
    assert pipeline.db_bot.manager is manager
    assert [helper.bot_name for helper in pipeline._bots] == list(
        prebuilt_managerless_pipeline_env.helper_names
    )

    helper_names = set(prebuilt_managerless_pipeline_env.helper_names)
    registered_names = {
        entry[0] for entry in prebuilt_managerless_pipeline_env.registry.registered
    }
    assert registered_names == helper_names
    promoted_names = {
        entry[0] for entry in prebuilt_managerless_pipeline_env.registry.promotions
    }
    assert promoted_names == helper_names
    assert all(
        entry[1] is manager
        for entry in prebuilt_managerless_pipeline_env.registry.promotions
    )
    assert not prebuilt_managerless_pipeline_env.bootstrap_attempts


def test_prebuilt_managerless_pipeline_defers_helpers_until_manager(
    prebuilt_managerless_pipeline_env: SimpleNamespace,
) -> None:
    pipeline = prebuilt_managerless_pipeline_env.pipeline

    assert pipeline.manager is None
    assert pipeline.initial_manager is None
    assert pipeline.comms_bot is None
    assert pipeline.predictor is None
    assert pipeline.handoff is None
    assert pipeline.db_bot is None
    assert pipeline._bots == []
    assert not prebuilt_managerless_pipeline_env.bootstrap_attempts


def test_prebuilt_reentrant_pipeline_stabilises_manager(
    reentrant_prebuilt_pipeline_env: SimpleNamespace,
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

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "ReentrantOwner",
            reentrant_prebuilt_pipeline_env.registry,
            reentrant_prebuilt_pipeline_env.data_bot,
            pipeline=reentrant_prebuilt_pipeline_env.pipeline,
            pipeline_manager=reentrant_prebuilt_pipeline_env.pipeline.manager,
            pipeline_promoter=reentrant_prebuilt_pipeline_env.promoter,
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


def test_prebuilt_pipeline_helpers_receive_real_manager(
    reentrant_prebuilt_pipeline_env: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ensure pipelines constructed before bootstrap still see real managers."""

    import menace.coding_bot_interface as cbi

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "PrebuiltManagerOwner",
            reentrant_prebuilt_pipeline_env.registry,
            reentrant_prebuilt_pipeline_env.data_bot,
            pipeline=reentrant_prebuilt_pipeline_env.pipeline,
            pipeline_manager=reentrant_prebuilt_pipeline_env.pipeline.manager,
            pipeline_promoter=reentrant_prebuilt_pipeline_env.promoter,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    pipeline = reentrant_prebuilt_pipeline_env.pipeline
    assert pipeline.manager is manager

    helpers = (
        pipeline.comms_bot,
        getattr(pipeline.comms_bot, "helper", None),
    )
    for helper in helpers:
        if helper is None:
            continue
        assert getattr(helper, "manager", None) is manager
        initial = getattr(helper, "initial_manager", manager)
        assert initial is manager
        assert not isinstance(helper.manager, cbi._DisabledSelfCodingManager)

    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )


def test_prebuilt_manager_kwarg_reject_pipeline_promotes_helpers(
    prebuilt_manager_kwarg_reject_env: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression for ``_typeerror_rejects_manager`` shim and sentinel promotion.

    The stub ``ModelAutomationPipeline`` refuses the ``manager`` kwarg so
    ``prepare_pipeline_for_bootstrap`` first raises ``TypeError`` before the
    shim retries without the kwarg.  The pipeline is instantiated prior to
    calling :func:`coding_bot_interface._bootstrap_manager`, mirroring the user
    script that triggered the regression.  The test ensures no re-entrant
    warnings leak, helpers share the sentinel during bootstrap, and repeated
    bootstrap attempts reuse the promoted manager instead of returning a
    :class:`_DisabledSelfCodingManager`.
    """

    import menace.coding_bot_interface as cbi

    env = prebuilt_manager_kwarg_reject_env
    pipeline = env.pipeline
    sentinel = pipeline.manager
    assert isinstance(sentinel, cbi._BootstrapManagerSentinel)
    assert pipeline.comms_bot.manager is sentinel
    assert pipeline.comms_bot.helper.manager is sentinel

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "PrebuiltManagerKwargRejectOwner",
            env.registry,
            env.data_bot,
            pipeline=pipeline,
            pipeline_manager=sentinel,
            pipeline_promoter=env.promoter,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    repeated_helper = env.communication_cls()
    assert repeated_helper.manager is manager
    assert repeated_helper.helper.manager is manager
    assert not isinstance(repeated_helper.manager, cbi._DisabledSelfCodingManager)


def test_script_fallback_pipeline_bootstrap_promotes_helpers(
    script_fallback_pipeline_env: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    env = script_fallback_pipeline_env
    env.bootstrap_calls.clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "ScriptFallbackOwner",
            env.registry,
            env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    repeated_helper = env.communication_cls()
    assert repeated_helper.manager is manager
    assert repeated_helper.helper.manager is manager
    assert not isinstance(repeated_helper.manager, cbi._DisabledSelfCodingManager)

    helper_names = set(env.helper_names)
    assert {entry[0] for entry in env.registry.registered} == helper_names
    assert {entry[0] for entry in env.registry.promotions} == helper_names
    assert all(entry[1] is manager for entry in env.registry.promotions)


def test_script_fallback_pipeline_shim_handles_managerless_constructor(
    script_fallback_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    env = script_fallback_pipeline_env
    env.bootstrap_calls.clear()

    manager_rejections: list[object] = []

    class ManagerlessScriptPipeline:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object | None = None,
            data_bot: object | None = None,
            **kwargs: object,
        ) -> None:
            if "manager" in kwargs:
                manager_rejections.append(kwargs["manager"])
                raise TypeError("unexpected manager kwarg for script pipeline")
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = None
            self.initial_manager = None
            self.comms_bot = env.communication_cls()
            self._bots = [self.comms_bot, self.comms_bot.helper]

    monkeypatch.setattr(
        env.pipeline_module,
        "ModelAutomationPipeline",
        ManagerlessScriptPipeline,
        raising=False,
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "ScriptFallbackShimOwner",
            env.registry,
            env.data_bot,
        )

    assert manager_rejections  # ensure constructor rejected the manager kwarg
    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    repeated_helper = env.communication_cls()
    assert repeated_helper.manager is manager
    assert repeated_helper.helper.manager is manager
    assert not isinstance(repeated_helper.manager, cbi._DisabledSelfCodingManager)

    helper_names = set(env.helper_names)
    assert {entry[0] for entry in env.registry.registered} == helper_names
    assert {entry[0] for entry in env.registry.promotions} == helper_names
    assert all(entry[1] is manager for entry in env.registry.promotions)


def test_direct_pipeline_promotes_prebuilt_helpers(
    direct_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    assert set(direct_pipeline_env.bootstrap_attempts) == set(
        direct_pipeline_env.helper_names
    )

    def _reuse_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        assert kwargs.get("pipeline_cls") is direct_pipeline_env.pipeline_cls
        return direct_pipeline_env.pipeline, direct_pipeline_env.promoter

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_pipeline)

    real_bootstrap = cbi._bootstrap_manager
    call_depth = 0

    def _guarded_bootstrap(*args: object, **kwargs: object) -> object:
        nonlocal call_depth
        if call_depth:
            raise AssertionError("_bootstrap_manager re-entered during direct pipeline bootstrap")
        call_depth += 1
        try:
            return real_bootstrap(*args, **kwargs)
        finally:
            call_depth -= 1

    monkeypatch.setattr(cbi, "_bootstrap_manager", _guarded_bootstrap)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "DirectPipelineOwner",
            direct_pipeline_env.registry,
            direct_pipeline_env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)

    pipeline = direct_pipeline_env.pipeline
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager

    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )


def test_manual_pipeline_helper_reuses_placeholder_manager(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ensure script-style helper bootstrap reuses fallback sentinels safely."""

    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    helper_name = "ManualFallbackHelper"
    owner_name = "ManualFallbackOwner"

    class _FailingManager:
        def __init__(self, *, bot_registry: object, data_bot: object) -> None:
            raise TypeError("force legacy fallback path")

    monkeypatch.setattr(
        cbi,
        "_resolve_self_coding_manager_cls",
        lambda: _FailingManager,
    )

    builder = SimpleNamespace(label="manual-fallback")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    placeholder_manager = cbi._DisabledSelfCodingManager(
        bot_registry=registry,
        data_bot=data_bot,
    )

    helper_bootstrap_calls: list[str] = []

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ManualFallbackHelper:
        name = helper_name

        def __init__(self) -> None:
            self.bot_name = self.name

    class ManualFallbackPipeline:
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
            self.helper = ManualFallbackHelper()
            self._bots = [self.helper]

    with monkeypatch.context() as patch_context:
        def _placeholder_bootstrap(name: str, *_args: object, **_kwargs: object) -> object:
            helper_bootstrap_calls.append(name)
            return placeholder_manager

        patch_context.setattr(cbi, "_bootstrap_manager", _placeholder_bootstrap)
        pipeline = ManualFallbackPipeline(
            context_builder=builder,
            bot_registry=registry,
            data_bot=data_bot,
            manager=placeholder_manager,
        )

    assert helper_bootstrap_calls == [helper_name]
    assert pipeline.helper.manager is placeholder_manager

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        ManualFallbackPipeline
    )

    promotion_calls: list[object] = []

    def _reuse_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        assert kwargs.get("pipeline_cls") is ManualFallbackPipeline

        def _promote(
            manager: object, *, extra_sentinels: Iterable[object] | None = None
        ) -> None:
            promotion_calls.append(manager)
            cbi._promote_pipeline_manager(
                pipeline,
                manager,
                placeholder_manager,
                extra_sentinels=extra_sentinels,
            )

        return pipeline, _promote

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_pipeline)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(owner_name, registry, data_bot)

    assert manager
    assert manager is not placeholder_manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert promotion_calls, "expected pipeline promoter to run for manual pipeline"
    assert pipeline.helper.manager is manager
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )


def test_real_pipeline_prebuilt_bootstrap_promotes_manager(
    real_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression ensuring real pipelines built pre-bootstrap promote helpers."""

    import menace.coding_bot_interface as cbi

    def _reuse_real_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        assert kwargs.get("pipeline_cls") is real_pipeline_env.pipeline_cls
        return real_pipeline_env.pipeline, real_pipeline_env.promoter

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_real_pipeline)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "RealPipelineOwner",
            real_pipeline_env.registry,
            real_pipeline_env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    pipeline = real_pipeline_env.pipeline
    assert pipeline.manager is manager
    assert pipeline.comms_bot.manager is manager
    assert pipeline.db_bot.manager is manager
    assert all(
        getattr(helper, "manager", manager) is manager
        for helper in real_pipeline_env.helpers
    )
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )


def test_managerless_pipeline_bootstrap_avoids_reentrant_warning(
    managerless_pipeline_env: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    import menace.coding_bot_interface as cbi

    caplog.clear()
    caplog.set_level(logging.WARNING, logger=cbi.logger.name)
    manager = cbi._bootstrap_manager(
        "ManagerlessOwner",
        managerless_pipeline_env.registry,
        managerless_pipeline_env.data_bot,
    )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )
    assert not any(
        "Re-entrant bootstrap" in record.message for record in caplog.records
    )

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.comms_bot.manager is manager
    assert pipeline.comms_bot.helper.manager is manager

    helper_names = set(managerless_pipeline_env.helper_names)
    assert {entry[0] for entry in managerless_pipeline_env.registry.registered} == helper_names
    assert {entry[0] for entry in managerless_pipeline_env.registry.promotions} == helper_names
    assert all(
        entry[1] is manager for entry in managerless_pipeline_env.registry.promotions
    )


def test_prepare_pipeline_managerless_constructor_prevents_reentrant_bootstrap_warning(
    managerless_pipeline_env: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    import menace.coding_bot_interface as cbi

    env = managerless_pipeline_env
    caplog.clear()
    caplog.set_level(logging.WARNING, logger=cbi.logger.name)

    pipeline, promote = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=env.pipeline_cls,
        context_builder=env.builder,
        bot_registry=env.registry,
        data_bot=env.data_bot,
    )

    assert pipeline is not None
    assert not any(
        "Re-entrant bootstrap" in record.message for record in caplog.records
    )

    placeholder_manager = getattr(pipeline, "manager", None)
    assert placeholder_manager is not None
    assert cbi._is_bootstrap_placeholder(placeholder_manager)

    assert pipeline.comms_bot.manager is placeholder_manager
    assert pipeline.comms_bot.helper.manager is placeholder_manager

    real_manager = SimpleNamespace(
        bot_registry=env.registry,
        data_bot=env.data_bot,
        pipeline=pipeline,
    )
    promote(real_manager)

    assert pipeline.manager is real_manager
    assert pipeline.comms_bot.manager is real_manager
    assert pipeline.comms_bot.helper.manager is real_manager


def test_preinstantiated_pipeline_bootstrap_promotes_without_reentrant_warning(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    builder = SimpleNamespace(label="preinstantiated-pipeline")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)
    monkeypatch.setattr(cbi, "ensure_self_coding_ready", lambda: (True, ()))

    recorded_pipelines: list[tuple[object | None, object | None]] = []

    def _recording_bootstrap_manager(
        name: str,
        bot_registry: object,
        data_bot: object,
        *,
        pipeline: object | None = None,
        pipeline_manager: object | None = None,
        pipeline_promoter: Callable[[object], None] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        context = cbi._current_bootstrap_context()
        context_pipeline = getattr(context, "pipeline", None) if context else None
        recorded_pipelines.append((pipeline, context_pipeline))
        resolved_pipeline = pipeline or context_pipeline
        return SimpleNamespace(
            name=name,
            bot_registry=bot_registry,
            data_bot=data_bot,
            pipeline=resolved_pipeline,
            evolution_orchestrator=None,
        )

    monkeypatch.setattr(cbi, "_bootstrap_manager", _recording_bootstrap_manager)

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class PreManagerHelper:
        name = "PreManagerHelper"

        def __init__(self) -> None:
            self.bot_name = self.name

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class ModelAutomationPipeline:
        name = "PreManagerPipeline"
        _bot_attribute_order = ("helper",)
        context_builder: object | None = None
        manager: object | None = None

        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object,
            data_bot: object,
            manager: object | None = None,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self.initial_manager = manager
            self.helper = PreManagerHelper()
            self._bots = [self.helper]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        ModelAutomationPipeline
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        pipeline = ModelAutomationPipeline(
            context_builder=builder,
            bot_registry=registry,
            data_bot=data_bot,
        )
    pipeline_manager = getattr(pipeline, "manager", None)
    assert pipeline_manager
    assert not isinstance(pipeline_manager, cbi._DisabledSelfCodingManager)

    helper_manager = getattr(pipeline.helper, "manager", None)
    assert helper_manager
    assert not isinstance(helper_manager, cbi._DisabledSelfCodingManager)
    assert getattr(helper_manager, "pipeline", None) is pipeline
    assert not any("Re-entrant bootstrap" in record.message for record in caplog.records)
    assert recorded_pipelines
    for pipeline_arg, context_pipeline in recorded_pipelines:
        assert context_pipeline is pipeline
        assert pipeline_arg is pipeline


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


def test_fallback_pipeline_without_manager_argument_installs_owner_sentinel(
    stub_bootstrap_env: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    builder = SimpleNamespace(label="ownerless-fallback")
    monkeypatch.setattr(cbi, "create_context_builder", lambda: builder)

    helper_name = "OwnerlessFallbackHelper"
    owner_name = "OwnerlessFallbackPipeline"

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class OwnerlessFallbackHelper:
        name = helper_name

        def __init__(self) -> None:
            self.bot_name = self.name

    class OwnerlessFallbackPipeline:
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
            self.initial_manager = getattr(self, "manager", None)
            self.helper = OwnerlessFallbackHelper()
            self._bots = [self.helper]

    stub_bootstrap_env["model_automation_pipeline"].ModelAutomationPipeline = (  # type: ignore[attr-defined]
        OwnerlessFallbackPipeline
    )

    caplog.set_level(logging.WARNING, logger=cbi.logger.name)
    manager = cbi._bootstrap_manager(owner_name, registry, data_bot)

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    pipeline = getattr(manager, "pipeline", None)
    assert pipeline is not None
    assert pipeline.helper.manager is manager

    registered = {name: payload for name, payload in registry.registered}
    assert helper_name in registered
    assert registered[helper_name]["manager"] is manager
    assert any(entry == (helper_name, manager) for entry in registry.promotions)


def test_prebuilt_communication_pipeline_promotes_eager_helpers(
    prebuilt_communication_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    def _reuse_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
        assert kwargs.get("pipeline_cls") is prebuilt_communication_pipeline_env.pipeline_cls
        return (
            prebuilt_communication_pipeline_env.pipeline,
            prebuilt_communication_pipeline_env.promoter,
        )

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_pipeline)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        manager = cbi._bootstrap_manager(
            "PrebuiltCommsOwner",
            prebuilt_communication_pipeline_env.registry,
            prebuilt_communication_pipeline_env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert prebuilt_communication_pipeline_env.pipeline.manager is manager
    assert getattr(manager, "pipeline", None) is prebuilt_communication_pipeline_env.pipeline

    helpers = [prebuilt_communication_pipeline_env.pipeline.comms_bot]
    helper = getattr(prebuilt_communication_pipeline_env.pipeline.comms_bot, "error_bot", None)
    if helper is not None:
        helpers.append(helper)

    for candidate in helpers:
        assert getattr(candidate, "manager", None) is manager
        assert not isinstance(candidate.manager, cbi._DisabledSelfCodingManager)

    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )


def test_prebuilt_communication_pipeline_logs_warning_without_sentinel(
    prebuilt_communication_pipeline_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import menace.coding_bot_interface as cbi

    original_activate = cbi._activate_bootstrap_sentinel

    def _activate_and_drop(manager: object | None) -> Callable[[], None]:
        restore = original_activate(manager)
        try:
            delattr(cbi._BOOTSTRAP_STATE, "sentinel_manager")
        except AttributeError:
            pass
        return restore

    monkeypatch.setattr(cbi, "_activate_bootstrap_sentinel", _activate_and_drop)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        pipeline, promoter = prebuilt_communication_pipeline_env.rebuild_pipeline()
        helper = getattr(pipeline.comms_bot, "error_bot", None)
        assert helper is not None

        def _reuse_pipeline(**kwargs: object) -> tuple[object, Callable[[object], None]]:
            assert kwargs.get("pipeline_cls") is prebuilt_communication_pipeline_env.pipeline_cls
            return pipeline, promoter

        monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _reuse_pipeline)

        manager = cbi._bootstrap_manager(
            "MissingSentinelOwner",
            prebuilt_communication_pipeline_env.registry,
            prebuilt_communication_pipeline_env.data_bot,
        )

    assert manager
    assert not isinstance(manager, cbi._DisabledSelfCodingManager)
    assert not any(
        "re-entrant initialisation depth" in record.message
        for record in caplog.records
    )

    assert helper.manager is manager

def test_bootstrap_manager_handles_falsy_owner_sentinel(
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
    assert all(
        entry[1].get("manager") is manager
        for entry in fallback_pipeline_env.registry.registered
    )
