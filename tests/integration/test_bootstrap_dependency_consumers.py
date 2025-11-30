from __future__ import annotations

import contextlib
import importlib
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

import coding_bot_interface as cbi


def _reset_cbi_state() -> None:
    broker = cbi._bootstrap_dependency_broker()
    broker.clear()
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()
    for attr in (
        "depth",
        "sentinel_manager",
        "pipeline",
        "owner_depths",
        "active_bootstrap_guard",
        "active_bootstrap_token",
    ):
        if hasattr(cbi._BOOTSTRAP_STATE, attr):
            delattr(cbi._BOOTSTRAP_STATE, attr)
    coordinator = getattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", None)
    if coordinator is not None and getattr(coordinator, "_active", None):
        coordinator._active = None


@contextlib.contextmanager
def _guarded_bootstrap(pipeline: object | None = None, manager: object | None = None):
    manager = manager or SimpleNamespace(bootstrap_placeholder=True)
    pipeline = pipeline or SimpleNamespace(
        manager=manager,
        initial_manager=None,
        bootstrap_placeholder=True,
    )
    manager.pipeline = getattr(manager, "pipeline", pipeline)
    cbi._mark_bootstrap_placeholder(pipeline)
    cbi._mark_bootstrap_placeholder(manager)
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]
    context = cbi._push_bootstrap_context(manager=manager, pipeline=pipeline)
    try:
        yield pipeline, manager
    finally:
        cbi._pop_bootstrap_context(context)
        _reset_cbi_state()


def test_research_aggregator_reuses_broker_pipeline(monkeypatch):
    _reset_cbi_state()
    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]
    broker = cbi._bootstrap_dependency_broker()
    broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", mock.Mock())

    module = importlib.import_module("menace_sandbox.research_aggregator_bot")
    monkeypatch.setattr(module, "registry", None)
    monkeypatch.setattr(module, "data_bot", None)
    monkeypatch.setattr(module, "_context_builder", None)
    monkeypatch.setattr(module, "engine", None)
    monkeypatch.setattr(module, "_PipelineCls", None)
    monkeypatch.setattr(module, "pipeline", None)
    monkeypatch.setattr(module, "evolution_orchestrator", None)
    monkeypatch.setattr(module, "manager", None)
    monkeypatch.setattr(module, "_runtime_state", None)
    monkeypatch.setattr(module, "_runtime_placeholder", None)
    monkeypatch.setattr(module, "_runtime_initializing", False)
    monkeypatch.setattr(module, "_self_coding_configured", False)

    class _Registry:
        pass

    class _DataBot:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    promotions: list[object] = []

    def _promote(manager: object | None) -> None:
        promotions.append(manager)

    monkeypatch.setattr(module, "BotRegistry", _Registry)
    monkeypatch.setattr(module, "DataBot", _DataBot)
    monkeypatch.setattr(module, "ContextBuilder", _ContextBuilder)
    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=_ContextBuilder()))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=SimpleNamespace(
        roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0
    )))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "self_coding_managed", lambda **_: (lambda cls: cls))

    manager_instance = SimpleNamespace(pipeline=pipeline_placeholder)
    monkeypatch.setattr(
        module,
        "internalize_coding_bot",
        mock.Mock(return_value=manager_instance),
    )
    monkeypatch.setattr(
        module,
        "ThresholdService",
        mock.Mock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=type("_Pipeline", (), {})))

    state = module._ensure_runtime_dependencies(
        pipeline_override=pipeline_placeholder,
        manager_override=sentinel_placeholder,
        promote_pipeline=_promote,
    )

    assert state.pipeline is pipeline_placeholder
    assert getattr(state.manager, "bootstrap_placeholder", False)
    assert promotions == [state.manager]
    module.internalize_coding_bot.assert_not_called()
    cbi.prepare_pipeline_for_bootstrap.assert_not_called()


def test_service_supervisor_reuses_dependency_broker_pipeline(monkeypatch, tmp_path, caplog):
    _reset_cbi_state()
    caplog.set_level("INFO")
    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)
    broker = cbi._bootstrap_dependency_broker()
    broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]

    code_db_module = ModuleType("code_database")
    code_db_module.CodeDB = type("_CodeDB", (), {})
    code_db_module.PatchRecord = type("_PatchRecord", (), {})
    monkeypatch.setitem(sys.modules, "code_database", code_db_module)

    master_module = ModuleType("menace_sandbox.menace_master")
    master_module._init_unused_bots = lambda: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.menace_master", master_module)

    sys.modules.pop("menace_sandbox.service_supervisor", None)
    sys.modules.pop("service_supervisor", None)

    supervisor_module = ModuleType("menace_sandbox.service_supervisor")
    supervisor_module.prepare_pipeline_for_bootstrap = mock.Mock()

    class _ServiceSupervisor:
        def __init__(
            self,
            *,
            context_builder: object,
            dependency_broker: object | None = None,
            pipeline: object | None = None,
            pipeline_promoter: Callable[[object], None] | None = None,
            **_: object,
        ) -> None:
            self.context_builder = context_builder
            self._bootstrap_dependency_broker = (
                dependency_broker if dependency_broker is not None else cbi._bootstrap_dependency_broker()
            )
            broker_pipeline, _sentinel = self._bootstrap_dependency_broker.resolve()
            self.pipeline = pipeline or broker_pipeline
            if pipeline_promoter is None:
                self._pipeline_promoter = lambda manager: setattr(self, "promoted", manager)
            else:
                self._pipeline_promoter = pipeline_promoter
            if self.pipeline is None:
                raise RuntimeError("ServiceSupervisor requires a pipeline")

        def _resolve_bootstrap_handles(self) -> tuple[object | None, Callable[[object], None] | None]:
            return self.pipeline, self._pipeline_promoter

    supervisor_module.ServiceSupervisor = _ServiceSupervisor
    monkeypatch.setitem(sys.modules, "menace_sandbox.service_supervisor", supervisor_module)

    import menace_sandbox.service_supervisor as ss

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    supervisor = ss.ServiceSupervisor(
        context_builder=_ContextBuilder(),
        log_path=str(tmp_path / "supervisor.log"),
        restart_log=str(tmp_path / "restart.log"),
        dependency_broker=broker,
        pipeline=pipeline_placeholder,
    )

    assert supervisor._pipeline_promoter is not None
    assert supervisor._resolve_bootstrap_handles()[0] is pipeline_placeholder
    ss.prepare_pipeline_for_bootstrap.assert_not_called()
    assert not any(
        "prepare_pipeline_for_bootstrap" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.integration
def test_error_bot_reuses_guarded_bootstrap_pipeline(monkeypatch):
    _reset_cbi_state()
    sys.modules.pop("menace_sandbox.error_bot", None)
    sys.modules.pop("error_bot", None)

    with _guarded_bootstrap() as (pipeline, manager):
        promotions: list[object] = []
        pipeline._pipeline_promoter = lambda mgr: promotions.append(mgr)

        module = importlib.import_module("menace_sandbox.error_bot")

        for attr in (
            "_context_builder",
            "_engine",
            "_pipeline",
            "_evolution_orchestrator",
            "_thresholds",
            "_manager_instance",
            "_pipeline_promoter",
        ):
            setattr(module, attr, None)

        module.create_context_builder = lambda: SimpleNamespace(label="guarded")
        module.BotRegistry = lambda *_, **__: SimpleNamespace()
        module.DataBot = lambda *_, **__: SimpleNamespace()
        module.SelfCodingEngine = lambda *_, **__: SimpleNamespace()
        module.CodeDB = lambda *_, **__: SimpleNamespace()
        module.GPTMemoryManager = lambda *_, **__: SimpleNamespace()
        module.get_orchestrator = lambda *_, **__: SimpleNamespace()
        module.get_thresholds = lambda *_: SimpleNamespace(
            roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0
        )
        module.persist_sc_thresholds = mock.Mock()
        module.ThresholdService = lambda: SimpleNamespace()
        module.internalize_coding_bot = mock.Mock(
            side_effect=AssertionError("internalize_coding_bot should not run")
        )
        module.prepare_pipeline_for_bootstrap = mock.Mock(
            side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")
        )

        resolved_manager = module._ensure_self_coding_manager()

        assert resolved_manager is manager
        assert module.prepare_pipeline_for_bootstrap.call_count == 0
        assert module.internalize_coding_bot.call_count == 0
        assert promotions == []


@pytest.mark.integration
def test_research_aggregator_respects_guarded_pipeline(monkeypatch):
    _reset_cbi_state()
    sys.modules.pop("menace_sandbox.research_aggregator_bot", None)

    with _guarded_bootstrap() as (pipeline, manager):
        promotions: list[object] = []

        module = importlib.import_module("menace_sandbox.research_aggregator_bot")
        module.pipeline = None
        module.manager = None
        module.registry = None
        module.data_bot = None
        module._context_builder = None
        module.engine = None
        module._PipelineCls = None
        module._runtime_state = None
        module._runtime_placeholder = None
        module._runtime_initializing = False
        module._self_coding_configured = False

        module.create_context_builder = mock.Mock(return_value=SimpleNamespace())
        module.BotRegistry = lambda *_, **__: SimpleNamespace()
        module.DataBot = lambda *_, **__: SimpleNamespace()
        module.SelfCodingEngine = lambda *_, **__: SimpleNamespace()
        module.CodeDB = lambda *_, **__: SimpleNamespace()
        module.GPTMemoryManager = lambda *_, **__: SimpleNamespace()
        module._resolve_pipeline_cls = mock.Mock(return_value=type("_Pipeline", (), {}))
        module.get_orchestrator = mock.Mock(return_value=SimpleNamespace())
        module.get_thresholds = mock.Mock(
            return_value=SimpleNamespace(
                roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0
            )
        )
        module.persist_sc_thresholds = mock.Mock()
        module.internalize_coding_bot = mock.Mock(return_value=manager)
        module.ThresholdService = mock.Mock(return_value=SimpleNamespace())
        module.self_coding_managed = lambda **_: (lambda cls: cls)
        module.prepare_pipeline_for_bootstrap = mock.Mock(
            side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")
        )

        state = module._ensure_runtime_dependencies(promote_pipeline=promotions.append)

        assert state.pipeline is pipeline
        assert state.manager is manager
        assert module.prepare_pipeline_for_bootstrap.call_count == 0
        assert promotions == [manager]


@pytest.mark.integration
def test_research_aggregator_waits_for_active_prepare(monkeypatch):
    _reset_cbi_state()
    sys.modules.pop("menace_sandbox.research_aggregator_bot", None)

    broker = cbi._bootstrap_dependency_broker()
    broker.clear()

    guard = object()
    cbi._BOOTSTRAP_STATE.active_bootstrap_guard = guard  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {guard: 1}  # type: ignore[attr-defined]
    cbi._ensure_owner_promise(guard)

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)

    def _advertise_later() -> None:
        time.sleep(0.05)
        broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)

    advertiser = threading.Thread(target=_advertise_later, daemon=True)
    advertiser.start()

    module = importlib.import_module("menace_sandbox.research_aggregator_bot")
    module.pipeline = None
    module.manager = None
    module.registry = None
    module.data_bot = None
    module._context_builder = None
    module.engine = None
    module._PipelineCls = None
    module._runtime_state = None
    module._runtime_placeholder = None
    module._runtime_initializing = False
    module._self_coding_configured = False

    module.create_context_builder = mock.Mock(return_value=SimpleNamespace())
    module.BotRegistry = lambda *_, **__: SimpleNamespace()
    module.DataBot = lambda *_, **__: SimpleNamespace()
    module.SelfCodingEngine = lambda *_, **__: SimpleNamespace()
    module.CodeDB = lambda *_, **__: SimpleNamespace()
    module.GPTMemoryManager = lambda *_, **__: SimpleNamespace()
    module._resolve_pipeline_cls = mock.Mock(return_value=type("_Pipeline", (), {}))
    module.get_orchestrator = mock.Mock(return_value=SimpleNamespace())
    module.get_thresholds = mock.Mock(
        return_value=SimpleNamespace(
            roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0
        )
    )
    module.persist_sc_thresholds = mock.Mock()
    module.internalize_coding_bot = mock.Mock(
        side_effect=AssertionError("internalize_coding_bot should not run")
    )
    module.ThresholdService = mock.Mock(return_value=SimpleNamespace())
    module.self_coding_managed = lambda **_: (lambda cls: cls)
    module.prepare_pipeline_for_bootstrap = mock.Mock(
        side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")
    )

    promotions: list[object] = []

    state = module._ensure_runtime_dependencies(
        promote_pipeline=promotions.append,
        manager_override=sentinel_placeholder,
    )

    advertiser.join(timeout=2)

    assert state.pipeline is pipeline_placeholder
    assert state.manager is sentinel_placeholder
    assert promotions == [sentinel_placeholder]
    module.prepare_pipeline_for_bootstrap.assert_not_called()
    module.internalize_coding_bot.assert_not_called()


@pytest.mark.integration
def test_prediction_manager_guard_opt_out_reuses_promise(monkeypatch):
    _reset_cbi_state()
    broker = cbi._bootstrap_dependency_broker()
    broker.clear()

    promise = cbi._BootstrapPipelinePromise()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR._active = promise

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)

    def _resolve_promise() -> None:
        time.sleep(0.05)
        broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)
        promise.resolve((pipeline_placeholder, lambda *_: None))

    resolver = threading.Thread(target=_resolve_promise, daemon=True)
    resolver.start()

    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_OPTOUT", "1")

    class _Unreachable:
        def __init__(self, **_: object) -> None:  # pragma: no cover - must not run
            raise AssertionError("pipeline constructor should be bypassed")

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Unreachable,
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        bootstrap_guard=True,
        bootstrap_wait_timeout=0.5,
    )

    resolver.join(timeout=2)

    assert pipeline is pipeline_placeholder
    assert getattr(pipeline, "manager", None) is sentinel_placeholder
    assert callable(promote)
    assert broker.resolve()[0] is pipeline_placeholder


@pytest.mark.integration
def test_watchdog_health_checks_reuse_advertised_pipeline(monkeypatch):
    _reset_cbi_state()
    broker = cbi._bootstrap_dependency_broker()
    broker.clear()

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)

    def _advertise():
        time.sleep(0.05)
        broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)

    threading.Thread(target=_advertise, daemon=True).start()

    sys.modules.pop("menace_sandbox.watchdog", None)
    module = importlib.import_module("menace_sandbox.watchdog")

    monkeypatch.setattr(module, "prepare_pipeline_for_bootstrap", mock.Mock())
    monkeypatch.setattr(module, "_bootstrap_dependency_broker", lambda: broker)

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    checker = module.Watchdog(context_builder=_ContextBuilder(), router=None, enable_health_checks=True)
    pipeline, manager = checker._bootstrap_dependency_broker.resolve()

    assert pipeline is pipeline_placeholder
    assert manager is sentinel_placeholder
    checker.prepare_pipeline_for_bootstrap.assert_not_called()
