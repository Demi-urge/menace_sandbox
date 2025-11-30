import itertools
import sys
import contextvars
import logging
import threading
import time
import types
from types import SimpleNamespace
from typing import Any

import pytest


class DummyManager:
    def __init__(self, *_, **kwargs):
        self.bot_registry = kwargs.get("bot_registry")
        self.data_bot = kwargs.get("data_bot")
        self.quick_fix = kwargs.get("quick_fix") or object()


stub = types.ModuleType("menace.self_coding_manager")
stub.SelfCodingManager = DummyManager
sys.modules.setdefault("menace.self_coding_manager", stub)

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
sce_stub.SelfCodingEngine = object
sys.modules.setdefault("menace.self_coding_engine", sce_stub)

# lightweight stubs for orchestrator dependencies used during auto-instantiation
evo_stub = types.ModuleType("menace.evolution_orchestrator")

class _StubOrchestrator:
    def __init__(self, data_bot, capital_bot, improvement_engine, evolution_manager, selfcoding_manager=None):
        self.data_bot = data_bot
        self.registered: list[str] = []

    def register_bot(self, name: str) -> None:
        self.registered.append(name)


evo_stub.EvolutionOrchestrator = _StubOrchestrator
sys.modules.setdefault("menace.evolution_orchestrator", evo_stub)

cap_stub = types.ModuleType("menace.capital_management_bot")

class CapitalManagementBot:
    def __init__(self, *a, **k):
        pass


cap_stub.CapitalManagementBot = CapitalManagementBot
sys.modules.setdefault("menace.capital_management_bot", cap_stub)

sie_stub = types.ModuleType("menace.self_improvement.engine")

class SelfImprovementEngine:
    def __init__(self, *a, **k):
        pass


sie_stub.SelfImprovementEngine = SelfImprovementEngine
sys.modules.setdefault("menace.self_improvement.engine", sie_stub)

sem_stub = types.ModuleType("menace.system_evolution_manager")

class SystemEvolutionManager:
    def __init__(self, bots, *a, **k):
        pass


sem_stub.SystemEvolutionManager = SystemEvolutionManager
sys.modules.setdefault("menace.system_evolution_manager", sem_stub)

import menace.coding_bot_interface as cbi
from menace.roi_thresholds import ROIThresholds

cbi.update_thresholds = lambda *a, **k: None
self_coding_managed = cbi.self_coding_managed


class DummyRegistry:
    def __init__(self):
        self.registered = []
        self.updated = []
        self.register_calls = []

    def register_bot(self, name, **kwargs):
        self.registered.append(name)
        self.register_calls.append((name, kwargs))

    def update_bot(self, name, module_path):
        self.updated.append((name, module_path))


class ProvenanceRegistry(DummyRegistry):
    def __init__(self):
        super().__init__()
        self.provenance_updates = []
        self.attempts = 0

    def update_bot(self, name, module_path, *, patch_id=None, commit=None):
        self.attempts += 1
        if patch_id is None or commit is None:
            raise RuntimeError("patch provenance required")
        self.provenance_updates.append((name, module_path, patch_id, commit))


class DummyDB:
    def __init__(self):
        self.logged = []

    def log_eval(self, name, metric, value):
        self.logged.append((name, metric, value))


class DummyDataBot:
    def __init__(self):
        self.db = DummyDB()
        self._thresholds = {}

    def roi(self, name):
        return 0.0

    def reload_thresholds(self, bot=None):
        rt = ROIThresholds(roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0)
        self._thresholds[bot or ""] = rt
        return rt


class DummyOrchestrator:
    def __init__(self):
        self.registered = []

    def register_bot(self, name):
        self.registered.append(name)


def test_missing_bot_registry():
    with pytest.raises(TypeError):
        @self_coding_managed(data_bot=DummyDataBot())  # type: ignore[call-arg]
        class Bot:
            def __init__(self):
                pass


def test_missing_data_bot():
    with pytest.raises(TypeError):
        @self_coding_managed(bot_registry=DummyRegistry())  # type: ignore[call-arg]
        class Bot:
            def __init__(self):
                pass


def test_auto_instantiates_orchestrator():
    class _Registry(DummyRegistry):
        def register_bot(self, name, *_args, **kwargs):
            return super().register_bot(name, **kwargs)

    registry = _Registry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "auto"

        def __init__(self):
            pass

    bot = Bot()
    assert registry.registered == ["auto"]
    assert bot.evolution_orchestrator.registered == ["auto"]


def test_auto_instantiates_with_stubbed_builder(monkeypatch):
    def _stub_resolve(_logger=None):
        class _StubBuilder:
            def refresh_db_weights(self):
                return None

        return _StubBuilder(), True

    monkeypatch.setattr(evo_stub, "resolve_context_builder", _stub_resolve)

    class _Registry(DummyRegistry):
        def register_bot(self, name, *_args, **kwargs):
            return super().register_bot(name, **kwargs)

    registry = _Registry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "stub"

        def __init__(self):
            pass

    bot = Bot()
    assert registry.registered == ["stub"]
    assert getattr(bot.evolution_orchestrator, "context_builder_degraded", False)


def test_orchestrator_autoinstantiation_failure(monkeypatch):
    class Broken(_StubOrchestrator):
        def __init__(self, *a, **k):  # pragma: no cover - simulate failure
            raise RuntimeError("boom")

    monkeypatch.setattr(evo_stub, "EvolutionOrchestrator", Broken)

    @self_coding_managed(bot_registry=DummyRegistry(), data_bot=DummyDataBot())
    class Bot:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="EvolutionOrchestrator is required"):
        Bot()


def test_successful_initialisation_registers():
    class _Registry(DummyRegistry):
        def register_bot(self, name, *_args, **kwargs):
            return super().register_bot(name, **kwargs)

    registry = _Registry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "sample"

        def __init__(self):
            pass
    Bot(evolution_orchestrator=orchestrator)

    assert registry.registered == ["sample"]
    assert orchestrator.registered == ["sample"]


def test_module_path_falls_back_to_module_name(monkeypatch):
    registry = DummyRegistry()
    data_bot = DummyDataBot()

    def _raise_getfile(_cls):
        raise OSError("no source")

    monkeypatch.setattr(cbi.inspect, "getfile", _raise_getfile)

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "fallback"

        def __init__(self):
            pass

    Bot()

    assert registry.updated, "bot update should be attempted"
    assert registry.updated[-1][1] == Bot.__module__


def test_thresholds_loaded_on_init():
    registry = DummyRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "loader"

        def __init__(self):
            pass

    Bot()
    assert "loader" in data_bot._thresholds


def test_unsigned_provenance_generated_when_metadata_missing():
    registry = ProvenanceRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "provless"

        def __init__(self):
            pass

    Bot()
    assert registry.registered == ["provless"]
    assert registry.attempts == 1
    assert len(registry.provenance_updates) == 1
    name, module_path, patch_id, commit = registry.provenance_updates[0]
    assert name == "provless"
    assert patch_id < 0
    assert commit.startswith("unsigned:")
    assert registry.register_calls[0][1].get("is_coding_bot") is True


def test_signed_policy_disables_self_coding_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MENACE_REQUIRE_SIGNED_PROVENANCE", "1")
    registry = ProvenanceRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "strictprov"

        def __init__(self):
            pass

    Bot()
    assert registry.registered == ["strictprov"]
    assert registry.attempts == 0
    assert registry.provenance_updates == []
    assert registry.register_calls[0][1].get("is_coding_bot") is False


def test_uses_manager_provenance_for_update():
    registry = ProvenanceRegistry()
    data_bot = DummyDataBot()
    manager = DummyManager(bot_registry=registry, data_bot=data_bot)
    manager._last_patch_id = 123
    manager._last_commit_hash = "deadbeef"

    @self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
    class Bot:
        name = "provbot"

        def __init__(self):
            pass

    Bot()
    assert [update[0] for update in registry.provenance_updates] == ["provbot"]
    assert registry.provenance_updates[0][2:] == (123, "deadbeef")


def test_falsey_manager_preserved_and_disabled_only_on_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class FalseyManager:
        def __init__(self) -> None:
            self.bot_registry = registry
            self.data_bot = data_bot

        def __bool__(self) -> bool:
            return False

    sentinel_manager = FalseyManager()

    def _fail_bootstrap(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("bootstrap should not execute when manager supplied")

    monkeypatch.setattr(cbi, "_bootstrap_manager", _fail_bootstrap)
    monkeypatch.setattr(cbi, "_self_coding_runtime_available", lambda: True)

    @self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
        manager=sentinel_manager,
    )
    class BotWithSentinel:
        name = "sentinel"

        def __init__(self) -> None:
            pass

    bot = BotWithSentinel()
    assert bot.manager is sentinel_manager
    assert not isinstance(bot.manager, cbi._DisabledSelfCodingManager)

    monkeypatch.setattr(cbi, "_self_coding_runtime_available", lambda: False)

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class BotWithFallback:
        name = "fallback"

        def __init__(self) -> None:
            pass

    fallback_bot = BotWithFallback()
    assert isinstance(fallback_bot.manager, cbi._DisabledSelfCodingManager)


def test_bootstrap_helpers_promotes_manager_with_reentrant_depth(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=cbi.logger.name)

    class _Registry(DummyRegistry):
        def register_bot(self, name, *_args, **kwargs):
            return super().register_bot(name, **kwargs)

    registry = _Registry()
    data_bot = DummyDataBot()
    sentinel = cbi._create_bootstrap_manager_sentinel(
        bot_registry=registry,
        data_bot=data_bot,
    )

    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1, raising=False)
    monkeypatch.setattr(
        cbi._BOOTSTRAP_STATE, "sentinel_manager", sentinel, raising=False
    )
    monkeypatch.setattr(cbi, "_self_coding_runtime_available", lambda: True)
    monkeypatch.setattr(cbi, "ensure_self_coding_ready", lambda: (True, ()))
    monkeypatch.setattr(cbi, "_registry_hot_swap_active", lambda _registry: False)

    monkeypatch.setattr(
        cbi,
        "_resolve_provenance_decision",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="active",
            reason=None,
            patch_id=None,
            commit=None,
            available=True,
            source="tests",
        ),
    )

    bootstrap_calls: dict[str, object] = {}

    def _tracking_bootstrap_manager(name, bot_registry, data_bot, **_kwargs):
        bootstrap_calls["depth_before"] = getattr(cbi._BOOTSTRAP_STATE, "depth", 0)
        manager = SimpleNamespace(
            bot_registry=bot_registry,
            data_bot=data_bot,
            pipeline=SimpleNamespace(manager=None, _bots=[]),
        )
        bootstrap_calls["manager"] = manager
        bootstrap_calls["depth_after"] = getattr(cbi._BOOTSTRAP_STATE, "depth", 0)
        return manager

    monkeypatch.setattr(cbi, "_bootstrap_manager", _tracking_bootstrap_manager)

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class NestedBot:
        name = "NestedBot"

        def __init__(self):
            pass

    bot = NestedBot()

    assert bootstrap_calls["manager"] is bot.manager
    assert bootstrap_calls["depth_before"] == 1
    assert "re-entrant initialisation depth" not in caplog.text


def test_prepare_pipeline_disabled_manager_skips_bootstrap(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=cbi.logger.name)

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    def _unexpected_bootstrap(*_args, **_kwargs):  # pragma: no cover - failure guard
        raise AssertionError("_bootstrap_manager should not run during pipeline bootstrap")

    monkeypatch.setattr(cbi, "_bootstrap_manager", _unexpected_bootstrap)
    monkeypatch.setattr(cbi, "_self_coding_runtime_available", lambda: True)
    monkeypatch.setattr(cbi, "ensure_self_coding_ready", lambda: (True, ()))
    monkeypatch.setattr(cbi, "_registry_hot_swap_active", lambda _registry: False)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1, raising=False)

    runtime_manager = cbi._DisabledSelfCodingManager(
        bot_registry=registry,
        data_bot=data_bot,
    )

    nested_state: dict[str, Any] = {}

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class NestedBot:
        name = "NestedPipelineBot"

        def __init__(self) -> None:
            nested_state["manager"] = getattr(self, "manager", None)

    class _PipelineProbe:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            **_kwargs,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            self._bots: list[Any] = []
            NestedBot()

    pipeline, _promote = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=_PipelineProbe,
        context_builder=SimpleNamespace(),
        bot_registry=registry,
        data_bot=data_bot,
        bootstrap_runtime_manager=runtime_manager,
    )

    assert nested_state["manager"] is runtime_manager
    assert isinstance(pipeline, _PipelineProbe)
    assert "re-entrant" not in caplog.text


def test_prepare_pipeline_timeout_emits_watchdog(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=cbi.logger.name)
    cbi._PREPARE_PIPELINE_WATCHDOG["stages"].clear()
    cbi._PREPARE_PIPELINE_WATCHDOG["timeouts"] = 0
    cbi._initialize_prepare_readiness(reset=True)

    setattr(cbi._BOOTSTRAP_STATE, "vector_heavy", False)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "vector_heavy", True)

    perf_calls = itertools.chain([0.0, 500.0, 500.0], itertools.repeat(500.0))
    monkeypatch.setattr(cbi.time, "perf_counter", lambda: next(perf_calls))

    class _Pipeline:
        def __init__(self, *_, **__):
            pass

    cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Pipeline,
        context_builder=SimpleNamespace(),
        bot_registry=DummyRegistry(),
        data_bot=DummyDataBot(),
        timeout=0.0,
    )

    degraded_events = [
        entry
        for entry in cbi._PREPARE_PIPELINE_WATCHDOG["stages"]
        if entry.get("timeout") and entry.get("degraded_but_online")
    ]
    assert degraded_events, "expected degraded timeout to be recorded"
    last_stage = degraded_events[-1]
    assert last_stage.get("watchdog_pending_gates"), "pending gates should be surfaced"
    assert last_stage.get("watchdog_readiness_ratio") is not None

    if hasattr(cbi._BOOTSTRAP_STATE, "vector_heavy"):
        delattr(cbi._BOOTSTRAP_STATE, "vector_heavy")


def test_prepare_pipeline_single_flight_reuses_active(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda: {})

    start_event = threading.Event()
    release_event = threading.Event()
    results: list[tuple[Any, Any]] = []
    secondary_results: list[tuple[Any, Any]] = []
    errors: list[BaseException] = []

    class _Pipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            **_kwargs,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            start_event.set()
            release_event.wait(timeout=1)

    def _prepare(target_list: list[tuple[Any, Any]]) -> None:
        try:
            target_list.append(
                cbi.prepare_pipeline_for_bootstrap(
                    pipeline_cls=_Pipeline,
                    context_builder=SimpleNamespace(),
                    bot_registry=DummyRegistry(),
                    data_bot=DummyDataBot(),
                )
            )
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    first_thread = threading.Thread(target=_prepare, args=(results,))
    second_thread = threading.Thread(target=_prepare, args=(secondary_results,))

    first_thread.start()
    try:
        assert start_event.wait(1), "first pipeline did not start"
        second_thread.start()
        release_event.set()
        first_thread.join(timeout=1)
        second_thread.join(timeout=1)
    finally:
        release_event.set()
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()

    assert not errors
    assert results and secondary_results
    pipeline_a, promote_a = results[0]
    pipeline_b, promote_b = secondary_results[0]
    assert pipeline_a is pipeline_b
    assert promote_a is promote_b

    prepare_logs = [
        record
        for record in caplog.records
        if "prepare_pipeline_for_bootstrap" in record.getMessage()
    ]
    assert len(prepare_logs) == 1


def test_prepare_pipeline_reentrancy_honours_active_promise(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    start_event = threading.Event()
    release_event = threading.Event()
    init_count = 0

    class _Pipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            **_kwargs,
        ) -> None:
            nonlocal init_count
            init_count += 1
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            start_event.set()
            release_event.wait(timeout=1)

    def _prepare(target_list: list[tuple[Any, Any]], guard: bool) -> None:
        target_list.append(
            cbi._prepare_pipeline_for_bootstrap_impl(
                pipeline_cls=_Pipeline,
                context_builder=SimpleNamespace(),
                bot_registry=DummyRegistry(),
                data_bot=DummyDataBot(),
                bootstrap_guard=guard,
            )
        )

    results: list[tuple[Any, Any]] = []
    secondary_results: list[tuple[Any, Any]] = []
    first_thread = threading.Thread(target=_prepare, args=(results, True))
    second_thread = threading.Thread(target=_prepare, args=(secondary_results, False))

    first_thread.start()
    try:
        assert start_event.wait(1)
        second_thread.start()
        release_event.set()
        first_thread.join(timeout=1)
        second_thread.join(timeout=1)
    finally:
        release_event.set()
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    assert init_count == 1
    pipeline_a, promote_a = results[0]
    pipeline_b, promote_b = secondary_results[0]
    assert pipeline_a is pipeline_b
    assert promote_a is promote_b


def test_prepare_pipeline_guardless_reuses_dependency_broker(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    start_event = threading.Event()
    release_event = threading.Event()
    init_count = 0

    class _Pipeline:
        def __init__(self, *_args, **_kwargs) -> None:
            nonlocal init_count
            init_count += 1
            start_event.set()
            release_event.wait(timeout=1)

    results: list[tuple[Any, Any]] = []
    secondary_results: list[tuple[Any, Any]] = []

    def _prepare(target_list: list[tuple[Any, Any]]) -> None:
        target_list.append(
            cbi._prepare_pipeline_for_bootstrap_impl(
                pipeline_cls=_Pipeline,
                context_builder=SimpleNamespace(),
                bot_registry=DummyRegistry(),
                data_bot=DummyDataBot(),
                bootstrap_guard=False,
            )
        )

    first_thread = threading.Thread(target=_prepare, args=(results,))
    second_thread = threading.Thread(target=_prepare, args=(secondary_results,))

    first_thread.start()
    try:
        assert start_event.wait(1)
        second_thread.start()
        release_event.set()
        first_thread.join(timeout=1)
        second_thread.join(timeout=1)
    finally:
        release_event.set()
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    assert init_count == 1
    pipeline_a, promote_a = results[0]
    pipeline_b, promote_b = secondary_results[0]
    assert pipeline_a is pipeline_b
    assert promote_a is promote_b


def test_prepare_pipeline_recursion_refused_when_broker_empty(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = cbi._build_bootstrap_placeholder_pipeline(
        sentinel_placeholder
    )
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.pipeline = pipeline_placeholder  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.sentinel_manager = sentinel_placeholder  # type: ignore[attr-defined]

    def _stub_inner(**_kwargs):  # pragma: no cover - defensive
        raise AssertionError("bootstrap inner should not be invoked")

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl_inner", _stub_inner)

    class _Pipeline:
        def __init__(self, *, manager=None, **_kwargs):
            self.manager = manager

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Pipeline,
        context_builder=SimpleNamespace(),
        bot_registry=DummyRegistry(),
        data_bot=DummyDataBot(),
        bootstrap_guard=False,
    )

    assert pipeline is pipeline_placeholder
    promote(SimpleNamespace())
    assert any("recursion_refused" in record.getMessage() for record in caplog.records)

    dependency_broker.clear()
    for attr in ("depth", "pipeline", "sentinel_manager"):
        if hasattr(cbi._BOOTSTRAP_STATE, attr):
            delattr(cbi._BOOTSTRAP_STATE, attr)


def test_prepare_pipeline_recursion_raises_without_broker_sentinel(monkeypatch):
    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = cbi._build_bootstrap_placeholder_pipeline(
        sentinel_placeholder
    )
    cbi._BOOTSTRAP_STATE.depth = 4  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.pipeline = pipeline_placeholder  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.sentinel_manager = sentinel_placeholder  # type: ignore[attr-defined]

    class _Pipeline:
        def __init__(self, *, manager=None, **_kwargs):
            self.manager = manager

    with pytest.raises(RecursionError):
        cbi._prepare_pipeline_for_bootstrap_impl(
            pipeline_cls=_Pipeline,
            context_builder=SimpleNamespace(),
            bot_registry=DummyRegistry(),
            data_bot=DummyDataBot(),
            bootstrap_guard=False,
        )

    dependency_broker.clear()
    for attr in ("depth", "pipeline", "sentinel_manager"):
        if hasattr(cbi._BOOTSTRAP_STATE, attr):
            delattr(cbi._BOOTSTRAP_STATE, attr)


def test_prepare_pipeline_preflight_reuses_broker_placeholder(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)
    dependency_broker.advertise(
        pipeline=pipeline_placeholder, sentinel=sentinel_placeholder, owner=True
    )

    def _fail_impl(**_kwargs):  # pragma: no cover - guard must short circuit
        raise AssertionError("prepare_pipeline_for_bootstrap_impl should be bypassed")

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl", _fail_impl)

    pipeline, promote = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=SimpleNamespace,
        context_builder=SimpleNamespace(),
        bot_registry=DummyRegistry(),
        data_bot=DummyDataBot(),
    )

    assert getattr(pipeline, "bootstrap_placeholder", False) is True
    assert getattr(pipeline, "manager", None) is sentinel_placeholder
    assert callable(promote)

    real_manager = SimpleNamespace()
    promote(real_manager)

    assert getattr(pipeline, "manager", None) is real_manager
    assert getattr(pipeline, "initial_manager", None) is real_manager
    assert any(
        record.message == "prepare_pipeline.bootstrap.preflight_broker_reuse"
        for record in caplog.records
    )

    dependency_broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()


def test_prepare_pipeline_preflight_waits_on_broker_promise(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)
    dependency_broker.advertise(
        pipeline=pipeline_placeholder, sentinel=sentinel_placeholder, owner=True
    )

    def _fail_impl(**_kwargs):  # pragma: no cover - guard must short circuit
        raise AssertionError("prepare_pipeline_for_bootstrap_impl should be bypassed")

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl", _fail_impl)

    owner, promise = cbi._GLOBAL_BOOTSTRAP_COORDINATOR.claim()
    assert promise is not None
    assert owner is True

    settled_pipeline = SimpleNamespace()
    settled_promote = lambda *_a, **_k: None

    def _settle() -> None:
        time.sleep(0.05)
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.settle(
            promise, result=(settled_pipeline, settled_promote)
        )

    threading.Thread(target=_settle, daemon=True).start()

    pipeline, promote = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=SimpleNamespace,
        context_builder=SimpleNamespace(),
        bot_registry=DummyRegistry(),
        data_bot=DummyDataBot(),
    )

    assert pipeline is settled_pipeline
    assert promote is settled_promote
    assert any(
        record.message == "prepare_pipeline.bootstrap.preflight_broker_wait"
        for record in caplog.records
    )

    dependency_broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

def test_parallel_helpers_share_dependency_broker_placeholder(monkeypatch):
    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()

    start_event = threading.Event()
    release_event = threading.Event()
    prepare_calls: list[dict[str, Any]] = []
    promotions: list[Any] = []
    broker_snapshots: list[tuple[Any | None, Any | None]] = []

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)

    class DummyModelAutomationPipeline:
        def __init__(self) -> None:
            self.context_builder = SimpleNamespace()
            self.manager = sentinel_placeholder
            self.initial_manager = sentinel_placeholder
            self._bot_attribute_order = []
            self.bootstrap_placeholder = True

    pipeline_placeholder = DummyModelAutomationPipeline()

    def _stub_inner(**kwargs):
        prepare_calls.append(kwargs)
        dependency_broker.advertise(
            pipeline=pipeline_placeholder, sentinel=sentinel_placeholder, owner=True
        )
        broker_snapshots.append(dependency_broker.resolve())
        start_event.set()
        release_event.wait(timeout=5)
        return pipeline_placeholder, lambda manager: promotions.append(manager)

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl_inner", _stub_inner)

    class _Pipeline:
        vector_bootstrap_heavy = True

        def __init__(self, *, manager=None, **_kwargs) -> None:
            self.manager = manager

    def _bootstrap_helper(results: list[tuple[Any, Any]]) -> None:
        results.append(
            cbi.prepare_pipeline_for_bootstrap(
                pipeline_cls=_Pipeline,
                context_builder=SimpleNamespace(),
                bot_registry=DummyRegistry(),
                data_bot=DummyDataBot(),
            )
        )

    owner_results: list[tuple[Any, Any]] = []
    aggregator_results: list[tuple[Any, Any]] = []
    prediction_results: list[tuple[Any, Any]] = []
    memory_results: list[tuple[Any, Any]] = []

    owner_thread = threading.Thread(
        target=_bootstrap_helper,
        args=(owner_results,),
    )
    owner_thread.start()
    assert start_event.wait(timeout=5)

    helper_threads = [
        threading.Thread(target=_bootstrap_helper, args=(aggregator_results,)),
        threading.Thread(target=_bootstrap_helper, args=(prediction_results,)),
        threading.Thread(target=_bootstrap_helper, args=(memory_results,)),
    ]
    for thread in helper_threads:
        thread.start()

    release_event.set()
    owner_thread.join(timeout=5)
    for thread in helper_threads:
        thread.join(timeout=5)

    assert len(prepare_calls) == 1

    all_results = owner_results + aggregator_results + prediction_results + memory_results
    assert all_results
    assert all(pipeline is pipeline_placeholder for pipeline, _promote in all_results)
    assert broker_snapshots[-1] == (pipeline_placeholder, sentinel_placeholder)
    assert dependency_broker.active_owner
    assert all(promote is all_results[0][1] for _pipe, promote in all_results)

    dependency_broker.clear()
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()


def test_prepare_pipeline_reentrancy_without_sentinel_attaches_placeholder(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda: {})

    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()

    owner_guard = object()
    cbi._BOOTSTRAP_STATE.active_bootstrap_guard = owner_guard
    cbi._BOOTSTRAP_STATE.owner_depths = {owner_guard: 1}
    if hasattr(cbi._BOOTSTRAP_STATE, "sentinel_manager"):
        delattr(cbi._BOOTSTRAP_STATE, "sentinel_manager")

    owner, active_promise = cbi._GLOBAL_BOOTSTRAP_COORDINATOR.claim()
    assert owner
    assert cbi._GLOBAL_BOOTSTRAP_COORDINATOR.peek_active() is active_promise

    pipeline = SimpleNamespace(manager=SimpleNamespace())

    def _settle() -> None:
        time.sleep(1.0)
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.settle(
            active_promise, result=(pipeline, lambda *_a, **_k: None)
        )

    threading.Thread(target=_settle, daemon=True).start()

    init_count = 0

    class _Pipeline:
        def __init__(self, *_a, **_k):
            nonlocal init_count
            init_count += 1

    try:
        resolved_pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
            pipeline_cls=_Pipeline,
            context_builder=SimpleNamespace(),
            bot_registry=DummyRegistry(),
            data_bot=DummyDataBot(),
        )
    finally:
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
        if hasattr(cbi._BOOTSTRAP_STATE, "active_bootstrap_guard"):
            delattr(cbi._BOOTSTRAP_STATE, "active_bootstrap_guard")
        if hasattr(cbi._BOOTSTRAP_STATE, "owner_depths"):
            delattr(cbi._BOOTSTRAP_STATE, "owner_depths")
        if hasattr(cbi._BOOTSTRAP_STATE, "sentinel_manager"):
            delattr(cbi._BOOTSTRAP_STATE, "sentinel_manager")

    assert resolved_pipeline is pipeline
    assert init_count == 0
    assert promote is not None
    broker_pipeline, broker_sentinel = dependency_broker.resolve()
    assert broker_pipeline is not None
    assert broker_sentinel is not None
    assert getattr(broker_sentinel, "bootstrap_placeholder", False)
    assert any(
        record.message
        in {
            "prepare_pipeline.bootstrap.recursion_attached_sentinel",
            "prepare_pipeline.bootstrap.recursion_deferred",
        }
        and getattr(record, "caller_module", None) == __name__
        for record in caplog.records
    )
    dependency_broker.clear()


def test_prepare_pipeline_heartbeat_reentrancy_reuses_active(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=cbi.logger.name)
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()

    dependency_broker = cbi._bootstrap_dependency_broker()
    dependency_broker.clear()

    active_pipeline = SimpleNamespace(manager=SimpleNamespace())
    dependency_broker.advertise(
        pipeline=active_pipeline, sentinel=active_pipeline.manager, owner=True
    )

    owner, active_promise = cbi._GLOBAL_BOOTSTRAP_COORDINATOR.claim()
    assert owner

    settled = threading.Event()

    def _settle_promise() -> None:
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.settle(
            active_promise, result=(active_pipeline, lambda *_a, **_k: None)
        )
        settled.set()

    threading.Thread(target=_settle_promise, daemon=True).start()

    init_count = 0

    class _Pipeline:
        def __init__(self, *_a, **_k) -> None:
            nonlocal init_count
            init_count += 1

    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda: {"active": True})

    try:
        pipeline, _promote = cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=_Pipeline,
            context_builder=SimpleNamespace(),
            bot_registry=DummyRegistry(),
            data_bot=DummyDataBot(),
        )
    finally:
        cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
        dependency_broker.clear()

    assert settled.wait(timeout=1)
    assert pipeline is active_pipeline
    assert init_count == 0
    assert any(
        record.message == "prepare_pipeline.bootstrap.heartbeat_reuse_promise"
        for record in caplog.records
    )


def test_prepare_pipeline_enforces_minimum_timeout(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=cbi.logger.name)
    cbi._PREPARE_PIPELINE_WATCHDOG["stages"].clear()
    cbi._PREPARE_PIPELINE_WATCHDOG["timeouts"] = 0
    cbi._initialize_prepare_readiness(reset=True)

    monkeypatch.setattr(cbi, "_BOOTSTRAP_WAIT_TIMEOUT", 30.0, raising=False)

    perf_calls = itertools.chain([0.0, 500.0, 500.0], itertools.repeat(500.0))
    monkeypatch.setattr(cbi.time, "perf_counter", lambda: next(perf_calls))

    class _Pipeline:
        def __init__(self, *_, **__):
            pass

    cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Pipeline,
        context_builder=SimpleNamespace(),
        bot_registry=DummyRegistry(),
        data_bot=DummyDataBot(),
        timeout=30.0,
    )

    degraded_events = [
        entry
        for entry in cbi._PREPARE_PIPELINE_WATCHDOG["stages"]
        if entry.get("timeout") and entry.get("degraded_but_online")
    ]
    assert degraded_events, "watchdog should record degraded readiness when overruns occur"
    last_stage = degraded_events[-1]
    assert last_stage.get("resolved_timeout", 0) >= cbi._MIN_STAGE_TIMEOUT

