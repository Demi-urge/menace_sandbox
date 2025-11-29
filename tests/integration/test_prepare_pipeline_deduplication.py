from collections import deque
import threading
import time
from types import SimpleNamespace

import pytest

import coding_bot_interface as cbi
from tests.test_bootstrap_manager_self_coding import DummyDataBot, DummyRegistry


@pytest.mark.integration
def test_prepare_pipeline_reuses_global_bootstrap_token() -> None:
    registry = DummyRegistry()
    data_bot = DummyDataBot()
    builder_primary = SimpleNamespace(label="primary-owner")
    builder_secondary = SimpleNamespace(label="secondary-owner")
    start_event = threading.Event()
    release_event = threading.Event()
    constructor_calls: list[object] = []

    class SlowPipeline:
        def __init__(
            self,
            *,
            context_builder: object,
            bot_registry: object,
            data_bot: object,
            manager: object,
            **_: object,
        ) -> None:
            constructor_calls.append(manager)
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager
            start_event.set()
            release_event.wait(timeout=5)

    def _bootstrap_primary() -> None:
        cbi._prepare_pipeline_for_bootstrap_impl(
            pipeline_cls=SlowPipeline,
            context_builder=builder_primary,
            bot_registry=registry,
            data_bot=data_bot,
            bootstrap_guard=False,
        )

    bootstrap_thread = threading.Thread(target=_bootstrap_primary)
    bootstrap_thread.start()
    assert start_event.wait(timeout=5)

    secondary_pipeline, secondary_promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=SlowPipeline,
        context_builder=builder_secondary,
        bot_registry=registry,
        data_bot=data_bot,
        bootstrap_guard=False,
    )

    release_event.set()
    bootstrap_thread.join(timeout=5)

    assert constructor_calls, "expected the slow pipeline to be constructed once"
    assert len(constructor_calls) == 1
    assert secondary_pipeline is not None
    assert callable(secondary_promote)


def test_nested_helper_reuses_active_pipeline(monkeypatch):
    owner_guard = object()
    sentinel = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_candidate = SimpleNamespace(
        manager=sentinel, initial_manager=None, bootstrap_placeholder=True
    )

    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.active_bootstrap_token = object()  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.active_bootstrap_guard = owner_guard  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.sentinel_manager = sentinel  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.pipeline = pipeline_candidate  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {owner_guard: 1}  # type: ignore[attr-defined]
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()

    class _Unreachable:
        def __init__(self, **_: object) -> None:  # pragma: no cover - guard must short circuit
            raise AssertionError("pipeline constructor should be bypassed")

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Unreachable,
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        bootstrap_guard=True,
    )

    guard_state = cbi._PREPARE_PIPELINE_WATCHDOG.get("bootstrap_guard", {})

    assert pipeline is pipeline_candidate
    assert guard_state.get("short_circuit") is True
    assert guard_state.get("owner_depths", {}).get(owner_guard) == 1
    assert callable(promote)

    promote(object())

    assert ("reuse", id(pipeline_candidate)) in cbi._PREPARE_PIPELINE_WATCHDOG.get(
        "promotion_callbacks", set()
    )


def test_reentrant_promoter_applies_manager_once(monkeypatch):
    sentinel = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_candidate = SimpleNamespace(
        manager=sentinel, initial_manager=None, bootstrap_placeholder=True
    )
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.sentinel_manager = sentinel  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.pipeline = pipeline_candidate  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()

    assigned: list[object] = []
    real_assign = cbi._assign_bootstrap_manager_placeholder

    def _spy_assign(target: object, placeholder: object, **kwargs: object) -> bool:
        assigned.append(placeholder)
        return real_assign(target, placeholder, **kwargs)

    monkeypatch.setattr(
        cbi,
        "_assign_bootstrap_manager_placeholder",
        _spy_assign,
    )

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=type("_SentinelCarrier", (), {}),
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        bootstrap_guard=True,
    )

    real_manager = object()
    promote(real_manager)

    assert pipeline is pipeline_candidate
    assert pipeline_candidate.manager is real_manager
    assert pipeline_candidate.initial_manager is real_manager
    assert assigned.count(real_manager) == 1


def test_reentrant_waits_for_dependency_broker(monkeypatch):
    owner_guard = object()
    broker = cbi._bootstrap_dependency_broker()
    broker.clear()
    cbi._BOOTSTRAP_DEPENDENCY_BROKER.set(broker)
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()
    cbi._PREPARE_PIPELINE_WATCHDOG.update(
        {"stages": deque(maxlen=32), "timeouts": 0, "extensions": []}
    )
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.active_bootstrap_token = object()  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.active_bootstrap_guard = owner_guard  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {owner_guard: 1}  # type: ignore[attr-defined]

    advertise_event = threading.Event()
    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )

    def _advertise_pipeline() -> None:
        time.sleep(0.05)
        cbi._mark_bootstrap_placeholder(sentinel_placeholder)
        cbi._mark_bootstrap_placeholder(pipeline_placeholder)
        broker.advertise(
            pipeline=pipeline_placeholder,
            sentinel=sentinel_placeholder,
        )
        advertise_event.set()

    thread = threading.Thread(target=_advertise_pipeline, daemon=True)
    thread.start()

    class _Unreachable:
        def __init__(self, **_: object) -> None:  # pragma: no cover - must not be called
            raise AssertionError("pipeline constructor should be bypassed during reentry wait")

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Unreachable,
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        bootstrap_guard=False,
        bootstrap_wait_timeout=0.5,
    )

    thread.join(timeout=2)

    guard_state = cbi._PREPARE_PIPELINE_WATCHDOG.get("bootstrap_guard", {})

    assert advertise_event.is_set(), "expected to wait for broker advertisement"
    assert getattr(pipeline, "bootstrap_placeholder", False) is True
    assert getattr(pipeline, "manager", None) is sentinel_placeholder
    assert callable(promote)
    assert guard_state.get("short_circuit") is True
    assert guard_state.get("reentry_waited") is True


def test_placeholder_reentry_reuses_pipeline_concurrently(caplog):
    broker = cbi._bootstrap_dependency_broker()
    broker.clear()
    caplog.set_level("INFO", logger=cbi.logger.name)
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()
    cbi._PREPARE_PIPELINE_WATCHDOG["stages"] = deque(maxlen=32)

    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.sentinel_manager = sentinel_placeholder  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.pipeline = pipeline_placeholder  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]
    broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)

    constructor_calls: list[object] = []

    class _Unreachable:
        def __init__(self, **_: object) -> None:  # pragma: no cover - must be bypassed
            constructor_calls.append(object())

    pipelines: list[object] = []
    promoters: list[Callable[[object], None]] = []

    def _invoke_prepare() -> None:
        cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
        cbi._BOOTSTRAP_STATE.sentinel_manager = sentinel_placeholder  # type: ignore[attr-defined]
        cbi._BOOTSTRAP_STATE.pipeline = pipeline_placeholder  # type: ignore[attr-defined]
        cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]
        pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
            pipeline_cls=_Unreachable,
            context_builder=object(),
            bot_registry=object(),
            data_bot=object(),
            bootstrap_guard=True,
            bootstrap_wait_timeout=0.5,
        )
        pipelines.append(pipeline)
        promoters.append(promote)

    threads = [threading.Thread(target=_invoke_prepare) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    guard_state = cbi._PREPARE_PIPELINE_WATCHDOG.get("bootstrap_guard", {})

    assert constructor_calls == []
    assert pipelines
    unique_pipelines = {id(pipe) for pipe in pipelines}
    assert all(getattr(pipe, "bootstrap_placeholder", False) for pipe in pipelines)
    assert len(unique_pipelines) == 1
    assert all(callable(promote) for promote in promoters)
    assert guard_state.get("short_circuit") is True
    assert any(
        record.message.startswith("prepare_pipeline.bootstrap.reentry_block")
        for record in caplog.records
    )
