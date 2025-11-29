import threading
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
