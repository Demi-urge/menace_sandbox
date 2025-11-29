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
