"""Regression tests for the bootstrap_self_coding helper script."""

from __future__ import annotations

import logging
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
