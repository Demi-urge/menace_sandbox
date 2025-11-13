"""Smoke tests for the manual bootstrap CLI helpers."""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import manual_bootstrap


def test_register_balolos_coder_promotes_pipeline(monkeypatch, caplog):
    """Ensure the bootstrap pipeline installs a concrete manager."""

    disabled_cls = getattr(manual_bootstrap, "_DisabledSelfCodingManager", None)
    assert disabled_cls is not None, "manual_bootstrap must expose the sentinel"

    pipeline_box: dict[str, SimpleNamespace] = {}
    manager_box: dict[str, SimpleNamespace] = {}
    promote_calls: list[SimpleNamespace] = []

    def fake_prepare_pipeline_for_bootstrap(
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
            sentinel = disabled_cls(bot_registry=bot_registry, data_bot=data_bot)
        pipeline = SimpleNamespace(
            pipeline_cls=pipeline_cls,
            context_builder=context_builder,
            bot_registry=bot_registry,
            data_bot=data_bot,
            manager=sentinel,
            sentinel_manager=sentinel,
        )
        pipeline_box["pipeline"] = pipeline

        def _promote(manager):
            pipeline.manager = manager
            promote_calls.append(manager)

        return pipeline, _promote

    monkeypatch.setattr(
        manual_bootstrap,
        "prepare_pipeline_for_bootstrap",
        fake_prepare_pipeline_for_bootstrap,
    )

    class DummyEngine:
        def __init__(self) -> None:
            self.context_builder = object()

    class DummyDataBot:
        pass

    class DummyRegistry:
        def __init__(self) -> None:
            self._bots: dict[str, SimpleNamespace] = {}

        def register_bot(self, name: str, manager: SimpleNamespace) -> None:
            self._bots[name] = manager

        def get_all_bots(self):  # pragma: no cover - exercised indirectly
            return list(self._bots.items())

    class DummyPipeline:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - compatibility shim
            self.kwargs = kwargs

    def fake_internalize_coding_bot(*, name, engine, pipeline, registry, **_):
        assert pipeline is pipeline_box["pipeline"]
        assert pipeline.manager is pipeline.sentinel_manager
        manager = SimpleNamespace(name=name, engine=engine, pipeline=pipeline)
        registry.register_bot(name, manager)
        manager_box["manager"] = manager
        return manager

    engine_module = types.ModuleType("menace.self_coding_engine")
    engine_module.SelfCodingEngine = DummyEngine
    data_module = types.ModuleType("menace.data_bot")
    data_module.DataBot = DummyDataBot
    registry_module = types.ModuleType("menace.bot_registry")
    registry_module.BotRegistry = DummyRegistry
    registry_module.internalize_coding_bot = fake_internalize_coding_bot
    pipeline_module = types.ModuleType("menace.model_automation_pipeline")
    pipeline_module.ModelAutomationPipeline = DummyPipeline

    monkeypatch.setitem(sys.modules, "menace.self_coding_engine", engine_module)
    monkeypatch.setitem(sys.modules, "menace.data_bot", data_module)
    monkeypatch.setitem(sys.modules, "menace.bot_registry", registry_module)
    monkeypatch.setitem(sys.modules, "menace.model_automation_pipeline", pipeline_module)

    caplog.set_level(logging.DEBUG)
    manual_bootstrap._register_balolos_coder()

    pipeline = pipeline_box["pipeline"]
    manager = manager_box["manager"]
    assert pipeline.manager is manager
    assert promote_calls == [manager]
    assert not isinstance(manager, disabled_cls)
    assert "re-entrant" not in caplog.text.lower()
