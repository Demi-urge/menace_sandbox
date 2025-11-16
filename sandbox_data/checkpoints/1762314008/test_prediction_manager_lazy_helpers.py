"""Regression tests for lazy helper initialization in prediction manager."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class _DummyRegistry:
    def __init__(self) -> None:
        self.graph = SimpleNamespace(nodes={})
        self.modules: dict[str, object] = {}

    def register_bot(
        self,
        name: str,
        *,
        roi_threshold=None,
        error_threshold=None,
        test_failure_threshold=None,
        manager=None,
        data_bot=None,
        module_path=None,
        is_coding_bot: bool = False,
    ) -> None:
        self.graph.nodes.setdefault(name, {})
        if module_path is not None:
            self.modules[name] = module_path

    def update_bot(
        self,
        name: str,
        module_path,
        *,
        patch_id=None,
        commit=None,
    ) -> None:
        self.modules[name] = module_path

    def hot_swap_active(self) -> bool:
        return False


class _DummyDataBot:
    def __init__(self, *args, **kwargs) -> None:
        self.db = SimpleNamespace(fetch=lambda *a, **k: [])

    def reload_thresholds(self, name: str):
        return SimpleNamespace(
            roi_drop=None, error_threshold=None, test_failure_threshold=None
        )


def _reload_prediction_manager(monkeypatch, registry_ctor, data_bot_ctor):
    package = types.ModuleType("menace_sandbox")
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    monkeypatch.setitem(sys.modules, "menace_sandbox", package)

    bot_registry_mod = types.ModuleType("menace_sandbox.bot_registry")
    bot_registry_mod.BotRegistry = registry_ctor
    monkeypatch.setitem(sys.modules, "menace_sandbox.bot_registry", bot_registry_mod)

    data_bot_mod = types.ModuleType("menace_sandbox.data_bot")
    data_bot_mod.DataBot = data_bot_ctor

    class _MetricsDB:
        pass

    data_bot_mod.MetricsDB = _MetricsDB
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", data_bot_mod)

    coding_mod = types.ModuleType("menace_sandbox.coding_bot_interface")

    def _noop_self_coding_managed(**_kwargs):
        def decorator(cls):
            return cls

        return decorator

    coding_mod.self_coding_managed = _noop_self_coding_managed
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.coding_bot_interface", coding_mod
    )

    sys.modules.pop("menace_sandbox.prediction_manager_bot", None)
    return importlib.import_module("menace_sandbox.prediction_manager_bot")


def test_import_does_not_instantiate_helpers(monkeypatch):
    registry_calls = []
    data_bot_calls = []

    def registry_ctor(*args, **kwargs):
        instance = _DummyRegistry()
        registry_calls.append((args, kwargs, instance))
        return instance

    def data_bot_ctor(*args, **kwargs):
        instance = _DummyDataBot()
        data_bot_calls.append((args, kwargs, instance))
        return instance

    module = _reload_prediction_manager(monkeypatch, registry_ctor, data_bot_ctor)

    assert registry_calls == []
    assert data_bot_calls == []

    registry = module._get_registry()
    data_bot = module._get_data_bot()

    assert registry_calls, "BotRegistry constructor should be invoked lazily"
    assert data_bot_calls, "DataBot constructor should be invoked lazily"
    assert registry is registry_calls[0][2]
    assert data_bot is data_bot_calls[0][2]

    # Cached helpers should not construct new instances on repeated access
    assert module._get_registry() is registry
    assert module._get_data_bot() is data_bot
    assert len(registry_calls) == 1
    assert len(data_bot_calls) == 1
