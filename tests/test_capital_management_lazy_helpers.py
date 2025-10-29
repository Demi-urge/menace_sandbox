"""Regression tests for lazy helper instantiation in capital_management_bot."""

from __future__ import annotations

import importlib
import sys
from unittest import mock

MODULE = "menace_sandbox.capital_management_bot"


def test_import_is_lazy_and_helpers_cache_instances() -> None:
    sys.modules.pop(MODULE, None)
    module = importlib.import_module(MODULE)

    assert module._registry_instance is None
    assert module._data_bot_instance is None
    assert module._context_builder_instance is None
    assert module._engine_instance is None

    with (
        mock.patch.object(module, "BotRegistry") as registry_ctor,
        mock.patch.object(module, "DataBot") as data_bot_ctor,
        mock.patch.object(module, "SelfCodingEngine") as engine_ctor,
        mock.patch.object(module, "create_context_builder") as ctx_ctor,
    ):
        module._registry_instance = None
        module._data_bot_instance = None
        module._context_builder_instance = None
        module._engine_instance = None
        module._capital_bot_class = None

        registry_instance = module._get_registry()
        assert registry_instance is registry_ctor.return_value
        assert registry_ctor.call_count == 1
        assert module._get_registry() is registry_instance

        data_bot_instance = module._get_data_bot()
        assert data_bot_instance is data_bot_ctor.return_value
        assert data_bot_ctor.call_count == 1
        assert module._get_data_bot() is data_bot_instance

        builder_instance = module._get_context_builder()
        assert builder_instance is ctx_ctor.return_value
        assert ctx_ctor.call_count == 1
        assert module._get_context_builder() is builder_instance

        engine_instance = module._get_engine()
        assert engine_instance is engine_ctor.return_value
        engine_ctor.assert_called_once()
        assert module._get_engine() is engine_instance

        capital_cls = module._get_capital_management_bot_class()
        assert capital_cls is module._get_capital_management_bot_class()
        assert getattr(module, "CapitalManagementBot") is capital_cls

    sys.modules.pop(MODULE, None)

