import importlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

MODULE_NAME = "menace_sandbox.workflow_evolution_bot"


def _clear_module() -> None:
    sys.modules.pop(MODULE_NAME, None)


def test_import_does_not_trigger_registry_construction():
    _clear_module()
    with mock.patch("menace_sandbox.bot_registry.BotRegistry", side_effect=RuntimeError("boom")) as registry_ctor:
        module = importlib.import_module(MODULE_NAME)
        assert module is not None
        assert registry_ctor.call_count == 0
        with pytest.raises(RuntimeError):
            module._ensure_runtime_dependencies()
        assert registry_ctor.call_count == 1


def test_import_does_not_trigger_data_bot_construction():
    _clear_module()

    def _stub_registry() -> SimpleNamespace:
        registry = SimpleNamespace(
            register_bot=lambda *args, **kwargs: None,
            update_bot=lambda *args, **kwargs: None,
            graph=SimpleNamespace(nodes={}),
            modules={},
        )
        return registry

    with mock.patch(
        "menace_sandbox.bot_registry.BotRegistry", side_effect=_stub_registry
    ) as registry_ctor, mock.patch(
        "menace_sandbox.data_bot.DataBot", side_effect=RuntimeError("timeout")
    ) as data_bot_ctor:
        module = importlib.import_module(MODULE_NAME)
        assert module is not None
        assert registry_ctor.call_count == 0
        assert data_bot_ctor.call_count == 0
        with pytest.raises(RuntimeError):
            module._ensure_runtime_dependencies()
        assert registry_ctor.call_count == 1
        assert data_bot_ctor.call_count == 1
