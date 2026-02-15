import importlib
import sys

import pytest


@pytest.mark.parametrize("package_alias", ["menace", "menace_sandbox"])
def test_error_bot_import_is_lazy_for_bootstrap_helpers(monkeypatch, package_alias):
    pydantic_stub = type(sys)("pydantic")
    pydantic_stub.BaseModel = object
    pydantic_stub.Field = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_stub)

    try:
        data_bot_mod = importlib.import_module(f"{package_alias}.data_bot")
        bot_registry_mod = importlib.import_module(f"{package_alias}.bot_registry")
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"{package_alias} alias is not available in this environment: {exc}")

    calls = {"registry": 0, "data_bot": 0, "watcher": 0}

    class FakeBotRegistry:
        def __init__(self, *args, **kwargs):
            calls["registry"] += 1

    class FakeDataBot:
        def __init__(self, *args, **kwargs):
            calls["data_bot"] += 1

    monkeypatch.setattr(bot_registry_mod, "BotRegistry", FakeBotRegistry)
    monkeypatch.setattr(data_bot_mod, "DataBot", FakeDataBot)

    module_name = f"{package_alias}.error_bot"
    sys.modules.pop(module_name, None)
    sys.modules.pop("menace.error_bot", None)
    sys.modules.pop("menace_sandbox.error_bot", None)

    try:
        error_bot = importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} import unavailable in this environment: {exc}")

    monkeypatch.setattr(
        error_bot,
        "_ensure_backfill_watcher",
        lambda *_args, **_kwargs: calls.__setitem__("watcher", calls["watcher"] + 1),
    )

    assert calls["registry"] == 0
    assert calls["data_bot"] == 0
    assert calls["watcher"] == 0

    error_bot._get_registry()
    error_bot._get_data_bot()
    error_bot._ensure_watcher_started()

    assert calls["registry"] == 1
    assert calls["data_bot"] == 1
    assert calls["watcher"] == 1
