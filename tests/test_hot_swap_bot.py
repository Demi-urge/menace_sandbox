import importlib
from menace.bot_registry import BotRegistry


def test_hot_swap_bot_reloads_module(tmp_path, monkeypatch):
    module = tmp_path / "dummy_bot.py"
    module.write_text("def greet():\n    return 'old'\n")
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    reg = BotRegistry()
    reg.update_bot("dummy", module.as_posix())
    assert dummy.greet() == "old"
    module.write_text("def greet():\n    return 'new'\n# change\n")
    importlib.invalidate_caches()
    reg.update_bot("dummy", module.as_posix())
    assert dummy.greet() == "new"
