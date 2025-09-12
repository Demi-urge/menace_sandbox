import importlib
import pytest
from menace.bot_registry import BotRegistry


def test_hot_swap_bot_reloads_module(tmp_path, monkeypatch):
    module = tmp_path / "dummy_bot.py"
    module.write_text("def greet():\n    return 'old'\n")
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    reg = BotRegistry()
    reg.update_bot("dummy", module.as_posix(), patch_id=1, commit="a")
    assert dummy.greet() == "old"
    module.write_text("def greet():\n    return 'new'\n# change\n")
    importlib.invalidate_caches()
    reg.update_bot("dummy", module.as_posix(), patch_id=2, commit="b")
    assert dummy.greet() == "new"


class DummyBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def publish(self, name: str, payload: dict) -> None:  # pragma: no cover - simple stub
        self.events.append((name, payload))


def test_hot_swap_failure_reverts_and_persists(tmp_path, monkeypatch):
    module_good = tmp_path / "dummy_bot.py"
    module_good.write_text("def greet():\n    return 'old'\n")
    module_bad = tmp_path / "dummy_bot_new.py"
    module_bad.write_text("def greet():\n    return 'broken'\n(")  # syntax error
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    db = tmp_path / "reg.db"
    bus = DummyBus()
    reg = BotRegistry(persist=db, event_bus=bus)
    reg.update_bot("dummy", module_good.as_posix(), patch_id=1, commit="a")
    importlib.invalidate_caches()
    with pytest.raises(Exception):
        reg.update_bot("dummy", module_bad.as_posix(), patch_id=2, commit="b")
    assert dummy.greet() == "old"
    assert bus.events[-1][0] == "bot:hot_swap_failed"
    assert reg.graph.nodes["dummy"]["version"] == 1
    reg2 = BotRegistry(persist=db)
    node = reg2.graph.nodes["dummy"]
    assert node["module"] == module_good.as_posix()
    assert node["version"] == 1
    assert node["last_good_module"] == module_good.as_posix()
    assert node["last_good_version"] == 1
