import importlib
import json
import types

import pytest

from menace.bot_registry import BotRegistry
import menace.patch_provenance as patch_provenance


def _stub_service(monkeypatch, commits):
    class DummyService:
        def __init__(self, *a, **k):
            self.db = self

        def get(self, pid):
            commit = commits.get(pid)
            if commit is None:
                return None
            return types.SimpleNamespace(summary=json.dumps({"commit": commit}))

    monkeypatch.setattr(patch_provenance, "PatchProvenanceService", DummyService)


def _set_manager(reg, patch_id, commit):
    reg.register_bot("dummy")
    node = reg.graph.nodes["dummy"]
    manager = node.get("selfcoding_manager")
    if not manager:
        manager = types.SimpleNamespace()
        node["selfcoding_manager"] = manager
    manager._last_patch_id = patch_id
    manager._last_commit_hash = commit
    return manager


def test_hot_swap_bot_reloads_module(tmp_path, monkeypatch):
    commits = {1: "a", 2: "b"}
    _stub_service(monkeypatch, commits)
    module = tmp_path / "dummy_bot.py"
    module.write_text("def greet():\n    return 'old'\n")
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    reg = BotRegistry()
    _set_manager(reg, 1, "a")
    reg.update_bot("dummy", module.as_posix(), patch_id=1, commit="a")
    assert dummy.greet() == "old"
    module.write_text("def greet():\n    return 'new'\n# change\n")
    importlib.invalidate_caches()
    _set_manager(reg, 2, "b")
    reg.update_bot("dummy", module.as_posix(), patch_id=2, commit="b")
    assert dummy.greet() == "new"


class DummyBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def publish(self, name: str, payload: dict) -> None:  # pragma: no cover - simple stub
        self.events.append((name, payload))


def test_hot_swap_failure_reverts_and_persists(tmp_path, monkeypatch):
    commits = {1: "a", 2: "b"}
    _stub_service(monkeypatch, commits)
    module_good = tmp_path / "dummy_bot.py"
    module_good.write_text("def greet():\n    return 'old'\n")
    module_bad = tmp_path / "dummy_bot_new.py"
    module_bad.write_text("def greet():\n    return 'broken'\n(")  # syntax error
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    bus = DummyBus()
    reg = BotRegistry(event_bus=bus)
    _set_manager(reg, 1, "a")
    reg.update_bot("dummy", module_good.as_posix(), patch_id=1, commit="a")
    importlib.invalidate_caches()
    _set_manager(reg, 2, "b")
    with pytest.raises(Exception):
        reg.update_bot("dummy", module_bad.as_posix(), patch_id=2, commit="b")
    assert dummy.greet() == "old"
    assert bus.events[-1][0] == "bot:hot_swap_failed"
    assert reg.graph.nodes["dummy"]["version"] == 1
    node = reg.graph.nodes["dummy"]
    assert node["module"] == module_good.as_posix()
    assert node["last_good_module"] == module_good.as_posix()
    assert node["last_good_version"] == 1


def test_health_check_failure_reverts(tmp_path, monkeypatch):
    commits = {1: "a", 2: "b"}
    _stub_service(monkeypatch, commits)
    module = tmp_path / "dummy_bot.py"
    module.write_text("def greet():\n    return 'old'\n")
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    bus = DummyBus()
    reg = BotRegistry(event_bus=bus)
    _set_manager(reg, 1, "a")
    reg.update_bot("dummy", module.as_posix(), patch_id=1, commit="a")
    module.write_text("def greet():\n    return 'new'\n")
    importlib.invalidate_caches()

    def bad_heartbeat(_name):
        raise RuntimeError("no heartbeat")

    monkeypatch.setattr(reg, "record_heartbeat", bad_heartbeat)
    _set_manager(reg, 2, "b")
    with pytest.raises(Exception):
        reg.update_bot("dummy", module.as_posix(), patch_id=2, commit="b")
    assert dummy.greet() == "old"
    assert bus.events[-1][0] == "bot:hot_swap_failed"
    node = reg.graph.nodes["dummy"]
    assert node["module"] == module.as_posix()
    assert node["version"] == 1


def test_manual_commit_mismatch_rejected(tmp_path, monkeypatch):
    commits = {1: "a", 2: "c"}
    _stub_service(monkeypatch, commits)
    module_old = tmp_path / "dummy_bot.py"
    module_old.write_text("def greet():\n    return 'old'\n")
    module_new = tmp_path / "dummy_bot_new.py"
    module_new.write_text("def greet():\n    return 'new'\n")
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    bus = DummyBus()
    reg = BotRegistry(event_bus=bus)
    _set_manager(reg, 1, "a")
    reg.update_bot("dummy", module_old.as_posix(), patch_id=1, commit="a")
    importlib.invalidate_caches()
    with pytest.raises(RuntimeError, match="update blocked"):
        reg.update_bot("dummy", module_new.as_posix(), patch_id=2, commit="b")
    assert dummy.greet() == "old"
    event_name, payload = bus.events[-1]
    assert event_name == "bot:update_blocked"
    assert payload["reason"] == "unverified_provenance"
    node = reg.graph.nodes["dummy"]
    assert node["module"] == module_old.as_posix()
    assert node["version"] == 1
    assert node["update_blocked"]


def test_missing_provenance_blocks_update(tmp_path, monkeypatch):
    commits = {1: "a", 2: "b"}
    _stub_service(monkeypatch, commits)
    module_old = tmp_path / "dummy_bot.py"
    module_old.write_text("def greet():\n    return 'old'\n")
    module_new = tmp_path / "dummy_bot_new.py"
    module_new.write_text("def greet():\n    return 'new'\n")
    monkeypatch.syspath_prepend(tmp_path)
    dummy = importlib.import_module("dummy_bot")
    bus = DummyBus()
    reg = BotRegistry(event_bus=bus)
    _set_manager(reg, 1, "a")
    reg.update_bot("dummy", module_old.as_posix(), patch_id=1, commit="a")
    importlib.invalidate_caches()
    _set_manager(reg, 2, "b")
    with pytest.raises(RuntimeError, match="update blocked"):
        reg.update_bot("dummy", module_new.as_posix(), patch_id=2, commit="")
    assert dummy.greet() == "old"
    event_name, payload = bus.events[-1]
    assert event_name == "bot:update_blocked"
    assert payload["reason"] == "unverified_provenance"
    node = reg.graph.nodes["dummy"]
    assert node["module"] == module_old.as_posix()
    assert node["update_blocked"]


def test_provenance_mismatch_notifies_manager(tmp_path, monkeypatch):
    commits = {1: "a", 2: "c"}
    _stub_service(monkeypatch, commits)
    module_old = tmp_path / "dummy_bot.py"
    module_old.write_text("def greet():\n    return 'old'\n")
    module_new = tmp_path / "dummy_bot_new.py"
    module_new.write_text("def greet():\n    return 'new'\n")
    monkeypatch.syspath_prepend(tmp_path)
    importlib.import_module("dummy_bot")
    bus = DummyBus()

    class DummyManager:
        def __init__(self):
            self.cycles = []

        def register_patch_cycle(self, description, context_meta=None):
            self.cycles.append((description, context_meta))

    reg = BotRegistry(event_bus=bus)
    reg.register_bot("dummy")
    node = reg.graph.nodes["dummy"]
    manager = DummyManager()
    manager._last_patch_id = 1
    manager._last_commit_hash = "a"
    node["selfcoding_manager"] = manager
    reg.update_bot("dummy", module_old.as_posix(), patch_id=1, commit="a")
    manager._last_patch_id = 2
    manager._last_commit_hash = "b"
    importlib.invalidate_caches()
    with pytest.raises(RuntimeError, match="update blocked"):
        reg.update_bot("dummy", module_new.as_posix(), patch_id=2, commit="b")
    manager = reg.graph.nodes["dummy"]["selfcoding_manager"]
    assert manager.cycles
    desc, meta = manager.cycles[0]
    assert "manual change" in desc.lower()
    assert meta["reason"] == "provenance_mismatch"
