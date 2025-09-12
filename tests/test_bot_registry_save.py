import json
import types

import pytest
from menace.bot_registry import BotRegistry
from menace.unified_event_bus import UnifiedEventBus
from db_router import init_db_router
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

pytest.importorskip("networkx")


def test_registry_save_roundtrip(tmp_path):
    path = tmp_path / "g.db"
    router = init_db_router("br", str(path), str(path))
    reg = BotRegistry()
    reg.register_interaction("a", "b", 2.0)
    reg.save(router)
    with router.get_connection("bots") as conn:
        row = conn.execute(
            "SELECT weight FROM bot_edges WHERE from_bot='a' AND to_bot='b'",
        ).fetchone()
    assert row and row[0] == 2.0
    assert router._access_counts["shared"]["bots"] >= 1
    router.close()


def test_registry_load(tmp_path):
    path = tmp_path / "g.db"
    router = init_db_router("br2", str(path), str(path))
    reg = BotRegistry()
    reg.register_interaction("a", "b", 2.0)
    reg.save(router)
    router.close()

    router = init_db_router("br2", str(path), str(path))
    reg2 = BotRegistry()
    reg2.load(router)
    assert reg2.graph.has_edge("a", "b")
    assert float(reg2.graph["a"]["b"].get("weight")) == 2.0
    assert router._access_counts["shared"]["bots"] >= 1
    router.close()


def test_update_bot_persists_module(tmp_path, monkeypatch):
    path = tmp_path / "g.db"
    router = init_db_router("br3", str(path), str(path))
    commits = {1: "abc"}
    _stub_service(monkeypatch, commits)
    reg = BotRegistry()
    module_file = tmp_path / "mod_x.py"
    module_file.write_text("x = 1\n")
    reg.update_bot("x", module_file.as_posix(), patch_id=1, commit="abc")
    reg.save(router)
    router.close()

    router = init_db_router("br3", str(path), str(path))
    reg2 = BotRegistry()
    reg2.load(router)
    assert reg2.graph.nodes["x"].get("module") == module_file.as_posix()
    with router.get_connection("bots") as conn:
        row = conn.execute(
            "SELECT module FROM bot_nodes WHERE name='x'",
        ).fetchone()
    assert row and row[0] == module_file.as_posix()
    router.close()


def test_init_logs_load_error(tmp_path, monkeypatch, caplog):
    path = tmp_path / "g.db"
    path.touch()

    def boom(self, src):
        raise RuntimeError("fail")

    monkeypatch.setattr(BotRegistry, "load", boom)
    caplog.set_level("ERROR")
    BotRegistry(persist=path)
    assert "Failed to load bot registry" in caplog.text


def test_register_bot_logs_save_error(tmp_path, monkeypatch, caplog):
    path = tmp_path / "g.db"
    reg = BotRegistry(persist=path)

    def boom(self, dest):
        raise RuntimeError("fail")

    monkeypatch.setattr(BotRegistry, "save", boom)
    caplog.set_level("ERROR")
    reg.register_bot("x")
    assert "Failed to save bot registry" in caplog.text


def test_update_bot_emits_event_and_increments_version(tmp_path, monkeypatch):
    commits = {1: "abc", 2: "def"}
    _stub_service(monkeypatch, commits)
    bus = UnifiedEventBus()
    events: list[dict[str, object]] = []
    bus.subscribe("bot:updated", lambda _t, e: events.append(e))
    reg = BotRegistry(event_bus=bus)
    module_file = tmp_path / "mod_x.py"
    module_file.write_text("x = 1\n")
    reg.update_bot("x", module_file.as_posix(), patch_id=1, commit="abc")
    reg.update_bot("x", module_file.as_posix(), patch_id=2, commit="def")
    assert reg.graph.nodes["x"].get("version") == 2
    assert events[0]["patch_id"] == 1 and events[0]["commit"] == "abc"
    assert events[1]["version"] == 2 and events[1]["commit"] == "def"
