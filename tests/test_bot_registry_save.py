import pytest
from menace.bot_registry import BotRegistry
from db_router import init_db_router

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


def test_update_bot_persists_module(tmp_path):
    path = tmp_path / "g.db"
    router = init_db_router("br3", str(path), str(path))
    reg = BotRegistry()
    reg.update_bot("x", "/mod/x")
    reg.save(router)
    router.close()

    router = init_db_router("br3", str(path), str(path))
    reg2 = BotRegistry()
    reg2.load(router)
    assert reg2.graph.nodes["x"].get("module") == "/mod/x"
    with router.get_connection("bots") as conn:
        row = conn.execute(
            "SELECT module FROM bot_nodes WHERE name='x'",
        ).fetchone()
    assert row and row[0] == "/mod/x"
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
