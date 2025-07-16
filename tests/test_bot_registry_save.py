import sqlite3
import pytest

pytest.importorskip("networkx")

from menace.bot_registry import BotRegistry


def test_registry_save_roundtrip(tmp_path):
    reg = BotRegistry()
    reg.register_interaction("a", "b", 2.0)
    path = tmp_path / "g.db"
    reg.save(path)
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT weight FROM bot_edges WHERE from_bot='a' AND to_bot='b'"
    ).fetchone()
    assert row and row[0] == 2.0
    conn.close()


def test_registry_load(tmp_path):
    reg = BotRegistry()
    reg.register_interaction("a", "b", 2.0)
    path = tmp_path / "g.db"
    reg.save(path)

    reg2 = BotRegistry()
    reg2.load(path)
    assert reg2.graph.has_edge("a", "b")
    assert float(reg2.graph["a"]["b"].get("weight")) == 2.0


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
