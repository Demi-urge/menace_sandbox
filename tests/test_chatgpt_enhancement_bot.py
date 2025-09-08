import json
import logging
import pytest
import sys
import types


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build(self, query, **_):
        return ""


_ctx_mod = types.SimpleNamespace(
    ContextBuilder=DummyBuilder, ErrorResult=Exception, FallbackResult=Exception
)
sys.modules.setdefault("vector_service", types.SimpleNamespace(context_builder=_ctx_mod))
sys.modules["vector_service.context_builder"] = _ctx_mod

import menace.chatgpt_enhancement_bot as ceb
import menace.chatgpt_idea_bot as cib


def test_summarise_text():
    text = "A. B. C."
    summary = ceb.summarise_text(text, ratio=0.34)
    assert "A" in summary
    assert summary.count(".") <= 2


def test_propose(monkeypatch, tmp_path):
    resp = {
        "choices": [
            {"message": {"content": json.dumps([{"idea": "New", "rationale": "More efficient"}])}}
        ]
    }
    builder = DummyBuilder()
    client = cib.ChatGPTClient("key", context_builder=builder)
    monkeypatch.setattr(ceb, "ask_with_memory", lambda *a, **k: resp)
    router = ceb.init_db_router("enhprop", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = ceb.EnhancementDB(tmp_path / "enh.db", router=router)
    db.add_embedding = lambda *a, **k: None
    bot = ceb.ChatGPTEnhancementBot(client, db=db, context_builder=builder)
    monkeypatch.setattr(bot, "_feasible", lambda e: True)
    results = bot.propose("Improve", num_ideas=1, context="ctx")
    assert results and results[0].context == "ctx"
    entries = db.fetch()
    assert entries and entries[0].idea == "New"


def test_requires_client_context_builder(tmp_path):
    class DummyClient:
        pass

    builder = DummyBuilder()
    db = ceb.EnhancementDB(tmp_path / "e.db")
    with pytest.raises(ValueError):
        ceb.ChatGPTEnhancementBot(DummyClient(), db=db, context_builder=builder)


def test_enhancementdb_duplicate(tmp_path, caplog, monkeypatch):
    router = ceb.init_db_router("enhdup", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = ceb.EnhancementDB(tmp_path / "enh.db", router=router)
    db.add_embedding = lambda *a, **k: None

    captured: dict[str, int | None] = {"id": None}
    orig = ceb.insert_if_unique

    def wrapper(*args, **kwargs):
        res = orig(*args, **kwargs)
        captured["id"] = res
        return res

    monkeypatch.setattr(ceb, "insert_if_unique", wrapper)

    enh = ceb.Enhancement(idea="i", rationale="r")
    first = db.add(enh)
    captured["id"] = None
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        second = db.add(ceb.Enhancement(idea="i", rationale="r"))
    assert first == second
    assert captured["id"] == first
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 1
    assert "duplicate" in caplog.text.lower()


def test_enhancementdb_content_hash_unique_index(tmp_path):
    old = ceb.GLOBAL_ROUTER
    router = ceb.init_db_router("enhidx", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = ceb.EnhancementDB(tmp_path / "enh.db", router=router)
    with db._connect() as conn:
        indexes = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA index_list('enhancements')").fetchall()
        }
    assert indexes.get("idx_enhancements_content_hash") == 1
    ceb.GLOBAL_ROUTER = old

