import json
import logging

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
    client = cib.ChatGPTClient("key")
    monkeypatch.setattr(ceb, "ask_with_memory", lambda *a, **k: resp)
    router = ceb.init_db_router("enhprop", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = ceb.EnhancementDB(tmp_path / "enh.db", router=router)
    db.add_embedding = lambda *a, **k: None
    bot = ceb.ChatGPTEnhancementBot(client, db=db)
    monkeypatch.setattr(bot, "_feasible", lambda e: True)
    results = bot.propose("Improve", num_ideas=1, context="ctx")
    assert results and results[0].context == "ctx"
    entries = db.fetch()
    assert entries and entries[0].idea == "New"


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
    assert first == second == captured["id"]
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 1
    assert "duplicate" in caplog.text.lower()
