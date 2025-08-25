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


def test_enhancementdb_duplicate(tmp_path, caplog):
    router = ceb.init_db_router("enhdup", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = ceb.EnhancementDB(tmp_path / "enh.db", router=router)
    db.add_embedding = lambda *a, **k: None
    enh = ceb.Enhancement(idea="i", rationale="r")
    first = db.add(enh)
    tags = ",".join(enh.tags)
    assigned = ",".join(enh.assigned_bots)
    assoc = ",".join(enh.associated_bots)
    values = {
        "source_menace_id": db.router.menace_id,
        "idea": enh.idea,
        "rationale": enh.rationale,
        "summary": enh.summary,
        "score": enh.score,
        "timestamp": enh.timestamp,
        "context": enh.context,
        "before_code": enh.before_code,
        "after_code": enh.after_code,
        "title": enh.title,
        "description": enh.description,
        "tags": tags,
        "type": enh.type_,
        "assigned_bots": assigned,
        "rejection_reason": enh.rejection_reason,
        "cost_estimate": enh.cost_estimate,
        "category": enh.category,
        "associated_bots": assoc,
        "triggered_by": enh.triggered_by,
    }
    hash_fields = ["idea", "summary", "before_code", "after_code", "description"]
    with caplog.at_level(logging.WARNING):
        with db._connect() as conn:
            dup_id, inserted = ceb.insert_if_unique(
                conn, "enhancements", values, hash_fields, db.router.menace_id
            )
    assert dup_id == first
    assert not inserted
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 1
    assert "duplicate" in caplog.text.lower()
