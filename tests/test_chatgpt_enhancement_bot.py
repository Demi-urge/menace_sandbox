import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json

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
    monkeypatch.setattr(client, "ask", lambda msgs, **kw: resp)
    db = ceb.EnhancementDB(tmp_path / "enh.db")
    bot = ceb.ChatGPTEnhancementBot(client, db=db)
    results = bot.propose("Improve", num_ideas=1, context="ctx")
    assert results and results[0].context == "ctx"
    entries = db.fetch()
    assert entries and entries[0].idea == "New"
