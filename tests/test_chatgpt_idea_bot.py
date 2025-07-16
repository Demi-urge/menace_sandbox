import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json
from types import SimpleNamespace

import menace.chatgpt_idea_bot as cib


def test_build_prompt():
    msg = cib.build_prompt(["ai", "fintech"], prior="e-commerce")
    assert "e-commerce" in msg[0]["content"]
    assert "ai, fintech" in msg[0]["content"]


def test_parse_ideas():
    resp = {
        "choices": [
            {"message": {"content": json.dumps([{"name": "A", "description": "B", "tags": ["t"]}])}}
        ]
    }
    ideas = cib.parse_ideas(resp)
    assert len(ideas) == 1
    assert ideas[0].name == "A"
    assert ideas[0].tags == ["t"]


def test_generate_and_filter(monkeypatch):
    fake_resp = {
        "choices": [
            {"message": {"content": json.dumps([
                {"name": "Idea1", "description": "d1", "tags": []},
                {"name": "Idea2", "description": "d2", "tags": []},
            ])}}
        ]
    }

    client = cib.ChatGPTClient("key")
    monkeypatch.setattr(client, "ask", lambda msgs: fake_resp)
    validator = cib.SocialValidator()
    monkeypatch.setattr(validator, "is_unique_online", lambda name: name == "Idea1")
    monkeypatch.setattr(cib.database_manager, "search_models", lambda name: [])

    ideas = cib.generate_and_filter(["ai"], client, validator)
    names = [i.name for i in ideas]
    assert names == ["Idea1",]


def test_handoff_to_database(monkeypatch, tmp_path):
    idea = cib.Idea(name="ModelA", description="d", tags=["x"])
    recorded = {}

    class FakeBot:
        def __init__(self, db_path):
            self.db_path = db_path

        def ingest_idea(self, name, *, tags=(), source="", urls=(), **k):
            recorded["args"] = (name, list(tags), source, list(urls), self.db_path)

    monkeypatch.setattr(cib, "DatabaseManagementBot", FakeBot)
    db = tmp_path / "m.db"
    cib.handoff_to_database(idea, db_path=db)

    assert recorded["args"][0] == "ModelA"
    assert recorded["args"][1] == ["x"]
