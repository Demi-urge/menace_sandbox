import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json  # noqa: E402
import types
from prompt_types import Prompt  # noqa: E402
import menace_sandbox.chatgpt_idea_bot as cib  # noqa: E402


def test_build_prompt():
    class FakeMemory:
        def search_context(self, _q, tags=None):
            entry = types.SimpleNamespace(prompt="P", response="R")
            return [entry]

    class DummyBuilder:
        def __init__(self):
            self.last_kwargs = None
            self.enrich_calls = []

        def refresh_db_weights(self):
            pass

        def build_prompt(self, intent, **kwargs):
            self.last_kwargs = (intent, kwargs)
            return Prompt(user=intent, examples=["ctx"], metadata={"m": 1})

        def enrich_prompt(self, prompt, *, tags=None, metadata=None, origin=None):
            record = (
                prompt,
                list(tags or []),
                dict(metadata or {}),
                origin,
            )
            self.enrich_calls.append(record)
            meta = dict(getattr(prompt, "metadata", {}) or {})
            extra_meta = dict(metadata or {})
            for key, value in extra_meta.items():
                if key == "tags":
                    continue
                meta.setdefault(key, value)
            existing_tags = list(getattr(prompt, "tags", []) or [])
            meta_tags = list(meta.get("tags", []) or [])
            requested = [str(tag) for tag in (tags or []) if str(tag)]
            meta_requested = [
                str(tag) for tag in extra_meta.get("tags", []) or [] if str(tag)
            ]
            merged = list(
                dict.fromkeys([*existing_tags, *meta_tags, *meta_requested, *requested])
            )
            if merged:
                meta["tags"] = list(merged)
                prompt.tags = list(merged)
            prompt.metadata = meta
            prompt.origin = origin or prompt.origin or "context_builder"
            return prompt

    builder = DummyBuilder()
    client = cib.ChatGPTClient("key", context_builder=builder, gpt_memory=FakeMemory())
    prompt = cib.build_prompt(client, builder, ["ai", "fintech"], prior="e-commerce")

    intent, kwargs = builder.last_kwargs
    assert intent == [cib.IMPROVEMENT_PATH, "ai", "fintech"]
    assert kwargs["intent_metadata"]["tags"] == [cib.IMPROVEMENT_PATH, "ai", "fintech"]
    assert kwargs["intent_metadata"]["prior_ideas"] == "e-commerce"
    assert prompt.user == intent
    assert "ctx" in getattr(prompt, "examples", [])
    assert builder.enrich_calls
    _, tags_used, meta_used, origin_used = builder.enrich_calls[0]
    assert tags_used == [cib.IMPROVEMENT_PATH, "ai", "fintech"]
    assert meta_used["tags"] == [cib.IMPROVEMENT_PATH, "ai", "fintech"]
    assert meta_used["prior_ideas"] == "e-commerce"
    assert origin_used == "context_builder"
    assert prompt.metadata["tags"] == [cib.IMPROVEMENT_PATH, "ai", "fintech"]
    assert prompt.metadata["prior_ideas"] == "e-commerce"


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

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_prompt(self, query, **_):
            return Prompt(user=query)
    builder = DummyBuilder()
    client = cib.ChatGPTClient("key", context_builder=builder)

    def fake_generate(prompt_obj, *, context_builder, tags):
        assert context_builder is builder
        return cib.LLMResult(raw=fake_resp, text=fake_resp["choices"][0]["message"]["content"])

    monkeypatch.setattr(client, "generate", fake_generate)
    validator = cib.SocialValidator()
    monkeypatch.setattr(validator, "is_unique_online", lambda name: name == "Idea1")
    monkeypatch.setattr(cib.database_manager, "search_models", lambda name: [])

    ideas = cib.generate_and_filter(["ai"], client, validator, builder)
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
