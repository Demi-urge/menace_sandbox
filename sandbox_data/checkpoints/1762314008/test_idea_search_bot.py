import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from types import SimpleNamespace

import menace.idea_search_bot as isb


def test_generate_queries():
    bank = isb.KeywordBank(topics=["ai"], phrases=["online business"])

    low_energy = 0.3
    queries_low = bank.generate_queries(low_energy)
    expected_low = [
        f"{p} {t}"
        for t in bank.quick_topics + bank.topics
        for p in bank.quick_phrases + bank.phrases
    ]
    assert queries_low == expected_low

    high_energy = 0.8
    queries_high = bank.generate_queries(high_energy)
    expected_high = [
        f"{p} {t}"
        for t in bank.scale_topics + bank.topics
        for p in bank.scale_phrases + bank.phrases
    ]
    assert queries_high == expected_high


def test_google_search_backoff(monkeypatch):
    calls = []

    def fake_get(url, params):
        calls.append(params["q"])
        if len(calls) == 1:
            return SimpleNamespace(status_code=429)
        return SimpleNamespace(status_code=200, json=lambda: {"items": []})

    session = SimpleNamespace(get=fake_get)
    client = isb.GoogleSearchClient("key", "cx", session=session, backoff=0.1, max_retries=2)
    data = client.search("test")
    assert calls == ["test", "test"]
    assert data == {"items": []}


def test_discover_filters_existing(monkeypatch):
    bank = isb.KeywordBank(topics=["ai"], phrases=["startup idea"])

    def fake_search(q):
        return {
            "items": [
                {"title": "AI Startup", "link": "a", "snippet": ""},
                {"title": "AI Startup", "link": "a", "snippet": ""},
                {"title": "Other", "link": "b", "snippet": "startup idea ai"},
            ]
        }

    client = SimpleNamespace(search=fake_search)
    results = isb.discover_new_models(client, bank, energy=0.3)
    links = [r.link for r in results]
    assert set(links) == {"a", "b"}


def test_handoff_to_database(monkeypatch, tmp_path):
    result = isb.Result(title="ModelA", link="http://a", snippet="")
    recorded = {}

    class FakeBot:
        def __init__(self, db_path):
            self.db_path = db_path

        def ingest_idea(self, name, *, tags=(), source="", urls=(), **kw):
            recorded["args"] = (name, list(tags), source, list(urls), self.db_path)

    monkeypatch.setattr(isb, "DatabaseManagementBot", FakeBot)
    isb.handoff_to_database(result, tags=["x"], db_bot=None, source="idea_search")
    # Since db_bot is None, FakeBot will be instantiated with db_path=None
    assert recorded["args"][0] == "ModelA"
    assert recorded["args"][1] == ["x"]

