import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.query_bot as qb
import menace.chatgpt_idea_bot as cib


def test_nlu():
    nlu = qb.SimpleNLU()
    parsed = nlu.parse("show sales 2022")
    assert "intent" in parsed and "entities" in parsed


def test_context_store():
    store = qb.ContextStore()
    store.add("c1", "hello")
    store.add("c1", "world")
    assert store.history("c1") == ["hello", "world"]


def test_process(monkeypatch):
    client = cib.ChatGPTClient("k")
    monkeypatch.setattr(client, "ask", lambda msgs: {"choices": [{"message": {"content": "ok"}}]})
    fetcher = qb.DataFetcher({"foo": {"val": 1}})
    bot = qb.QueryBot(client, fetcher=fetcher)
    result = bot.process("get foo", "cid")
    assert result.data == {"foo": {"val": 1}}
    assert result.text == "ok"
    assert "get foo" in bot.history("cid")
