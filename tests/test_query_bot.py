import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.query_bot as qb  # noqa: E402
import menace.chatgpt_idea_bot as cib  # noqa: E402
from vector_service.context_builder import ContextBuilder  # noqa: E402
from prompt_types import Prompt  # noqa: E402


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
    builder = ContextBuilder()
    builder.refresh_db_weights()
    monkeypatch.setattr(
        builder,
        "build_prompt",
        lambda q, **k: Prompt(user=q, examples=["x"]),
    )
    client = cib.ChatGPTClient("k", context_builder=builder)
    captured = {}

    def fake_ask(msgs, **_):
        captured["messages"] = msgs
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(client, "ask", fake_ask)
    fetcher = qb.DataFetcher({"foo": {"val": 1}})
    bot = qb.QueryBot(client, fetcher=fetcher, context_builder=builder)
    result = bot.process("get foo", "cid")
    assert result.data == {"foo": {"val": 1}}
    assert result.text == "ok"
    assert "get foo" in bot.history("cid")
    assert any("x" in m.get("content", "") for m in captured["messages"])


def test_requires_context_builder():
    with pytest.raises(TypeError):
        qb.QueryBot()
