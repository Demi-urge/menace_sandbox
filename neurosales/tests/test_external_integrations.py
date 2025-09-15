import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.external_integrations import (
    RedditHarvester,
    TwitterTracker,
    GPT4Client,
    PineconeLogger,
    InfluenceGraphUpdater,
)
from unittest.mock import patch, MagicMock
import pytest
from typing import Any
from prompt_types import Prompt
from billing.prompt_notice import PAYMENT_ROUTER_NOTICE, prepend_payment_notice


def test_reddit_harvester_basic():
    harvester = RedditHarvester()
    resp = MagicMock()
    resp.json.return_value = {"data": [{"id": "1", "body": "dopamine", "parent_id": "0"}]}
    with patch("neurosales.external_integrations.requests.Session.get", return_value=resp) as p:
        comments = harvester.harvest(["neuro"], ["dopamine"])
    assert comments[0]["id"] == "1"
    assert p.called


def test_twitter_tracker_refresh():
    calls = []

    def token_hook():
        calls.append(1)
        return "tok"

    tracker = TwitterTracker(token_hook)
    resp1 = MagicMock()
    resp1.status_code = 401
    resp2 = MagicMock()
    resp2.status_code = 200
    resp2.json.return_value = {"data": []}
    with patch("neurosales.external_integrations.requests.Session.get", side_effect=[resp1, resp2]):
        data = tracker.search_hashtag("#SfN")
    assert data["data"] == []
    assert len(calls) >= 2


def test_gpt4_client_stream(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_chat(prompt: Prompt, **kwargs):
        captured["prompt_user"] = prompt.user
        captured["intent"] = prompt.metadata.get("intent")
        return {"choices": [{"message": {"content": "Hi!"}}]}

    monkeypatch.setattr(
        "neurosales.external_integrations.chat_completion_create", fake_chat
    )

    class DummyBuilder:
        def build_prompt(self, query: str, *, intent=None, **kwargs):
            captured["query"] = query
            captured["intent_called"] = intent
            return Prompt(user=query)

    client = GPT4Client("k", context_builder=DummyBuilder())
    out = list(client.stream_chat("user", [0.1], "persuade", "hello"))
    assert "".join(out) == "Hi!"
    assert captured["query"] == "hello"
    assert captured["intent_called"]["objective"] == "persuade"


def test_gpt4_client_stream_with_context_builder(monkeypatch):
    captured: dict[str, str] = {}

    class DummyBuilder:
        def build_prompt(self, query: str, *, intent=None, **kwargs):
            captured["query"] = query
            return Prompt(user="CTX")

    def fake_chat(prompt: Prompt, **kwargs):
        return {"choices": [{"message": {"content": "Hi!"}}]}

    monkeypatch.setattr(
        "neurosales.external_integrations.chat_completion_create", fake_chat
    )

    client = GPT4Client("k", context_builder=DummyBuilder())
    out = list(client.stream_chat("user", [0.1], "persuade", "hello"))
    assert "".join(out) == "Hi!"
    assert captured["query"] == "hello"


def test_gpt4_client_requires_context_builder():
    with pytest.raises(TypeError):
        GPT4Client("k")  # type: ignore[misc]


def test_gpt4_client_env(monkeypatch, caplog):
    caplog.set_level("WARNING")

    class DummyBuilder:
        def build_prompt(self, query: str, *, intent=None, **kwargs):  # pragma: no cover - simple stub
            return Prompt(user="CTX")

    monkeypatch.setenv("NEURO_OPENAI_KEY", "env-k")
    client = GPT4Client(None, context_builder=DummyBuilder())
    assert client.api_key == "env-k"
    assert client.enabled
    monkeypatch.delenv("NEURO_OPENAI_KEY")
    caplog.clear()
    client2 = GPT4Client(None, context_builder=DummyBuilder())
    assert not client2.enabled
    assert "disabled" in caplog.text.lower()


def test_pinecone_logger_operations():
    fake_index = MagicMock()
    with patch("neurosales.external_integrations.pinecone") as pc:
        pc.init.return_value = None
        pc.list_indexes.return_value = ["idx"]
        pc.Index.return_value = fake_index
        logger = PineconeLogger("idx", api_key="k", environment="us-east")
        logger.log("u", [0.0] * 1536, "hi")
        fake_index.upsert.assert_called_once()
        fake_index.query.return_value = {"matches": []}
        res = logger.query([0.0] * 1536)
    assert "matches" in res


def test_pinecone_logger_env_defaults(monkeypatch):
    fake_index = MagicMock()
    monkeypatch.setenv("NEURO_PINECONE_INDEX", "idx")
    monkeypatch.setenv("NEURO_PINECONE_KEY", "key")
    monkeypatch.setenv("NEURO_PINECONE_ENV", "us-east")
    with patch("neurosales.external_integrations.pinecone") as pc:
        pc.init.return_value = None
        pc.list_indexes.return_value = []
        pc.Index.return_value = fake_index
        logger = PineconeLogger()
    pc.init.assert_called_with(api_key="key", environment="us-east")
    pc.create_index.assert_called_with("idx", dimension=1536)
    assert logger.enabled


def test_pinecone_logger_missing_conf(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.delenv("NEURO_PINECONE_INDEX", raising=False)
    monkeypatch.delenv("NEURO_PINECONE_KEY", raising=False)
    monkeypatch.delenv("NEURO_PINECONE_ENV", raising=False)
    with patch("neurosales.external_integrations.pinecone"):
        logger = PineconeLogger()
    assert not logger.enabled
    assert "disabled" in caplog.text.lower()


def test_gpt4_client_stream_warns_when_unavailable(monkeypatch, caplog):
    class DummyBuilder:
        def build_prompt(self, query: str, *, intent=None, **kwargs):  # pragma: no cover - simple stub
            return Prompt(user="CTX")

    client = GPT4Client(None, context_builder=DummyBuilder())
    caplog.set_level("WARNING")
    caplog.clear()
    out = list(client.stream_chat("user", [0.1], "persuade", "hi"))
    assert out == [""]
    assert any("unavailable" in r.getMessage() for r in caplog.records)


def test_pinecone_logger_warns_when_unavailable(caplog):
    caplog.set_level("WARNING")
    with patch("neurosales.external_integrations.pinecone", None):
        logger = PineconeLogger("idx")
        caplog.clear()
        logger.log("u", [], "t")
        logger.query([])
    assert any("library unavailable" in r.getMessage() for r in caplog.records)


def test_influence_graph_updater_warns_when_unavailable(caplog):
    caplog.set_level("WARNING")
    with patch("neurosales.external_integrations.GraphDatabase", None):
        updater = InfluenceGraphUpdater("bolt://x", ("u", "p"))
        caplog.clear()
        updater.batch_update(["A"], [("A", "B", "influences")])
    assert any("library unavailable" in r.getMessage() for r in caplog.records)


def test_influence_graph_updater_batch():
    fake_session = MagicMock()
    fake_driver = MagicMock()
    fake_driver.session.return_value.__enter__.return_value = fake_session
    with patch("neurosales.external_integrations.GraphDatabase.driver", return_value=fake_driver):
        updater = InfluenceGraphUpdater("bolt://x", ("u", "p"))
        updater.batch_update(["A"], [("A", "B", "influences")])
    assert fake_session.run.called


def test_influence_graph_updater_env(monkeypatch):
    monkeypatch.setenv("NEURO_NEO4J_URI", "bolt://env")
    monkeypatch.setenv("NEURO_NEO4J_USER", "user")
    monkeypatch.setenv("NEURO_NEO4J_PASS", "pass")
    fake_driver = MagicMock()
    with patch("neurosales.external_integrations.GraphDatabase.driver", return_value=fake_driver) as drv:
        updater = InfluenceGraphUpdater()
    drv.assert_called_with("bolt://env", auth=("user", "pass"))
    assert updater.enabled


def test_check_external_services_injects_notice(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_chat(prompt, **kwargs):
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        for ex in prompt.examples:
            messages.append({"role": "system", "content": ex})
        messages.append({"role": "user", "content": prompt.user})
        captured["messages"] = prepend_payment_notice(messages)
        return {}

    from types import SimpleNamespace
    import sys
    monkeypatch.setitem(
        sys.modules,
        "billing.openai_wrapper",
        SimpleNamespace(chat_completion_create=fake_chat),
    )
    monkeypatch.setitem(
        sys.modules,
        "context_builder_util",
        SimpleNamespace(
            create_context_builder=lambda: SimpleNamespace(
                build_prompt=lambda *_args, **_kwargs: Prompt(user="CTX")
            )
        ),
    )
    from neurosales.scripts import check_external_services as ces
    monkeypatch.setattr(ces.config, "is_openai_enabled", lambda cfg: True)

    cfg = SimpleNamespace(openai_key="k")
    assert ces.check_openai(cfg)
    assert captured["messages"][0]["content"].startswith(PAYMENT_ROUTER_NOTICE)
