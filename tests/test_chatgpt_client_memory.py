import sys
import types

sys.modules.setdefault(
    "menace.database_manager", types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: [])
)
sys.modules.setdefault(
    "menace.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object)
)

import menace.chatgpt_idea_bot as cib  # noqa: E402


class FakeMemory:
    def __init__(self):
        self.logged = []

    def fetch_context(self, tags):
        return "ctx:" + ",".join(tags)

    def log_interaction(self, prompt, response, tags):
        self.logged.append((prompt, response, tags))


def test_build_prompt_with_memory():
    mem = FakeMemory()

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=DummyBuilder())
    msgs = client.build_prompt_with_memory(["ai"], "hello")
    assert msgs[0]["role"] == "system"
    assert "ctx:ai" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hello"


def test_ask_logs_interaction(monkeypatch):
    mem = FakeMemory()

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=DummyBuilder())
    client.session = None  # force offline response
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "resp"}}]},
    )
    client.ask([{"role": "user", "content": "hi"}])
    assert mem.logged[0][0] == "hi"
    assert mem.logged[0][1] == "resp"
    assert cib.INSIGHT in mem.logged[0][2]
