import sys
import types

# Ensure lightweight stubs for modules expected by chatgpt_idea_bot
sys.modules.setdefault(
    "menace_sandbox.database_manager",
    types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: []),
)
sys.modules.setdefault(
    "menace_sandbox.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object)
)

import menace_sandbox.chatgpt_idea_bot as cib  # noqa: E402
import menace_sandbox.newsreader_bot as nrb  # noqa: E402


class FakeMemory:
    def __init__(self):
        self.logged = []

    def log_interaction(self, prompt, response, tags):
        self.logged.append((prompt, response, list(tags)))


def test_monetise_event_records_interaction(monkeypatch):
    mem = FakeMemory()

    from prompt_types import Prompt

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_prompt(self, query, **_):
            return Prompt(user=query)

    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=DummyBuilder())
    client.session = None  # force offline mode
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "plan"}}]},
    )
    event = nrb.Event("title", "summary", "source", "now")
    result = nrb.monetise_event(client, event)
    assert result == "plan"
    assert mem.logged[0][0].startswith("Suggest monetisation strategies")
    assert mem.logged[0][1] == "plan"
    assert cib.INSIGHT in mem.logged[0][2]
