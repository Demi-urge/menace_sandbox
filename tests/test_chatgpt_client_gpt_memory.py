import sys
import types

# stub modules required by ChatGPTClient
sys.modules.setdefault(
    "menace.database_manager", types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: [])
)
sys.modules.setdefault(
    "menace.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object)
)

# stub sentence_transformers to avoid heavy import
stub_st = types.ModuleType("sentence_transformers")


class _DummyModel:
    def encode(self, text):
        return [0.0]


stub_st.SentenceTransformer = _DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)

import menace.chatgpt_idea_bot as cib  # noqa: E402
from gpt_memory import GPTMemoryManager  # noqa: E402


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build(self, query, **_):
        return ""


def test_build_prompt_injects_summary_and_logs(monkeypatch):
    mem = GPTMemoryManager(db_path=":memory:")
    # previous interaction stored for tag 'topic'
    mem.log_interaction("early prompt", "early resp", ["topic"])

    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=DummyBuilder())
    client.session = None  # offline mode
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "later resp"}}]},
    )

    msgs = client.build_prompt_with_memory(["topic"], "new question")
    assert msgs[0]["role"] == "system"
    assert "early prompt" in msgs[0]["content"]

    client.ask(msgs)
    # confirm interaction was logged with default tags
    entries = mem.search_context("new question")
    assert any(e.response == "later resp" for e in entries)


def test_summarize_and_prune_via_client(monkeypatch):
    mem = GPTMemoryManager(db_path=":memory:")
    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=DummyBuilder())
    client.session = None
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "resp"}}]},
    )

    for i in range(3):
        client.ask([{"role": "user", "content": f"ask{i}"}], tags=["insight"])

    assert len(mem.search_context("", tags=["insight"])) == 3
    mem.compact({"insight": 1})
    remaining = [
        e for e in mem.search_context("", tags=["insight"]) if "summary" not in e.tags
    ]
    assert len(remaining) == 1 and remaining[0].prompt == "ask2"
    summaries = [
        e for e in mem.search_context("", tags=["insight"]) if "summary" in e.tags
    ]
    assert summaries
