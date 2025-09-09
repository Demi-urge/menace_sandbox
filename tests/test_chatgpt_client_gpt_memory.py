import sys
import types

# stub modules required by ChatGPTClient
sys.modules.setdefault(
    "menace_sandbox.database_manager",
    types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: []),
)
sys.modules.setdefault(
    "menace_sandbox.database_management_bot",
    types.SimpleNamespace(DatabaseManagementBot=object),
)
sys.modules.setdefault(
    "menace_sandbox.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "menace_sandbox.memory_logging", types.SimpleNamespace(log_with_tags=lambda *a, **k: None)
)
sys.modules.setdefault(
    "menace_sandbox.memory_aware_gpt_client",
    types.SimpleNamespace(ask_with_memory=lambda *a, **k: {}),
)
sys.modules.setdefault(
    "menace_sandbox.local_knowledge_module",
    types.SimpleNamespace(LocalKnowledgeModule=lambda *a, **k: types.SimpleNamespace(memory=None)),
)
sys.modules.setdefault(
    "menace_sandbox.knowledge_retriever",
    types.SimpleNamespace(
        get_feedback=lambda *a, **k: [],
        get_improvement_paths=lambda *a, **k: [],
        get_error_fixes=lambda *a, **k: [],
    ),
)
sys.modules.setdefault(
    "governed_retrieval",
    types.SimpleNamespace(govern_retrieval=lambda *a, **k: None, redact=lambda x: x),
)
sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(SharedVectorService=object),
)

# stub sentence_transformers to avoid heavy import
stub_st = types.ModuleType("sentence_transformers")


class _DummyModel:
    def encode(self, text):
        return [0.0]


stub_st.SentenceTransformer = _DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)

import menace_sandbox.chatgpt_idea_bot as cib  # noqa: E402
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

    msgs = client.build_prompt_with_memory(
        ["topic"], "new question", context_builder=client.context_builder
    )
    assert msgs[0]["role"] == "user"
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
