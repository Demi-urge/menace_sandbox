import types
import sys
import pytest

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

import menace_sandbox.chatgpt_idea_bot as cib  # noqa: E402


class DummyBuilder:
    def __init__(self):
        self.calls = []
        self.kwargs = []

    def refresh_db_weights(self):
        self.refreshed = True

    def build(self, query, **kwargs):
        self.calls.append(query)
        self.kwargs.append(kwargs)
        return "vector:" + query


def test_builder_context_included():
    builder = DummyBuilder()
    client = cib.ChatGPTClient(context_builder=builder)
    msgs = client.build_prompt_with_memory(
        ["alpha", "beta"], "hi", context_builder=builder
    )
    assert builder.calls == ["alpha beta"]
    assert "session_id" in builder.kwargs[0]
    assert (
        builder.kwargs[0]["session_id"]
        == msgs[0]["metadata"]["retrieval_session_id"]
    )
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"].startswith("hi")
    assert "vector:alpha beta" in msgs[0]["content"]


def test_fallback_result_empty_context():
    class FB(DummyBuilder):
        def build(self, query, **kwargs):
            super().build(query, **kwargs)
            return cib.FallbackResult()

    builder = FB()
    client = cib.ChatGPTClient(context_builder=builder)
    msgs = client.build_prompt_with_memory(["alpha"], "hi", context_builder=builder)
    assert builder.calls == ["alpha"]
    assert "session_id" in builder.kwargs[0]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hi"
    assert msgs[0]["metadata"]["retrieval_session_id"]


def test_requires_context_builder():
    builder = DummyBuilder()
    client = cib.ChatGPTClient(context_builder=builder)
    with pytest.raises(ValueError):
        client.build_prompt_with_memory(["alpha"], "hi", context_builder=None)


def test_builder_context_compressed():
    long_text = "x" * 500

    class LongBuilder(DummyBuilder):
        def build(self, query, **kwargs):
            super().build(query, **kwargs)
            return long_text

    builder = LongBuilder()
    client = cib.ChatGPTClient(context_builder=builder)
    msgs = client.build_prompt_with_memory(["alpha"], "hi", context_builder=builder)
    content = msgs[0]["content"].split("\n", 1)[1]
    assert len(content) <= 200
    assert content.endswith("...")
