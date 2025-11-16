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
from prompt_types import Prompt  # noqa: E402


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

    def build_prompt(self, query, *, intent_metadata=None, prior=None, **_):
        session_id = "sid"
        tags = list((intent_metadata or {}).get("tags", []) or [])
        if isinstance(query, (list, tuple)):
            query_text = " ".join(str(q) for q in query)
        else:
            query_text = str(query)
        context_raw = self.build(
            " ".join(tags) if tags else query_text,
            session_id=session_id,
        )
        context = context_raw if isinstance(context_raw, str) else ""
        if context and len(context) > 200:
            context = context[:197] + "..."
        memory_ctx = ""
        mem = getattr(self, "memory", None)
        if mem:
            fetch = getattr(mem, "fetch_context", None)
            if callable(fetch):
                memory_ctx = fetch(tags)
            else:
                search = getattr(mem, "search_context", None)
                if callable(search):
                    entries = search("", tags=tags)
                    if entries:
                        first = entries[0]
                        memory_ctx = getattr(first, "prompt", "") or getattr(first, "response", "")
        parts = [prior, memory_ctx, context, query_text]
        user = "\n".join(p for p in parts if p)
        meta = {"retrieval_session_id": session_id, "origin": "context_builder"}
        if tags:
            meta["tags"] = list(tags)
            meta["intent_tags"] = list(tags)
        if intent_metadata:
            extra_meta = dict(intent_metadata)
            extra_meta.pop("tags", None)
            meta.update(extra_meta)
        return Prompt(
            user,
            system="",
            examples=[],
            tags=list(tags),
            metadata=meta,
            origin="context_builder",
        )


def test_builder_context_included():
    builder = DummyBuilder()
    client = cib.ChatGPTClient(context_builder=builder)
    prompt = client.build_prompt_with_memory(
        ["alpha", "beta"], prior="hi", context_builder=builder
    )
    assert builder.calls == ["alpha beta"]
    assert "session_id" in builder.kwargs[0]
    assert (
        builder.kwargs[0]["session_id"]
        == prompt.metadata["retrieval_session_id"]
    )
    assert prompt.user.startswith("hi")
    assert "vector:alpha beta" in prompt.user


def test_fallback_result_empty_context():
    class FB(DummyBuilder):
        def build(self, query, **kwargs):
            super().build(query, **kwargs)
            return cib.FallbackResult()

    builder = FB()
    client = cib.ChatGPTClient(context_builder=builder)
    prompt = client.build_prompt_with_memory(
        ["alpha"], prior="hi", context_builder=builder
    )
    assert builder.calls == ["alpha"]
    assert "session_id" in builder.kwargs[0]
    assert prompt.user.splitlines()[0] == "hi"
    assert "alpha" in prompt.user
    assert prompt.metadata["retrieval_session_id"]


def test_requires_context_builder():
    builder = DummyBuilder()
    client = cib.ChatGPTClient(context_builder=builder)
    with pytest.raises(ValueError):
        client.build_prompt_with_memory(["alpha"], prior="hi", context_builder=None)


def test_builder_context_compressed():
    long_text = "x" * 500

    class LongBuilder(DummyBuilder):
        def build(self, query, **kwargs):
            super().build(query, **kwargs)
            return long_text

    builder = LongBuilder()
    client = cib.ChatGPTClient(context_builder=builder)
    prompt = client.build_prompt_with_memory(
        ["alpha"], prior="hi", context_builder=builder
    )
    lines = prompt.user.splitlines()
    assert len(lines) >= 3
    # Second line contains the compressed context
    assert len(lines[1]) <= 200
    assert lines[1].endswith("...")
