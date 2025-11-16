import sys
import types

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
sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(SharedVectorService=object),
)

import menace_sandbox.chatgpt_idea_bot as cib  # noqa: E402
from prompt_types import Prompt  # noqa: E402


class TrackingBuilder:
    def __init__(self):
        self.enrich_calls: list = []

    def refresh_db_weights(self) -> None:
        pass

    def build_prompt(self, goal, *, intent_metadata=None, **_):
        meta = dict(intent_metadata or {})
        tags = list(meta.get("tags", []) or [])
        if tags and not meta.get("intent_tags"):
            meta["intent_tags"] = list(tags)
        meta.setdefault("origin", "context_builder")
        return Prompt(
            goal,
            tags=tags,
            metadata=meta,
            origin=meta.get("origin"),
        )

    def enrich_prompt(self, prompt, *, tags=None, metadata=None, origin=None):
        self.enrich_calls.append((prompt, tuple(tags) if tags is not None else None, metadata, origin))
        meta = dict(getattr(prompt, "metadata", {}) or {})
        if metadata:
            meta.update(metadata)
        prompt.metadata = meta
        if origin:
            prompt.origin = origin
        return prompt


@pytest.fixture
def offline_response(monkeypatch):
    def _factory(client):
        client.session = None
        monkeypatch.setattr(
            client,
            "_offline_response",
            lambda _msgs: {"choices": [{"message": {"content": "ok"}}]},
        )
        monkeypatch.setattr(cib, "log_with_tags", lambda *a, **k: None)
        return client

    return _factory


def test_ask_rejects_message_list(offline_response):
    builder = TrackingBuilder()
    client = cib.ChatGPTClient(context_builder=builder, gpt_memory=None)
    offline_response(client)

    with pytest.raises(ValueError):
        client.ask([{"role": "user", "content": "hi"}])


def test_prompt_input_invokes_enrichment(offline_response):
    builder = TrackingBuilder()
    client = cib.ChatGPTClient(context_builder=builder, gpt_memory=None)
    offline_response(client)

    prompt_obj = builder.build_prompt("hello", intent_metadata={"tags": ["alpha"]})
    client.ask(prompt_obj, tags=["alpha"], use_memory=False)

    assert builder.enrich_calls, "context builder enrichment should be invoked"
    prompt_used, tags_used, metadata_used, origin_used = builder.enrich_calls[0]
    assert prompt_used is prompt_obj
    assert tags_used is None or list(tags_used)  # ensure tags were provided or normalized
    assert metadata_used is None or metadata_used.get("origin") == "context_builder"
    assert origin_used == "context_builder"
