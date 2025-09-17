import sys
import types

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

        def build_prompt(self, query, *, intent_metadata=None, prior=None, **_):
            session_id = "sid"
            tags = list((intent_metadata or {}).get("tags", []) or [])
            if isinstance(query, (list, tuple)):
                query_text = " ".join(str(q) for q in query)
            else:
                query_text = str(query)
            effective_tags = list(tags) if tags else [query_text]
            context_raw = self.build(
                " ".join(tags) if tags else query_text,
                session_id=session_id,
            )
            context = context_raw if isinstance(context_raw, str) else ""
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
            if effective_tags:
                meta["intent_tags"] = list(effective_tags)
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

    builder = DummyBuilder()
    builder.memory = mem
    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=builder)
    prompt = client.build_prompt_with_memory(
        ["ai"], prior="hello", context_builder=builder
    )
    assert "hello" in prompt.user
    assert "ctx:ai" in prompt.user


def test_ask_logs_interaction(monkeypatch):
    mem = FakeMemory()

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

        def build_prompt(self, query, *, intent_metadata=None, prior=None, **_):
            session_id = "sid"
            tags = list((intent_metadata or {}).get("tags", []) or [])
            if isinstance(query, (list, tuple)):
                query_text = " ".join(str(q) for q in query)
            else:
                query_text = str(query)
            effective_tags = list(tags) if tags else [query_text]
            context_raw = self.build(
                " ".join(tags) if tags else query_text,
                session_id=session_id,
            )
            context = context_raw if isinstance(context_raw, str) else ""
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
            if effective_tags:
                meta["intent_tags"] = list(effective_tags)
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
    builder = DummyBuilder()
    builder.memory = mem
    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=builder)
    client.session = None  # force offline response
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "resp"}}]},
    )
    logged = []
    monkeypatch.setattr(
        cib,
        "log_with_tags",
        lambda mem, prompt, response, tags: logged.append((prompt, response, tags)),
    )
    prompt_obj = builder.build_prompt("hi")
    client.ask(prompt_obj, use_memory=False)
    assert "hi" in logged[0][0]
    assert logged[0][1] == "resp"
    assert cib.INSIGHT in logged[0][2]
