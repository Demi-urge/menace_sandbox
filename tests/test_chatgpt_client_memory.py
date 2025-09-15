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

        def build_prompt(self, tags, prior=None, intent_metadata=None):
            session_id = "sid"
            context = self.build(" ".join(tags), session_id=session_id)
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
            parts = [prior, memory_ctx, context]
            user = "\n".join(p for p in parts if p)
            return types.SimpleNamespace(
                user=user,
                examples=None,
                system=None,
                metadata={"retrieval_session_id": session_id},
            )

    builder = DummyBuilder()
    builder.memory = mem
    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=builder)
    msgs = client.build_prompt_with_memory(
        ["ai"], prior="hello", context_builder=builder
    )
    assert msgs[0]["role"] == "user"
    assert "hello" in msgs[0]["content"]
    assert "ctx:ai" in msgs[0]["content"]


def test_ask_logs_interaction(monkeypatch):
    mem = FakeMemory()

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

        def build_prompt(self, tags, prior=None, intent_metadata=None):
            session_id = "sid"
            context = self.build(" ".join(tags), session_id=session_id)
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
            parts = [prior, memory_ctx, context]
            user = "\n".join(p for p in parts if p)
            return types.SimpleNamespace(
                user=user,
                examples=None,
                system=None,
                metadata={"retrieval_session_id": session_id},
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
    client.ask([{"role": "user", "content": "hi"}], use_memory=False)
    assert logged[0][0] == "hi"
    assert logged[0][1] == "resp"
    assert cib.INSIGHT in logged[0][2]
