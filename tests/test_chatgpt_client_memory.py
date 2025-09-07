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

    builder = DummyBuilder()
    client = cib.ChatGPTClient(gpt_memory=mem, context_builder=builder)
    msgs = client.build_prompt_with_memory(["ai"], "hello", context_builder=builder)
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
