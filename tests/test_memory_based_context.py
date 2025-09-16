import types
import sys

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
from log_tags import IMPROVEMENT_PATH  # noqa: E402


class DummyMemory:
    def __init__(self) -> None:
        self.entries = []
        self.fetch_calls = []

    def log_interaction(self, prompt: str, response: str, tags):
        self.entries.append((prompt, response, list(tags)))

    def fetch_context(self, tags):
        self.fetch_calls.append(list(tags))
        for prompt, response, stored_tags in reversed(self.entries):
            if set(tags) & set(stored_tags):
                return response
        return ""


def test_memory_based_context(monkeypatch):
    mem = DummyMemory()
    mem.log_interaction("Initial feedback", "Improve caching strategy", tags=[IMPROVEMENT_PATH])

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

    def offline_response(msgs):
        ctx = ""
        if msgs:
            content = msgs[-1]["content"]
            if "\n" in content:
                ctx = content.split("\n", 1)[1]
        return {
            "choices": [
                {"message": {"content": f"Follow-up: {ctx} now with more details"}}
            ]
        }

    monkeypatch.setattr(client, "_offline_response", offline_response)

    prompt = client.build_prompt_with_memory(
        [IMPROVEMENT_PATH], prior="What's next?", context_builder=builder
    )
    result = client.generate(
        prompt,
        context_builder=builder,
        tags=[IMPROVEMENT_PATH],
    )
    text = result.text or result.raw["choices"][0]["message"]["content"]

    assert mem.fetch_calls, "memory context was not retrieved"
    assert "Improve caching strategy" in text
