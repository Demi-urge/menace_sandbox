import sys
import types

sys.modules.setdefault(
    "menace_sandbox.database_manager",
    types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: []),
)
sys.modules.setdefault(
    "menace_sandbox.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object)
)
sys.modules.setdefault(
    "menace_sandbox.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
def _log_with_tags(mem, prompt, response, tags):
    if hasattr(mem, "log_interaction"):
        mem.log_interaction(prompt, response, tags)

sys.modules[
    "menace_sandbox.memory_logging"
] = types.SimpleNamespace(log_with_tags=_log_with_tags)
sys.modules.setdefault(
    "menace_sandbox.memory_aware_gpt_client", types.SimpleNamespace(ask_with_memory=lambda *a, **k: {})
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

import menace_sandbox.chatgpt_idea_bot as cib
cib.log_with_tags = _log_with_tags
from prompt_types import Prompt


class DummyResp:
    status_code = 200

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": "ok"}}]}


class DummySession:
    def __init__(self, record):
        self.record = record

    def post(self, url, headers=None, json=None, timeout=0):
        self.record["messages"] = json["messages"]
        return DummyResp()


class DummyKnowledge:
    def __init__(self, record):
        self.record = record
        self.logged = None

    class Entry:
        def __init__(self, prompt: str, response: str) -> None:
            self.prompt = prompt
            self.response = response

    def get_similar_entries(self, query, limit=5, use_embeddings=False):
        self.record["query"] = query
        return [(1.0, self.Entry("p1", "r1"))]

    def log_interaction(self, prompt, response, tags):
        self.logged = (prompt, response, list(tags))


def test_ask_injects_context_and_logs(monkeypatch):
    record = {}
    # stub requests module so ChatGPTClient doesn't require real dependency
    cib.requests = type("R", (), {"Timeout": Exception, "RequestException": Exception})

    session = DummySession(record)

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

    client = cib.ChatGPTClient(api_key="key", session=session, context_builder=DummyBuilder())
    monkeypatch.setattr(cib, "govern_retrieval", lambda *a, **k: ({}, None))
    monkeypatch.setattr(cib, "redact", lambda x: x)
    knowledge = DummyKnowledge(record)

    resp = client.ask(
        [{"role": "user", "content": "hello"}],
        knowledge=knowledge,
        use_memory=True,
        relevance_threshold=0.5,
        max_summary_length=20,
        tags=["t"],
    )

    assert record["query"] == "hello"
    msgs = record["messages"]
    assert msgs[0]["role"] == "system"
    assert "Prompt: p1" in msgs[0]["content"]
    assert msgs[1]["content"] == "hello"
    assert knowledge.logged == ("hello", "ok", ["chatgpt_idea_bot.generate", "t"])
    assert resp["choices"][0]["message"]["content"] == "ok"


def test_prompt_equivalence(monkeypatch):
    record = {}
    cib.requests = type("R", (), {"Timeout": Exception, "RequestException": Exception})
    session = DummySession(record)

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

    client = cib.ChatGPTClient(
        api_key="key", session=session, context_builder=DummyBuilder(), gpt_memory=None
    )
    prompt_obj = Prompt(
        system="sys",
        user="hello",
        examples=["ex"],
        metadata={"m": 1},
    )
    resp = client.ask(prompt_obj, use_memory=False)

    expected = cib.prepend_payment_notice(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ex"},
            {"role": "user", "content": "hello", "metadata": {"m": 1}},
        ]
    )
    assert record["messages"] == expected
    assert resp["choices"][0]["message"]["content"] == "ok"
