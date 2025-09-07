import types

from gpt_memory import GPTMemoryManager
from knowledge_retriever import (
    get_feedback,
    get_error_fixes,
    get_improvement_paths,
    recent_feedback,
    recent_error_fix,
    recent_improvement_path,
)
from types import SimpleNamespace

from memory_aware_gpt_client import ask_with_memory
from local_knowledge_module import LocalKnowledgeModule
from log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH


class DummyModel:
    def encode(self, text):
        text = text.lower()
        if "password" in text or "credential" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


class DummyClient:
    def __init__(self):
        self.messages = []
        self.next_response = ""

    def ask(self, messages, **kwargs):
        self.messages = messages
        return {"choices": [{"message": {"content": self.next_response}}]}


def test_memory_aware_client_persists_across_runs(tmp_path):
    db = tmp_path / "mem.db"
    embedder = DummyModel()

    client = DummyClient()
    mgr = GPTMemoryManager(db_path=str(db), embedder=embedder)
    module = LocalKnowledgeModule(manager=mgr)
    builder = SimpleNamespace(build=lambda *a, **k: "")

    client.next_response = "Great success"
    ask_with_memory(
        client,
        "auth.reset_password",
        "reset my password",
        memory=module,
        context_builder=builder,
        tags=[FEEDBACK],
    )

    client.next_response = "This fixes the error."
    ask_with_memory(
        client,
        "auth.reset_password",
        "credential bug encountered",
        memory=module,
        context_builder=builder,
        tags=[ERROR_FIX],
    )

    client.next_response = "An improvement is to apply a patch."
    ask_with_memory(
        client,
        "auth.reset_password",
        "any improvement suggestions?",
        memory=module,
        context_builder=builder,
        tags=[IMPROVEMENT_PATH],
    )
    mgr.close()

    mgr2 = GPTMemoryManager(db_path=str(db), embedder=embedder)
    module2 = LocalKnowledgeModule(manager=mgr2)

    assert mgr2.search_context(
        "auth.reset_password", tags=[FEEDBACK], use_embeddings=False
    ) == []

    fb = [e.response for e in get_feedback(mgr2, "auth.reset_password") if "insight" not in e.tags]
    fixes = [
        e.response
        for e in get_error_fixes(mgr2, "auth.reset_password")
        if "insight" not in e.tags
    ]
    improvs = [
        e.response
        for e in get_improvement_paths(mgr2, "auth.reset_password")
        if "insight" not in e.tags
    ]

    assert fb == ["Great success"]
    assert fixes == ["This fixes the error."]
    assert improvs == ["An improvement is to apply a patch."]

    client2 = DummyClient()
    client2.next_response = "final"
    ask_with_memory(
        client2,
        "auth.reset_password",
        "what next?",
        memory=module2,
        context_builder=builder,
    )
    sent_prompt = client2.messages[0]["content"]
    assert "Great success" in sent_prompt
    assert "This fixes the error." in sent_prompt
    assert "An improvement is to apply a patch." in sent_prompt

    assert mgr2.search_context("what next?", limit=1)

    service = module2.knowledge
    assert recent_feedback(service)
    assert recent_error_fix(service)
    assert recent_improvement_path(service)
    mgr2.close()

