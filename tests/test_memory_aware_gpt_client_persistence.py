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
from gpt_knowledge_service import GPTKnowledgeService
from memory_aware_gpt_client import ask_with_memory
from log_tags import FEEDBACK


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

    client.next_response = "Great success"
    ask_with_memory(
        client,
        "lost credentials",
        "reset my password",
        memory=mgr,
        tags=[FEEDBACK],
    )

    client.next_response = "This fixes the error."
    ask_with_memory(
        client,
        "lost credentials",
        "credential bug encountered",
        memory=mgr,
        tags=["bogus"],
    )

    client.next_response = "An improvement is to apply a patch."
    ask_with_memory(
        client,
        "lost credentials",
        "any improvement suggestions?",
        memory=mgr,
    )
    mgr.close()

    mgr2 = GPTMemoryManager(db_path=str(db), embedder=embedder)

    assert mgr2.search_context(
        "lost credentials", tags=[FEEDBACK], use_embeddings=False
    ) == []

    fb = get_feedback(mgr2, "lost credentials")
    fixes = get_error_fixes(mgr2, "lost credentials")
    improvs = get_improvement_paths(mgr2, "lost credentials")

    assert [e.response for e in fb] == ["Great success"]
    assert [e.response for e in fixes] == ["This fixes the error."]
    assert [e.response for e in improvs] == ["An improvement is to apply a patch."]

    client2 = DummyClient()
    client2.next_response = "final"
    ask_with_memory(client2, "lost credentials", "what next?", memory=mgr2)
    sent_prompt = client2.messages[0]["content"]
    assert "Great success" in sent_prompt
    assert "This fixes the error." in sent_prompt
    assert "An improvement is to apply a patch." in sent_prompt

    assert mgr2.search_context("what next?", limit=1)

    service = GPTKnowledgeService(mgr2, max_per_tag=5)
    assert recent_feedback(service)
    assert recent_error_fix(service)
    assert recent_improvement_path(service)
    mgr2.close()

