import types
import sys

# Stub sentence_transformers to avoid heavy import
stub_st = types.ModuleType("sentence_transformers")
class DummyModel:
    def encode(self, text):
        text = text.lower()
        if "password" in text or "credential" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]
stub_st.SentenceTransformer = DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)

from menace_sandbox.gpt_memory import GPTMemoryManager
from knowledge_retriever import get_feedback
from log_tags import FEEDBACK


def test_semantic_search_recovers_related_prompt():
    embedder = DummyModel()
    mgr = GPTMemoryManager(db_path=":memory:", embedder=embedder)
    mgr.log_interaction("reset my password", "done", [FEEDBACK])

    exact = mgr.search_context("forgot credentials", use_embeddings=False)
    assert exact == []

    semantic = mgr.search_context("forgot credentials", use_embeddings=True)
    assert len(semantic) == 1
    assert semantic[0].prompt == "reset my password"


def test_knowledge_retriever_uses_embeddings_by_default():
    embedder = DummyModel()
    mgr = GPTMemoryManager(db_path=":memory:", embedder=embedder)
    mgr.log_interaction("reset my password", "done", [FEEDBACK])

    res = get_feedback(mgr, "lost credentials")
    assert len(res) == 1
    assert res[0].prompt == "reset my password"

    no_embed = get_feedback(mgr, "lost credentials", use_embeddings=False)
    assert no_embed == []
