"""Tests for GPT memory persistence across sessions."""

import sys
import types


# Stub heavy dependencies so the import is lightweight during tests.
stub_mm = types.ModuleType("menace_memory_manager")
stub_mm._summarise_text = lambda text, ratio=0.2: text[: max(1, int(len(text) * ratio))]
sys.modules.setdefault("menace_memory_manager", stub_mm)

stub_st = types.ModuleType("sentence_transformers")


class _DummyModel:
    def encode(self, text):
        return [0.0]


stub_st.SentenceTransformer = _DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)


from gpt_memory import GPTMemoryManager


def test_logging_and_retrieval_across_sessions(tmp_path):
    """Interactions logged in one manager are retrievable in a new session."""

    db_path = tmp_path / "memory.db"

    first = GPTMemoryManager(db_path)
    first.log_interaction("hello", "hi", ["greeting"])
    first.close()

    second = GPTMemoryManager(db_path)
    results = second.search_context("hello", limit=5)
    assert results, "expected stored interaction to be retrieved"
    entry = results[0]
    assert entry.prompt == "hello"
    assert entry.response == "hi"
    assert entry.tags == ["greeting"]
