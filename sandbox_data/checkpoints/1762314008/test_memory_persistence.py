import pytest

from menace_sandbox.gpt_memory import GPTMemoryManager, FEEDBACK, ERROR_FIX, _summarise_text
from knowledge_retriever import get_feedback
from local_knowledge_module import LocalKnowledgeModule


def test_memory_persists_across_restart(tmp_path):
    db_file = tmp_path / "memory.db"
    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("fb prompt", "fb response", tags=[FEEDBACK])
    mgr.log_interaction("other prompt", "other response", tags=[ERROR_FIX])

    module = LocalKnowledgeModule(manager=mgr)
    insight_before = module.get_insights(FEEDBACK)
    # sanity check that insight summarises the feedback interaction
    assert insight_before == _summarise_text("fb prompt fb response")
    module.memory.close()

    mgr2 = GPTMemoryManager(db_file)

    fb_entries = get_feedback(mgr2, "fb prompt", use_embeddings=True)
    assert [e.response for e in fb_entries] == ["fb response"]

    # tag filtering - non feedback entries should not be returned
    assert get_feedback(mgr2, "other prompt") == []

    # missing embeddings edge case - requesting with use_embeddings when none exist
    assert get_feedback(mgr2, "unseen", use_embeddings=True) == []

    module2 = LocalKnowledgeModule(manager=mgr2)
    assert module2.get_insights(FEEDBACK) == insight_before
    module2.memory.close()
