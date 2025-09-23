import pytest

from menace_sandbox.gpt_memory import GPTMemoryManager, INSIGHT, ERROR_FIX
from gpt_knowledge_service import GPTKnowledgeService


def test_service_generates_insights():
    mgr = GPTMemoryManager(db_path=":memory:")
    # populate with a couple of entries under the same tag
    mgr.log_interaction("prompt1", "fixed bug", tags=[ERROR_FIX])
    mgr.log_interaction("prompt2", "another fix", tags=[ERROR_FIX])

    service = GPTKnowledgeService(mgr, max_per_tag=5)

    # ensure a summary entry was created with INSIGHT tag
    entries = mgr.retrieve("", tags=[INSIGHT, ERROR_FIX], limit=1)
    assert entries, "summary should be stored"  # at least one insight
    insight_text = entries[0].response
    assert insight_text  # non-empty summary

    # retrieval helper should surface the same summary
    assert service.get_recent_insights(ERROR_FIX) == insight_text
