import pytest

from menace_sandbox.gpt_memory import GPTMemoryManager, FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH
from gpt_knowledge_service import GPTKnowledgeService


def test_memory_and_insights_persist(tmp_path):
    db_file = tmp_path / "memory.db"
    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("fb prompt", "fb response", tags=[FEEDBACK])
    mgr.log_interaction("err prompt", "err response", tags=[ERROR_FIX])
    mgr.log_interaction("imp prompt", "imp response", tags=[IMPROVEMENT_PATH])

    service = GPTKnowledgeService(mgr, max_per_tag=5)
    summaries = {
        FEEDBACK: service.get_recent_insights(FEEDBACK),
        ERROR_FIX: service.get_recent_insights(ERROR_FIX),
        IMPROVEMENT_PATH: service.get_recent_insights(IMPROVEMENT_PATH),
    }

    mgr.close()

    mgr2 = GPTMemoryManager(db_file)
    assert [e.response for e in mgr2.search_context("fb prompt", tags=[FEEDBACK])] == ["fb response"]
    assert [e.response for e in mgr2.search_context("err prompt", tags=[ERROR_FIX])] == ["err response"]
    assert [e.response for e in mgr2.search_context("imp prompt", tags=[IMPROVEMENT_PATH])] == ["imp response"]

    service2 = GPTKnowledgeService(mgr2, max_per_tag=5)
    for tag, summary in summaries.items():
        assert service2.get_recent_insights(tag) == summary
    mgr2.close()
