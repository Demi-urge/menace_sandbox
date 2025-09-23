import pytest

from menace_sandbox.gpt_memory import GPTMemoryManager, FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX
from gpt_knowledge_service import GPTKnowledgeService


def test_insights_persist_across_restart(tmp_path):
    db_file = tmp_path / "memory.db"
    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("feedback prompt", "feedback response", tags=[FEEDBACK])
    mgr.log_interaction("improvement prompt", "improvement response", tags=[IMPROVEMENT_PATH])
    mgr.log_interaction("error prompt", "error response", tags=[ERROR_FIX])

    service = GPTKnowledgeService(mgr, max_per_tag=5)
    service.update_insights()

    summaries = {
        FEEDBACK: service.get_recent_insights(FEEDBACK),
        IMPROVEMENT_PATH: service.get_recent_insights(IMPROVEMENT_PATH),
        ERROR_FIX: service.get_recent_insights(ERROR_FIX),
    }
    for tag, summary in summaries.items():
        assert summary, f"expected non-empty summary for {tag}"

    mgr.close()

    mgr2 = GPTMemoryManager(db_file)
    service2 = GPTKnowledgeService(mgr2, max_per_tag=5)
    for tag, summary in summaries.items():
        assert service2.get_recent_insights(tag) == summary
    mgr2.close()
