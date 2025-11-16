import pytest

from menace_sandbox.gpt_memory import GPTMemoryManager, FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX
from knowledge_retriever import get_feedback, get_improvement_paths, get_error_fixes


def test_retrieval_after_restart(tmp_path):
    db_file = tmp_path / "memory.db"
    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("fb prompt", "fb response", tags=[FEEDBACK])
    mgr.log_interaction("imp prompt", "imp response", tags=[IMPROVEMENT_PATH])
    mgr.log_interaction("err prompt", "err response", tags=[ERROR_FIX])
    mgr.close()

    mgr2 = GPTMemoryManager(db_file)
    try:
        fb_entries = get_feedback(mgr2, "fb prompt")
        imp_entries = get_improvement_paths(mgr2, "imp prompt")
        err_entries = get_error_fixes(mgr2, "err prompt")

        assert [e.response for e in fb_entries] == ["fb response"]
        assert [e.response for e in imp_entries] == ["imp response"]
        assert [e.response for e in err_entries] == ["err response"]
    finally:
        mgr2.close()
