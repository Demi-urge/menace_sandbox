import pytest

from menace_sandbox.gpt_memory import GPTMemoryManager, FEEDBACK
from knowledge_retriever import get_feedback


def test_search_context_applies_governance(tmp_path):
    db = tmp_path / "mem.db"
    mgr = GPTMemoryManager(db)
    mgr.log_interaction("p", "eval('data')", tags=[FEEDBACK])
    try:
        hits = mgr.search_context("p")
        assert hits and hits[0].metadata
        assert hits[0].metadata.get("semantic_alerts")
    finally:
        mgr.close()


def test_knowledge_retriever_filters_disallowed_licenses(tmp_path):
    db = tmp_path / "mem.db"
    mgr = GPTMemoryManager(db)
    mgr.log_interaction("ok", "eval('data')", tags=[FEEDBACK])
    mgr.log_interaction("bad", "GNU General Public License", tags=[FEEDBACK])
    try:
        entries = get_feedback(mgr, "")
        assert len(entries) == 1
        assert entries[0].metadata.get("semantic_alerts")
    finally:
        mgr.close()
