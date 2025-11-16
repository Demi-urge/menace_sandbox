import sys
import types

import types
import sys

# Stub sentence_transformers to keep import lightweight.
stub_st = types.ModuleType("sentence_transformers")
class _DummyModel:
    def encode(self, text):
        return [0.0]
stub_st.SentenceTransformer = _DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)

from menace_sandbox.gpt_memory import GPTMemoryManager


def test_store_and_retrieve_with_tags():
    mem = GPTMemoryManager(db_path=":memory:")
    mem.log_interaction("fix login bug", "patched", ["error_fix"])
    mem.log_interaction("add feature", "improve UX", ["improvement_path"])

    bug = mem.retrieve("fix", tags=["error_fix"])
    assert len(bug) == 1
    assert bug[0].prompt == "fix login bug"
    assert bug[0].tags == ["error_fix"]

    none = mem.retrieve("fix", tags=["improvement_path"])
    assert none == []

    all_entries = mem.retrieve("add")
    assert len(all_entries) == 1
    assert all_entries[0].tags == ["improvement_path"]
    mem.close()


def test_summarize_and_prune_removes_old_entries():
    mem = GPTMemoryManager(db_path=":memory:")
    for i in range(3):
        mem.log_interaction(f"p{i}", f"r{i}", tags=["general"])

    mem.compact({"general": 1})

    # Oldest entries should be pruned; newest remains
    entries = [e for e in mem.retrieve("p0", tags=["general"]) if "summary" not in e.tags]
    assert entries == []
    remaining = [e for e in mem.retrieve("p2", tags=["general"]) if "summary" not in e.tags]
    assert len(remaining) == 1

    # Summary entry should be stored
    summaries = [e for e in mem.search_context("", tags=["general"]) if "summary" in e.tags]
    assert len(summaries) == 1
    assert summaries[0].response
    mem.close()


def test_manager_get_similar_entries_text():
    mgr = GPTMemoryManager(db_path=":memory:")
    mgr.log_interaction("hello world", "hi")
    mgr.log_interaction("weather today", "sunny")
    results = mgr.get_similar_entries("weather", limit=2, use_embeddings=False)
    assert len(results) == 1
    score, entry = results[0]
    assert entry.prompt == "weather today"
    assert score > 0
    mgr.close()


def test_compaction_prunes_old_entries():
    mgr = GPTMemoryManager(db_path=":memory:")
    for i in range(5):
        mgr.log_interaction(f"p{i}", f"r{i}", tags=["error_fix"])

    removed = mgr.compact({"error_fix": 2})
    assert removed == 3

    cur = mgr.conn.execute("SELECT prompt, tags FROM interactions ORDER BY id")
    rows = cur.fetchall()
    assert len(rows) == 3  # two recent entries + one summary

    prompts = [p for p, t in rows if "summary" not in t]
    assert prompts == ["p3", "p4"]

    summaries = [p for p, t in rows if "summary" in t]
    assert summaries  # summary entry exists
    mgr.close()


def test_prune_old_entries_limits_per_tag():
    mgr = GPTMemoryManager(db_path=":memory:")
    for i in range(4):
        mgr.log_interaction(f"p{i}", f"r{i}", tags=["foo"])

    removed = mgr.prune_old_entries(2)
    assert removed == 2

    cur = mgr.conn.execute("SELECT prompt, tags FROM interactions ORDER BY id")
    rows = cur.fetchall()
    assert len(rows) == 3
    prompts = [p for p, t in rows if "summary" not in t]
    assert prompts == ["p2", "p3"]
    mgr.close()


def test_deduplication_skips_duplicate_entries():
    mgr = GPTMemoryManager(db_path=":memory:")
    mgr.log_interaction("same", "resp")
    mgr.log_interaction("same", "resp")
    cur = mgr.conn.execute("SELECT COUNT(*) FROM interactions")
    assert cur.fetchone()[0] == 1
    mgr.close()


def test_persistence_across_sessions(tmp_path):
    db_file = tmp_path / "memory.db"

    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("first question", "first answer", tags=["init"])
    mgr.close()

    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("second question", "second answer")
    mgr.close()

    mgr = GPTMemoryManager(db_file)
    first = mgr.search_context("first")
    second = mgr.search_context("second")
    assert [e.response for e in first] == ["first answer"]
    assert [e.response for e in second] == ["second answer"]
    mgr.close()
