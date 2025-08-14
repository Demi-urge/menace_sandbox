import sys
import types

# Stub menace_memory_manager so gpt_memory can import it without heavy deps.
stub_mm = types.ModuleType("menace_memory_manager")

class _Entry:
    def __init__(self, key, data, version=1, tags="", ts="now"):
        self.key = key
        self.data = data
        self.version = version
        self.tags = tags
        self.ts = ts

class _StubManager:
    def __init__(self):
        self.items = []

    def store(self, key, data, tags="", bot_id=None, info_id=None):
        self.items.append(_Entry(key, data, tags=tags))
        return len(self.items)

    def search(self, text, limit=20):
        res = []
        for e in self.items:
            if text in e.data or text in e.key:
                res.append(e)
        return res[:limit]

stub_mm.MenaceMemoryManager = _StubManager
stub_mm._summarise_text = lambda text, ratio=0.2: text[: max(1, int(len(text) * ratio))]
sys.modules.setdefault("menace_memory_manager", stub_mm)

# Stub sentence_transformers to keep import lightweight.
stub_st = types.ModuleType("sentence_transformers")
class _DummyModel:
    def encode(self, text):
        return [0.0]
stub_st.SentenceTransformer = _DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)

from gpt_memory import GPTMemory, GPTMemoryManager


def test_store_and_retrieve_with_tags():
    mem = GPTMemory()
    mem.store("fix login bug", "patched", ["bugfix"])
    mem.store("add feature", "improve UX", ["improvement"])

    bug = mem.retrieve("fix", tags=["bugfix"])
    assert len(bug) == 1
    assert bug[0].prompt == "fix login bug"
    assert bug[0].tags == ["bugfix"]

    none = mem.retrieve("fix", tags=["improvement"])
    assert none == []

    all_entries = mem.retrieve("add")
    assert len(all_entries) == 1
    assert all_entries[0].tags == ["improvement"]


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
        mgr.log_interaction(f"p{i}", f"r{i}", tags=["bugfix"])

    removed = mgr.compact({"bugfix": 2})
    assert removed == 3

    cur = mgr.conn.execute("SELECT prompt, tags FROM interactions ORDER BY id")
    rows = cur.fetchall()
    assert len(rows) == 3  # two recent entries + one summary

    prompts = [p for p, t in rows if "summary" not in t]
    assert prompts == ["p3", "p4"]

    summaries = [p for p, t in rows if "summary" in t]
    assert summaries  # summary entry exists
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
