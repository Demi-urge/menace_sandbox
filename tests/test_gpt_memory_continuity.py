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

from menace_sandbox.gpt_memory import GPTMemoryManager, _summarise_text


def test_memory_continuity_and_summary(tmp_path):
    db_file = tmp_path / "memory.db"

    # Session 1: store a couple of interactions
    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("initial prompt 1", "initial response 1", tags=["topic"])
    mgr.log_interaction("initial prompt 2", "initial response 2", tags=["topic"])
    mgr.close()

    # Session 2: load existing DB and add another entry
    mgr = GPTMemoryManager(db_file)
    mgr.log_interaction("later prompt", "later response", tags=["topic"])

    past = mgr.search_context("initial")
    assert [e.prompt for e in past] == ["initial prompt 1", "initial prompt 2"]

    # Compact keeping only the most recent 'topic' entry
    expected_summary = _summarise_text(
        "initial prompt 1 initial response 1\ninitial prompt 2 initial response 2"
    )
    mgr.compact({"topic": 1})
    mgr.close()

    # Session 3: verify persistence of latest entry and summary
    mgr = GPTMemoryManager(db_file)

    recent = mgr.search_context("later")
    assert [e.prompt for e in recent] == ["later prompt"]

    summary_entries = mgr.search_context("summary:topic")
    assert len(summary_entries) == 1
    summary = summary_entries[0]
    assert summary.response == expected_summary
    assert set(summary.tags) == {"topic", "summary"}

    # Old context should now be represented by the summary
    old = mgr.search_context("initial")
    assert [e.prompt for e in old] == ["summary:topic"]

    mgr.close()
