import sqlite3
import sys
import types
from datetime import datetime


class DummyEntry:
    def __init__(self, key, data, version, tags, ts=""):
        self.key = key
        self.data = data
        self.version = version
        self.tags = tags
        self.ts = ts or datetime.utcnow().isoformat()


class DummyManager:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE memory(rowid INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, data TEXT, version INTEGER, tags TEXT, ts TEXT)"
        )
        self.has_fts = False

    def log(self, entry, *, bot_id=None, info_id=None):
        self.conn.execute(
            "INSERT INTO memory(key, data, version, tags, ts) VALUES (?,?,?,?,?)",
            (entry.key, entry.data, entry.version, entry.tags, entry.ts),
        )
        self.conn.commit()

    def search(self, text, limit):
        cur = self.conn.execute(
            "SELECT key, data, version, tags, ts FROM memory WHERE data LIKE ? LIMIT ?",
            (f"%{text}%", limit),
        )
        rows = cur.fetchall()
        return [DummyEntry(*row) for row in rows]

    def store(self, key, data, tags="", *, bot_id=None, info_id=None):
        if isinstance(data, dict):
            import json

            data = json.dumps(data)
        entry = DummyEntry(key, data, 1, tags)
        self.log(entry)
        return 1


def _summarise_text(text: str, ratio: float = 0.2) -> str:
    return text[: max(1, int(len(text) * ratio))]


dummy_module = types.ModuleType("menace_memory_manager")
dummy_module.MenaceMemoryManager = DummyManager
dummy_module.MemoryEntry = DummyEntry
dummy_module._summarise_text = _summarise_text
sys.modules.setdefault("menace_memory_manager", dummy_module)

from gpt_memory import GPTMemory


def test_log_and_search():
    manager = DummyManager()
    memory = GPTMemory(manager)

    memory.log_interaction("hello", "hi", ["greeting"])
    results = memory.search_context("hello", 5)

    assert results, "expected at least one result"
    record = results[0]
    assert record["prompt"] == "hello"
    assert record["response"] == "hi"
    metadata = record["metadata"]
    assert metadata["feedback"] == []
    assert metadata["error_fixes"] == []
    assert metadata["improvement_paths"] == []


def test_summarises_long_entries():
    manager = DummyManager()
    memory = GPTMemory(manager, summary_threshold=10)

    prompt = "p" * 20
    response = "r" * 20
    memory.log_interaction(prompt, response, ["test"])

    results = memory.search_context("p" * 5, 5)
    assert results
    assert "summary" in results[0]
    assert results[0]["summary"].strip()


def test_retention_prunes_and_merges():
    manager = DummyManager()
    memory = GPTMemory(manager, max_entries=2, summary_threshold=100)

    memory.log_interaction("a1", "b1", [])
    memory.log_interaction("a2", "b2", [])
    memory.log_interaction("a3", "b3", [])  # triggers retention

    cur = manager.conn.execute("SELECT key FROM memory WHERE key LIKE 'a1%'")
    assert cur.fetchone() is None
    cur = manager.conn.execute("SELECT key FROM memory WHERE key='memory:summary'")
    assert cur.fetchone() is not None
    cur = manager.conn.execute("SELECT COUNT(*) FROM memory")
    count = cur.fetchone()[0]
    assert count <= 3
