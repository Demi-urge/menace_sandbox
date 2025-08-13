import sys
import types

# Stub menace_memory_manager before importing gpt_memory
class DummyEntry:
    def __init__(self, key, data, version, tags, ts=""):
        self.key = key
        self.data = data
        self.version = version
        self.tags = tags
        self.ts = ts


class DummyManager:
    def __init__(self):
        self.entries = []

    def log(self, entry, *, bot_id=None, info_id=None):
        self.entries.append(entry)

    def search(self, text, limit):
        return [e for e in self.entries if text in e.data][:limit]


dummy_module = types.ModuleType("menace_memory_manager")
dummy_module.MenaceMemoryManager = DummyManager
dummy_module.MemoryEntry = DummyEntry
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
