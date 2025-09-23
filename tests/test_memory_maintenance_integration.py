import sqlite3
import time
from pathlib import Path

from menace_sandbox.gpt_memory import GPTMemoryManager
from gpt_knowledge_service import GPTKnowledgeService
from memory_maintenance import MemoryMaintenance


def test_compaction_summarises_and_preserves_insights(monkeypatch, tmp_path: Path):
    original_connect = sqlite3.connect

    def patched_connect(*args, **kwargs):
        kwargs.setdefault("check_same_thread", False)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", patched_connect)

    db = tmp_path / "mem.db"
    mgr = GPTMemoryManager(str(db))

    for i in range(3):
        mgr.log_interaction(f"p{i}", f"r{i}", tags=["foo"])

    service = GPTKnowledgeService(mgr)
    maint = MemoryMaintenance(
        mgr, interval=0.1, retention={"foo": 1}, knowledge_service=service
    )
    maint.start()
    time.sleep(0.25)
    maint.stop()

    entries = mgr.retrieve("", tags=["foo"], limit=10)
    prompts = {e.prompt for e in entries}
    assert "p0" not in prompts and "p1" not in prompts
    assert any(p.startswith("summary:foo") for p in prompts)

    insight = service.get_recent_insights("foo")
    assert insight
