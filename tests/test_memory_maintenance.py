import os
import time
from pathlib import Path

from menace_sandbox.gpt_memory import GPTMemoryManager
from memory_maintenance import MemoryMaintenance


def test_memory_maintenance_compacts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv('GPT_MEMORY_RETENTION', 'foo=1')
    db = tmp_path / 'mem.db'
    mgr = GPTMemoryManager(str(db))
    called = []

    def fake_compact(rules):
        called.append(rules.copy())
        return 0

    monkeypatch.setattr(mgr, 'compact', fake_compact)
    maint = MemoryMaintenance(mgr, interval=0.1)
    maint.start()
    time.sleep(0.25)
    maint.stop()
    assert called and called[0] == {'foo': 1}


def test_memory_maintenance_prunes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv('GPT_MEMORY_MAX_ROWS', '2')
    db = tmp_path / 'mem.db'
    mgr = GPTMemoryManager(str(db))
    called: list[int] = []

    def fake_prune(limit: int) -> int:
        called.append(limit)
        return 0

    monkeypatch.setattr(mgr, 'prune_old_entries', fake_prune)
    maint = MemoryMaintenance(mgr, interval=0.1)
    maint.start()
    time.sleep(0.25)
    maint.stop()
    assert called and called[0] == 2
