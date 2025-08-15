import os
import time
from pathlib import Path

from gpt_memory import GPTMemoryManager
from memory_maintenance import MemoryMaintenance


def test_memory_maintenance_compacts(monkeypatch, tmp_path: Path):
    os.environ['GPT_MEMORY_RETENTION'] = 'foo=1'
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
