import json
import multiprocessing
from pathlib import Path
from sqlite3 import Connection, connect

import pytest

from db_write_queue import append_record
from sync_shared_db import _sync_once, process_queue_file


def _worker(n: int, queue_dir: str) -> None:
    """Helper for concurrent write test."""
    append_record("example", {"n": n}, "m1", Path(queue_dir))


def _init_db(path: Path) -> Connection:
    conn = connect(path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE example (id INTEGER PRIMARY KEY, value TEXT, content_hash TEXT UNIQUE)"
    )
    return conn


def test_sync_processes_queue(tmp_path: Path) -> None:
    """Queued records are written to the shared database."""
    queue_dir = tmp_path / "queue"
    db_path = tmp_path / "shared.db"
    conn = _init_db(db_path)

    append_record("example", {"value": "a"}, "m1", queue_dir)
    append_record("example", {"value": "b"}, "m1", queue_dir)

    stats = _sync_once(queue_dir, conn)
    assert stats.processed == 2

    rows = conn.execute("SELECT value FROM example ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["a", "b"]
    path = queue_dir / "m1.jsonl"
    assert not path.exists()


def test_sync_removes_empty_file_backup_preserved(tmp_path: Path) -> None:
    """Processed queue file is removed and logged to backup."""
    queue_dir = tmp_path / "queue"
    db_path = tmp_path / "shared.db"
    conn = _init_db(db_path)

    append_record("example", {"value": "c"}, "m1", queue_dir)

    _sync_once(queue_dir, conn)

    path = queue_dir / "m1.jsonl"
    backup = queue_dir / "queue.log.bak"
    assert not path.exists()
    assert backup.exists()
    contents = backup.read_text(encoding="utf-8")
    assert "\"value\": \"c\"" in contents


def test_sync_crash_leaves_unprocessed_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-processing leaves queue file intact for later retry."""
    queue_dir = tmp_path / "queue"
    db_path = tmp_path / "shared.db"
    conn = _init_db(db_path)

    append_record("example", {"value": "a"}, "m1", queue_dir)
    append_record("example", {"value": "b"}, "m1", queue_dir)
    path = queue_dir / "m1.jsonl"

    call_count = {"n": 0}
    orig = process_queue_file.__globals__["insert_if_unique"]

    def boom(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise KeyboardInterrupt("boom")
        return orig(*args, **kwargs)

    monkeypatch.setitem(process_queue_file.__globals__, "insert_if_unique", boom)

    with pytest.raises(KeyboardInterrupt):
        process_queue_file(path, conn=conn)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    rows = conn.execute("SELECT value FROM example ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["a"]


def test_sync_deduplicates_hashes(tmp_path: Path) -> None:
    """Duplicate records are skipped based on content hash."""
    queue_dir = tmp_path / "queue"
    db_path = tmp_path / "shared.db"
    conn = _init_db(db_path)

    append_record("example", {"value": "x"}, "m1", queue_dir)
    append_record("example", {"value": "x"}, "m1", queue_dir)

    stats = _sync_once(queue_dir, conn)
    assert stats.processed == 1
    assert stats.duplicates == 1

    rows = conn.execute("SELECT value FROM example").fetchall()
    assert rows == [("x",)]


def test_concurrent_writes_use_file_locks(tmp_path: Path) -> None:
    """Multiple processes can append without interleaving writes."""
    queue_dir = tmp_path / "queue"
    path = queue_dir / "m1.jsonl"

    processes = [
        multiprocessing.Process(target=_worker, args=(i, str(queue_dir))) for i in range(10)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    payloads = [json.loads(line)["data"]["n"] for line in lines]
    assert set(payloads) == set(range(10))
