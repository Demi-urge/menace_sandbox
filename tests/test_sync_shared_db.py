import json
from sqlite3 import connect

import pytest

from db_write_queue import append_record
from sync_shared_db import _sync_once


@pytest.fixture
def queue_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sqlite_conn():
    conn = connect(":memory:")  # noqa: SQL001
    conn.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "value TEXT NOT NULL, content_hash TEXT UNIQUE)"
    )
    yield conn
    conn.close()


def test_sync_once_processes_records(queue_dir, sqlite_conn):
    append_record("test", {"value": "a"}, "m1", queue_dir)
    append_record("test", {"value": "a"}, "m1", queue_dir)
    append_record("test", {"value": None}, "m1", queue_dir)

    stats = _sync_once(queue_dir, sqlite_conn)
    assert stats.processed == 1
    assert stats.duplicates == 1
    assert stats.failures == 1

    rows = sqlite_conn.execute("SELECT value FROM test").fetchall()
    assert rows == [("a",)]

    queue_file = queue_dir / "m1.jsonl"
    assert queue_file.read_text() == ""

    failed_path = queue_dir / "queue.failed.jsonl"
    lines = failed_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["record"]["data"]["value"] is None
    assert "error" in entry
