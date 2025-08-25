import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

import sync_shared_db
from db_write_queue import queue_insert
from queue_cleanup import cleanup


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE)"
    )
    conn.commit()
    yield conn, db_path
    conn.close()


def _names(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name FROM foo ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]


def test_process_queue_moves_errors(tmp_path, db):
    conn, db_path = db
    queue_dir = tmp_path
    queue_insert("foo", {"name": "ok"}, ["name"], queue_dir)
    queue_file = queue_dir / "foo_queue.jsonl"

    bad = {"table": "missing", "op": "insert", "data": {"name": "bad"}}
    with queue_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(bad) + "\n")

    sync_shared_db.process_queue_file(queue_file, conn=conn, max_retries=1)

    assert _names(db_path) == ["ok"]
    assert queue_file.read_text() == ""
    failed = queue_dir / "foo_queue.failed.jsonl"
    assert failed.exists()
    rec = json.loads(failed.read_text().strip())
    assert rec["record"]["data"]["name"] == "bad"
    assert "error" in rec


def test_malformed_lines_moved_to_error(tmp_path, db):
    conn, db_path = db
    queue_dir = tmp_path
    queue_insert("foo", {"name": "ok"}, ["name"], queue_dir)
    queue_file = queue_dir / "foo_queue.jsonl"

    with queue_file.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")

    sync_shared_db.process_queue_file(queue_file, conn=conn, max_retries=1)

    assert _names(db_path) == ["ok"]
    assert queue_file.read_text() == ""
    err = queue_dir / "foo_queue.error.jsonl"
    assert err.exists()
    assert "not-json" in err.read_text()


def test_duplicate_hashes_committed_once(tmp_path, db):
    conn, db_path = db
    queue_dir = tmp_path
    queue_insert("foo", {"name": "dup"}, ["name"], queue_dir)
    queue_insert("foo", {"name": "dup"}, ["name"], queue_dir)
    queue_file = queue_dir / "foo_queue.jsonl"

    sync_shared_db.process_queue_file(queue_file, conn=conn, max_retries=1)

    assert _names(db_path) == ["dup"]
    assert queue_file.read_text() == ""
    assert not (queue_dir / "foo_queue.error.jsonl").exists()


def test_cleanup_removes_old_files(tmp_path):
    queue_dir = tmp_path
    old_tmp = queue_dir / "foo_queue.jsonl.tmp"
    old_fail = queue_dir / "foo_queue.failed.jsonl"
    recent = queue_dir / "bar_queue.failed.jsonl"
    for p in (old_tmp, old_fail, recent):
        p.write_text("x")
    old_ts = time.time() - 10 * 86400
    os.utime(old_tmp, (old_ts, old_ts))
    os.utime(old_fail, (old_ts, old_ts))

    cleanup(queue_dir, days=7)
    assert not old_tmp.exists()
    assert not old_fail.exists()
    assert recent.exists()

