import json
import sys
from pathlib import Path
from sqlite3 import connect

import sync_shared_db
from db_write_queue import queue_insert


def _run_sync_once(queue_dir: Path, db_path: Path, monkeypatch, max_retries: int = 1) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_shared_db.py",
            "--queue-dir",
            str(queue_dir),
            "--db-path",
            str(db_path),
            "--once",
            "--max-retries",
            str(max_retries),
        ],
    )
    sync_shared_db.main()


def test_sync_main_processes_queue(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE)"
    )
    conn.commit()
    conn.close()

    queue_dir = tmp_path / "queues"
    queue_insert("foo", {"name": "ok"}, ["name"], queue_dir)

    _run_sync_once(queue_dir, db_path, monkeypatch)

    conn = connect(db_path)  # noqa: SQL001
    rows = conn.execute("SELECT name FROM foo").fetchall()
    conn.close()
    assert rows == [("ok",)]
    assert (queue_dir / "foo_queue.jsonl").read_text() == ""


def test_sync_main_moves_failed_inserts(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE foo ("
        "id INTEGER PRIMARY KEY, "
        "name TEXT UNIQUE, "
        "other TEXT, "
        "content_hash TEXT UNIQUE)"
    )
    conn.commit()
    conn.close()

    queue_dir = tmp_path / "queues"
    queue_insert("foo", {"name": "dup", "other": "one"}, ["other"], queue_dir)
    queue_insert("foo", {"name": "dup", "other": "two"}, ["other"], queue_dir)

    _run_sync_once(queue_dir, db_path, monkeypatch, max_retries=1)

    conn = connect(db_path)  # noqa: SQL001
    rows = conn.execute("SELECT name, other FROM foo").fetchall()
    conn.close()
    assert rows == [("dup", "one")]

    queue_file = queue_dir / "foo_queue.jsonl"
    assert queue_file.read_text() == ""
    failed_file = queue_dir / "foo_queue.failed.jsonl"
    rec = json.loads(failed_file.read_text().strip())
    assert rec["record"]["data"]["other"] == "two"
    assert "UNIQUE constraint failed" in rec["error"]
