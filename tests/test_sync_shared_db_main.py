import json
import sys
from pathlib import Path
from sqlite3 import connect

import sync_shared_db


def _run_sync_once(queue_dir: Path, db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_shared_db.py",  # path-ignore
            "--queue-dir",
            str(queue_dir),
            "--db-path",
            str(db_path),
            "--once",
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
    queue_dir.mkdir()
    queue_file = queue_dir / "m1.jsonl"
    entry = {"table": "foo", "record": {"name": "ok"}, "menace_id": "m1"}
    queue_file.write_text(json.dumps(entry) + "\n")

    _run_sync_once(queue_dir, db_path, monkeypatch)

    conn = connect(db_path)  # noqa: SQL001
    rows = conn.execute("SELECT name FROM foo").fetchall()
    conn.close()
    assert rows == [("ok",)]
    assert queue_file.read_text() == ""


def test_sync_main_moves_failed_inserts(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT UNIQUE, "
        "other TEXT, content_hash TEXT UNIQUE)"
    )
    conn.commit()
    conn.close()

    queue_dir = tmp_path / "queues"
    queue_dir.mkdir()
    queue_file = queue_dir / "m1.jsonl"
    one = {"table": "foo", "record": {"name": "dup", "other": "one"}, "menace_id": "m1"}
    two = {"table": "foo", "record": {"name": "dup", "other": "two"}, "menace_id": "m1"}
    with queue_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(one) + "\n")
        fh.write(json.dumps(two) + "\n")

    _run_sync_once(queue_dir, db_path, monkeypatch)

    conn = connect(db_path)  # noqa: SQL001
    rows = conn.execute("SELECT name, other FROM foo").fetchall()
    conn.close()
    assert rows == [("dup", "one")]

    assert queue_file.read_text() == ""
    failed_file = queue_dir / "queue.failed.jsonl"
    rec = json.loads(failed_file.read_text().strip())
    assert rec["record"]["record"]["other"] == "two"
    assert "UNIQUE constraint failed" in rec["error"]


def test_replay_failed(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE)"
    )
    conn.commit()
    conn.close()

    queue_dir = tmp_path / "queues"
    queue_dir.mkdir()
    failed_file = queue_dir / "queue.failed.jsonl"
    record = {"table": "foo", "record": {"name": "ok"}, "menace_id": "m1"}
    failed_entry = {"record": record, "error": "temporary"}
    failed_file.write_text(json.dumps(failed_entry) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_shared_db.py",  # path-ignore
            "--queue-dir",
            str(queue_dir),
            "--db-path",
            str(db_path),
            "--once",
            "--replay-failed",
        ],
    )
    sync_shared_db.main()

    conn = connect(db_path)  # noqa: SQL001
    rows = conn.execute("SELECT name FROM foo").fetchall()
    conn.close()
    assert rows == [("ok",)]
    assert not failed_file.exists()
    assert (queue_dir / "queue.failed.jsonl.bak").exists()
