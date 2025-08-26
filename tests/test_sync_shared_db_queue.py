import json
from sqlite3 import connect
from pathlib import Path

from db_dedup import compute_content_hash
import sync_shared_db


def _names(db_path: Path) -> list[str]:
    conn = connect(db_path)  # noqa: SQL001
    rows = conn.execute("SELECT name FROM foo ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]


def test_process_queue_success_and_failure(tmp_path):
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE)"
    )
    conn.commit()

    queue_file = tmp_path / "m1.jsonl"
    good = {"table": "foo", "record": {"name": "ok"}, "menace_id": "m1"}
    bad = {"table": "missing", "record": {"name": "bad"}, "menace_id": "m1"}
    with queue_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write(json.dumps(bad) + "\n")

    stats = sync_shared_db.process_queue_file(queue_file, conn=conn)
    conn.close()

    assert stats.processed == 1
    assert stats.failures == 1
    assert stats.duplicates == 0
    assert _names(db_path) == ["ok"]
    assert queue_file.read_text() == ""
    failed = tmp_path / "queue.failed.jsonl"
    rec = json.loads(failed.read_text().strip())
    assert rec["record"]["table"] == "missing"


def test_duplicate_lines(tmp_path):
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE)"
    )
    conn.commit()

    queue_file = tmp_path / "m1.jsonl"
    entry = {"table": "foo", "record": {"name": "dup"}, "menace_id": "m1"}
    with queue_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.write(json.dumps(entry) + "\n")

    stats = sync_shared_db.process_queue_file(queue_file, conn=conn)
    conn.close()

    assert stats.processed == 1
    assert stats.duplicates == 1
    assert stats.failures == 0
    assert _names(db_path) == ["dup"]
    assert queue_file.read_text() == ""
    assert not (tmp_path / "queue.failed.jsonl").exists()


def test_recovery_skips_processed_hashes(tmp_path):
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)  # noqa: SQL001
    conn.execute(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE)"
    )
    conn.commit()

    entry1 = {"table": "foo", "record": {"name": "a"}, "menace_id": "m1"}
    entry2 = {"table": "foo", "record": {"name": "b"}, "menace_id": "m1"}
    for e in (entry1, entry2):
        h = compute_content_hash(e["record"])
        e["hash"] = h
        e["hash_fields"] = list(e["record"].keys())

    queue_file = tmp_path / "m1.jsonl"
    with queue_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(entry1) + "\n")
        fh.write(json.dumps(entry2) + "\n")

    processed_log = tmp_path / "processed.log"
    processed_log.write_text(entry1["hash"] + "\n")

    stats = sync_shared_db.process_queue_file(queue_file, conn=conn)
    conn.close()

    assert stats.processed == 1
    assert stats.duplicates == 1
    assert stats.failures == 0
    assert _names(db_path) == ["b"]
    assert queue_file.read_text() == ""
    logged = processed_log.read_text().splitlines()
    assert entry1["hash"] in logged and entry2["hash"] in logged
