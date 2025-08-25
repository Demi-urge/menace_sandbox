import json
import logging
import os
import time
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text

import sync_shared_db
from sync_shared_db import process_queue_file
from queue_cleanup import cleanup
from db_write_queue import queue_insert


@pytest.fixture
def engine(tmp_path):
    db_path = tmp_path / "db.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    meta = MetaData()
    Table(
        "foo",
        meta,
        Column("id", Integer, primary_key=True),
        Column("name", String),
        Column("content_hash", String, unique=True),
    )
    meta.create_all(engine)
    return engine


def _write_records(path: Path, records):
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec))
            fh.write("\n")


def test_process_queue_moves_failed_after_retries(tmp_path, engine, caplog):
    queue_file = tmp_path / "foo_queue.jsonl"
    ok = {
        "table": "foo",
        "op": "insert",
        "data": {"name": "ok"},
        "content_hash": "h1",
        "source_menace_id": "",
    }
    bad = {
        "table": "missing",
        "op": "insert",
        "data": {"name": "bad"},
        "content_hash": "h2",
        "source_menace_id": "",
    }
    _write_records(queue_file, [ok, bad])

    with caplog.at_level(logging.INFO):
        process_queue_file(queue_file, engine=engine)
    # One commit and one rollback logged
    assert any("\"event\": \"commit\"" in r.message for r in caplog.records)
    assert any("\"event\": \"rollback\"" in r.message for r in caplog.records)

    # Failing record left in queue with fail_count=1
    data = json.loads(queue_file.read_text().strip())
    assert data["fail_count"] == 1

    # Retry twice more -> moved to failed file
    process_queue_file(queue_file, engine=engine)
    process_queue_file(queue_file, engine=engine)
    assert queue_file.read_text() == ""
    failed = tmp_path / "foo_queue.failed.jsonl"
    assert failed.exists()
    rec = json.loads(failed.read_text().strip())
    assert rec["fail_count"] == 3
    assert not (tmp_path / "foo_queue.jsonl.tmp").exists()


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


def test_multiple_instances_commit_once(tmp_path, engine, monkeypatch):
    queue_dir = tmp_path
    monkeypatch.setenv("MENACE_ID", "alpha")
    queue_insert("foo", {"name": "a"}, ["name"], queue_dir)
    monkeypatch.setenv("MENACE_ID", "beta")
    queue_insert("foo", {"name": "b"}, ["name"], queue_dir)
    queue_file = queue_dir / "foo_queue.jsonl"

    process_queue_file(queue_file, engine=engine)

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT name FROM foo ORDER BY name")).fetchall()
    assert [r[0] for r in rows] == ["a", "b"]

    process_queue_file(queue_file, engine=engine)
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM foo")).scalar()
    assert count == 2


def test_crash_mid_batch_recovers(tmp_path, engine, monkeypatch):
    queue_dir = tmp_path
    monkeypatch.setenv("MENACE_ID", "alpha")
    queue_insert("foo", {"name": "a"}, ["name"], queue_dir)
    monkeypatch.setenv("MENACE_ID", "beta")
    queue_insert("foo", {"name": "b"}, ["name"], queue_dir)
    queue_file = queue_dir / "foo_queue.jsonl"

    real_insert = sync_shared_db.insert_if_unique
    calls = {"n": 0}

    def crash_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_insert(*args, **kwargs)

    monkeypatch.setattr(sync_shared_db, "insert_if_unique", crash_on_second)

    with pytest.raises(KeyboardInterrupt):
        process_queue_file(queue_file, engine=engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM foo")).scalar()
    assert count == 1
    assert queue_file.read_text().count("\n") == 2

    monkeypatch.setattr(sync_shared_db, "insert_if_unique", real_insert)
    process_queue_file(queue_file, engine=engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM foo")).scalar()
    assert count == 2
    assert queue_file.read_text() == ""


def test_malformed_lines_left_intact(tmp_path, engine, monkeypatch):
    queue_dir = tmp_path
    monkeypatch.setenv("MENACE_ID", "alpha")
    queue_insert("foo", {"name": "ok"}, ["name"], queue_dir)
    queue_file = queue_dir / "foo_queue.jsonl"
    with queue_file.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")

    process_queue_file(queue_file, engine=engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM foo")).scalar()
    assert count == 1
    assert queue_file.read_text() == "not-json\n"


def test_duplicate_hashes_committed_once(tmp_path, engine, monkeypatch):
    queue_dir = tmp_path
    monkeypatch.setenv("MENACE_ID", "alpha")
    queue_insert("foo", {"name": "dup"}, ["name"], queue_dir)
    monkeypatch.setenv("MENACE_ID", "beta")
    queue_insert("foo", {"name": "dup"}, ["name"], queue_dir)
    queue_file = queue_dir / "foo_queue.jsonl"

    process_queue_file(queue_file, engine=engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM foo")).scalar()
    assert count == 1
    assert queue_file.read_text() == ""
