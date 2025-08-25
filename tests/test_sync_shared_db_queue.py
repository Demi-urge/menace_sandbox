import json
import logging
import os
import time
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from sync_shared_db import process_queue_file
from queue_cleanup import cleanup


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
