import json
import logging
import sqlite3

import db_router


def test_shared_table_queue(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DB_ROUTER_QUEUE_DIR", str(tmp_path))
    menace_id = "m1"
    caplog.set_level(logging.INFO)
    db_router.queue_insert("telemetry", {"foo": 1}, menace_id)
    queue_file = tmp_path / f"{menace_id}.jsonl"
    assert queue_file.exists()
    line = queue_file.read_text().strip()
    data = json.loads(line)
    assert data["table"] == "telemetry"
    assert data["data"] == {"foo": 1}
    assert data["source_menace_id"] == menace_id
    record = json.loads(caplog.records[0].message)
    assert record["menace_id"] == menace_id
    assert record["table_name"] == "telemetry"
    assert record["operation"] == "queue_insert"


def test_local_table_insert(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_ROUTER_QUEUE_DIR", str(tmp_path))
    router = db_router.DBRouter("loc", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db_router.GLOBAL_ROUTER = router
    conn = router.get_connection("events", "write")
    conn.execute("CREATE TABLE events (name TEXT)")
    conn.commit()
    db_router.queue_insert("events", {"name": "bob"}, "loc")
    rows = conn.execute("SELECT name FROM events").fetchall()
    assert rows == [("bob",)]
    assert not (tmp_path / "loc.jsonl").exists()
