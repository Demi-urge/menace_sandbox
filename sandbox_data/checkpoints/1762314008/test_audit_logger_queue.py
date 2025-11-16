import json
import sqlite3
from pathlib import Path

import pytest

import audit_logger
from audit_logger import flush_queued_events, queue_event, queue_event_for_later


@pytest.fixture(autouse=True)
def reset_audit_logger_state() -> None:
    """Ensure each test starts with a clean audit logger queue."""

    with audit_logger._AUDIT_QUEUE_LOCK:
        audit_logger._AUDIT_QUEUE.clear()
    audit_logger._set_bootstrap_queueing_enabled(True)
    yield
    with audit_logger._AUDIT_QUEUE_LOCK:
        audit_logger._AUDIT_QUEUE.clear()
    audit_logger._set_bootstrap_queueing_enabled(True)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.mark.usefixtures("tmp_path")
def test_queue_and_flush(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "audit.jsonl"
    db_path = tmp_path / "audit.db"

    event_id = queue_event_for_later(
        "test-event",
        {"value": 42},
        db_path=db_path,
        jsonl_path=jsonl_path,
    )

    entries = _load_jsonl(jsonl_path)
    assert entries
    assert entries[0]["event_id"] == event_id
    assert entries[0]["data"] == {"value": 42}

    flush_queued_events()

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT event_id, event_type, timestamp FROM events"
        ).fetchall()
        assert rows == [(event_id, "test-event", entries[0]["timestamp"])]
        data_rows = conn.execute(
            "SELECT key, value FROM event_data WHERE event_id=?",
            (event_id,),
        ).fetchall()
        assert data_rows == [("value", "42")]

    # A second flush should be a no-op without errors.
    flush_queued_events()


def test_queue_event_alias(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "alias.jsonl"
    db_path = tmp_path / "alias.db"

    event_id = queue_event(
        "alias-event",
        {"value": "queued"},
        db_path=db_path,
        jsonl_path=jsonl_path,
    )

    entries = _load_jsonl(jsonl_path)
    assert entries and entries[0]["event_id"] == event_id
    # Ensure the event is pending flush rather than immediately persisted.
    with audit_logger._AUDIT_QUEUE_LOCK:
        assert any(queued.payload.event_id == event_id for queued in audit_logger._AUDIT_QUEUE)


def test_log_to_sqlite_defers_until_flush(monkeypatch, tmp_path: Path) -> None:
    jsonl_path = tmp_path / "boot.jsonl"
    db_path = tmp_path / "boot.db"

    # Ensure predictable environment paths.
    monkeypatch.setattr(audit_logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(audit_logger, "SQLITE_PATH", db_path)

    event_id = audit_logger.log_to_sqlite(
        "boot-event",
        {"phase": "bootstrap"},
        db_path=db_path,
        jsonl_path=jsonl_path,
    )

    # SQLite file should not exist until an explicit flush occurs.
    assert not db_path.exists()
    with audit_logger._AUDIT_QUEUE_LOCK:
        assert any(queued.payload.event_id == event_id for queued in audit_logger._AUDIT_QUEUE)

    flush_queued_events()

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT event_id, event_type FROM events ORDER BY rowid",
        ).fetchall()
    assert rows == [(event_id, "boot-event")]

    # After flushing, subsequent writes should mirror directly.
    second_id = audit_logger.log_to_sqlite(
        "post-boot",
        {"phase": "steady"},
        db_path=db_path,
        jsonl_path=jsonl_path,
    )
    with audit_logger._AUDIT_QUEUE_LOCK:
        assert all(queued.payload.event_id != second_id for queued in audit_logger._AUDIT_QUEUE)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT event_id, event_type FROM events ORDER BY rowid",
        ).fetchall()
    assert [row[0] for row in rows][-1] == second_id
