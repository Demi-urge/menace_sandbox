import json
import sqlite3
from pathlib import Path

import pytest

from audit_logger import flush_queued_events, queue_event_for_later


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
