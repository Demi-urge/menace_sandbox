import json
from menace import audit_logger as al


def test_jsonl_logging_and_retrieval(tmp_path, monkeypatch):
    log_dir = tmp_path
    jsonl = log_dir / "audit.jsonl"
    db = log_dir / "audit.db"
    monkeypatch.setattr(al, "LOG_DIR", str(log_dir))
    monkeypatch.setattr(al, "JSONL_PATH", str(jsonl))
    monkeypatch.setattr(al, "SQLITE_PATH", str(db))

    al.log_event("test", {"x": 1}, str(jsonl))
    events = al.get_recent_events(jsonl_path=str(jsonl), db_path=str(db))
    assert len(events) == 1
    assert events[0]["data"]["x"] == 1

    csv_path = log_dir / "out.csv"
    al.export_to_csv(str(jsonl), str(csv_path))
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) >= 2
