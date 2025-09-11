import sqlite3
from menace import audit_logger as al


def test_jsonl_logging_and_retrieval(tmp_path, monkeypatch):
    log_dir = tmp_path
    jsonl = log_dir / "audit.jsonl"
    db = log_dir / "audit.db"
    monkeypatch.setattr(al, "LOG_DIR", log_dir)
    monkeypatch.setattr(al, "JSONL_PATH", jsonl)
    monkeypatch.setattr(al, "SQLITE_PATH", db)

    al.log_event("test", {"x": 1}, jsonl)

    class Router:
        def __init__(self, path):
            self.path = path

        def get_connection(self, _):
            return getattr(sqlite3, "connect")(self.path)

    router = Router(db)
    monkeypatch.setattr(al, "GLOBAL_ROUTER", router)
    monkeypatch.setattr(al, "init_db_router", lambda *_a, **_k: router)

    events = al.get_recent_events(jsonl_path=jsonl, db_path=db)
    assert len(events) == 1
    assert events[0]["data"]["x"] == 1

    csv_path = log_dir / "out.csv"
    al.export_to_csv(jsonl, csv_path)
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) >= 2
