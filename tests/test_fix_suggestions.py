
import json
import menace.error_bot as eb
import menace.error_logger as elog


def test_log_fix_suggestions_records_and_triggers(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db)
    calls = []

    def fake_patch(module, *a, **k):
        calls.append(module)
        return 1

    monkeypatch.setattr(elog, "generate_patch", fake_patch, raising=False)

    event = logger.log_fix_suggestions([("mod", "hint")], task_id="t1", bot_id="b1")
    assert event.fix_suggestions == ["hint"]
    assert calls == ["mod"]
    rows = db.conn.execute("SELECT fix_suggestions FROM telemetry").fetchall()
    assert rows and rows[0][0] == json.dumps(event.fix_suggestions)
