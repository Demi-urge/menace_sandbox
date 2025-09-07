
import json
import menace.error_bot as eb
import menace.error_logger as elog


class DummyBuilder:
    def refresh_db_weights(self):
        pass


def test_log_fix_suggestions_records_and_triggers(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder())
    calls = []

    def fake_patch(module, *a, **k):
        calls.append(module)
        return 1

    monkeypatch.setattr(elog, "generate_patch", fake_patch, raising=False)
    monkeypatch.setattr(elog, "propose_fix", lambda m, p: [("mod", "hint")])
    ticket_file = tmp_path / "tickets.txt"
    monkeypatch.setenv("FIX_TICKET_FILE", str(ticket_file))

    events = logger.log_fix_suggestions({"metric": 0.1}, {}, task_id="t1", bot_id="b1")
    assert isinstance(events, list) and events
    event = events[0]
    assert event.fix_suggestions == ["hint"]
    assert event.bottlenecks == ["mod"]
    assert event.error_type == elog.ErrorCategory.MetricBottleneck
    assert calls == ["mod"]
    rows = db.conn.execute("SELECT fix_suggestions, bottlenecks FROM telemetry").fetchall()
    assert rows and rows[0][0] == json.dumps(event.fix_suggestions)
    assert rows[0][1] == json.dumps(event.bottlenecks)
    assert ticket_file.read_text().strip()
