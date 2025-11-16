
import json
import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import menace.error_bot as eb
import menace.error_logger as elog
import types


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyManager:
    def __init__(self):
        self.calls = []
        self.evolution_orchestrator = types.SimpleNamespace(provenance_token="tok", event_bus=None)

    def generate_patch(self, module, description="", context_builder=None, provenance_token="", **kwargs):  # pragma: no cover - stub
        self.calls.append(module)
        return 1


def test_log_fix_suggestions_records_and_triggers(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e.db")
    manager = DummyManager()
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder(), manager=manager)
    monkeypatch.setattr(elog, "propose_fix", lambda m, p: [("mod", "hint")])
    ticket_file = tmp_path / "tickets.txt"
    monkeypatch.setenv("FIX_TICKET_FILE", str(ticket_file))

    events = logger.log_fix_suggestions({"metric": 0.1}, {}, task_id="t1", bot_id="b1")
    assert isinstance(events, list) and events
    event = events[0]
    assert event.fix_suggestions == ["hint"]
    assert event.bottlenecks == ["mod"]
    assert event.error_type == elog.ErrorCategory.MetricBottleneck
    assert manager.calls == ["mod"]
    rows = db.conn.execute("SELECT fix_suggestions, bottlenecks FROM telemetry").fetchall()
    assert rows and rows[0][0] == json.dumps(event.fix_suggestions)
    assert rows[0][1] == json.dumps(event.bottlenecks)
    assert ticket_file.read_text().strip()
