import json
import pytest
import sqlite3
import menace.error_logger as elog
import types


class DummyDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE telemetry (id INTEGER PRIMARY KEY, fix_suggestions TEXT, bottlenecks TEXT)"
        )

    def add_telemetry(self, event):
        self.conn.execute(
            "INSERT INTO telemetry (fix_suggestions, bottlenecks) VALUES (?, ?)",
            (json.dumps(event.fix_suggestions), json.dumps(event.bottlenecks)),
        )
        self.conn.commit()


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


def test_log_fix_suggestions_emits_events_and_triggers_patch_and_codex(
    tmp_path, monkeypatch, caplog
):
    db = DummyDB(tmp_path / "e.db")
    manager = DummyManager()
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder(), manager=manager)
    monkeypatch.setattr(
        elog,
        "propose_fix",
        lambda m, p: [("mod1", "hint1"), ("", "generic hint")],
    )
    monkeypatch.setattr(elog, "path_for_prompt", lambda module: module)
    caplog.set_level("INFO", logger=elog.__name__)

    events = logger.log_fix_suggestions({"a": 0.1}, {}, task_id="t1", bot_id="b1")

    assert [e.fix_suggestions for e in events] == [["hint1"], ["generic hint"]]
    assert [e.bottlenecks for e in events] == [["mod1"], []]
    assert manager.calls == ["mod1"]
    assert not any("Codex prompt" in r.message for r in caplog.records)
    rows = db.conn.execute(
        "SELECT fix_suggestions, bottlenecks FROM telemetry ORDER BY id"
    ).fetchall()
    assert [json.loads(r[0]) for r in rows] == [["hint1"], ["generic hint"]]
    assert [json.loads(r[1]) for r in rows] == [["mod1"], []]


def test_log_fix_suggestions_requires_builder(tmp_path):
    db = DummyDB(tmp_path / "e.db")
    with pytest.raises(TypeError):
        elog.ErrorLogger(db, context_builder=None, manager=DummyManager())  # type: ignore[arg-type]
