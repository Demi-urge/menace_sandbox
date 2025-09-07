import json
import pytest
import sqlite3
import menace.error_logger as elog


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


def test_log_fix_suggestions_emits_events_and_triggers_patch_and_codex(
    tmp_path, monkeypatch, caplog
):
    db = DummyDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder())
    patch_calls = []

    def fake_patch(module, *a, **k):
        patch_calls.append(module)
        return 1

    monkeypatch.setattr(elog, "generate_patch", fake_patch, raising=False)
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
    assert patch_calls == ["mod1"]
    assert any(
        "Codex prompt" in r.message and "generic hint" in r.message for r in caplog.records
    )
    rows = db.conn.execute(
        "SELECT fix_suggestions, bottlenecks FROM telemetry ORDER BY id"
    ).fetchall()
    assert [json.loads(r[0]) for r in rows] == [["hint1"], ["generic hint"]]
    assert [json.loads(r[1]) for r in rows] == [["mod1"], []]


def test_log_fix_suggestions_requires_builder(tmp_path):
    db = DummyDB(tmp_path / "e.db")
    with pytest.raises(TypeError):
        elog.ErrorLogger(db, context_builder=None)  # type: ignore[arg-type]
