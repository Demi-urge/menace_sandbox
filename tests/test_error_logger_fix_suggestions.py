import json
import menace.error_bot as eb
import menace.error_logger as elog


class DummyBuilder:
    def refresh_db_weights(self):
        pass


def test_log_fix_suggestions_emits_events_and_triggers_patch_and_codex(
    tmp_path, monkeypatch, caplog
):
    db = eb.ErrorDB(tmp_path / "e.db")
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


def test_log_fix_suggestions_without_builder_does_not_patch(
    tmp_path, monkeypatch
):
    db = eb.ErrorDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db, context_builder=None)
    patch_calls = []

    def fake_patch(module, *a, **k):
        patch_calls.append(module)
        return 1

    monkeypatch.setattr(elog, "generate_patch", fake_patch, raising=False)
    monkeypatch.setattr(
        elog, "propose_fix", lambda m, p: [("mod1", "hint1")]
    )

    logger.log_fix_suggestions({"a": 0.1}, {}, task_id="t1", bot_id="b1")
    assert patch_calls == []
