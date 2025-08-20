import json
import menace.error_bot as eb
import menace.error_logger as elog


def test_log_fix_suggestions_emits_events_and_triggers_patch_and_codex(
    tmp_path, monkeypatch, caplog
):
    db = eb.ErrorDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db)
    patch_calls = []

    def fake_patch(module, *a, **k):
        patch_calls.append(module)
        return 1

    monkeypatch.setattr(elog, "generate_patch", fake_patch, raising=False)
    caplog.set_level("INFO", logger=elog.__name__)
    suggestions = [("mod1", "hint1"), "generic hint"]
    events = logger.log_fix_suggestions(suggestions, task_id="t1", bot_id="b1")

    assert [e.fix_suggestions for e in events] == [["hint1"], ["generic hint"]]
    assert patch_calls == ["mod1"]
    assert any("Codex prompt" in r.message and "generic hint" in r.message for r in caplog.records)
    rows = db.conn.execute("SELECT fix_suggestions FROM telemetry ORDER BY id").fetchall()
    assert [json.loads(r[0]) for r in rows] == [["hint1"], ["generic hint"]]
