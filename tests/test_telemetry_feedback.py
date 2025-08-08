import menace.error_bot as eb
import menace.error_logger as elog
import menace.telemetry_feedback as tf
from pathlib import Path


class DummyEngine:
    def __init__(self):
        self.calls = []

    def apply_patch(self, path: Path, desc: str):
        self.calls.append((path, desc))
        return 1, False, 0.0


def _setup(tmp_path):
    db = eb.ErrorDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db)
    engine = DummyEngine()
    mod = tmp_path / "bot.py"
    mod.write_text("def x():\n    pass\n")
    return db, logger, engine, mod


def test_feedback_triggers_patch(tmp_path, monkeypatch):
    db, logger, engine, mod = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    trace = "Traceback...\nKeyError: boom"
    for _ in range(3):
        db.add_telemetry(
            elog.TelemetryEvent(
                error_type=elog.ErrorType.RUNTIME_FAULT,
                stack_trace=trace,
                root_module="bot",
            )
        )
    fb = tf.TelemetryFeedback(logger, engine, threshold=3)
    fb.check()
    assert engine.calls and engine.calls[0][0] == Path("bot.py")
    rows = db.conn.execute("SELECT patch_id, resolution_status FROM telemetry").fetchall()
    assert all(r[0] == 1 and r[1] == "attempted" for r in rows)


def test_feedback_threshold(tmp_path, monkeypatch):
    db, logger, engine, _ = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    trace = "Traceback...\nKeyError: boom"
    for _ in range(2):
        db.add_telemetry(
            elog.TelemetryEvent(
                error_type=elog.ErrorType.RUNTIME_FAULT,
                stack_trace=trace,
                root_module="bot",
            )
        )
    fb = tf.TelemetryFeedback(logger, engine, threshold=3)
    fb.check()
    assert not engine.calls
    rows = db.conn.execute("SELECT patch_id FROM telemetry").fetchall()
    assert all(r[0] is None for r in rows)
