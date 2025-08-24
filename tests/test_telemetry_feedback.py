import os
import types
import sys

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("menace.self_coding_engine", types.SimpleNamespace(SelfCodingEngine=object))
sys.modules.setdefault("menace.data_bot", types.SimpleNamespace(MetricsDB=object, DataBot=object))

import menace.error_bot as eb  # noqa: E402
import menace.error_logger as elog  # noqa: E402
import menace.telemetry_feedback as tf  # noqa: E402
from pathlib import Path  # noqa: E402
import pytest  # noqa: E402


class DummyEngine:
    def __init__(self):
        self.calls = []

    def apply_patch(self, path: Path, desc: str, **_: object):
        self.calls.append((path, desc))
        return 1, False, 0.0


class DummyGraph:
    def __init__(self):
        self.events = []
        self.updated = None

    def add_telemetry_event(self, *a, **k):
        self.events.append((a, k))

    def update_error_stats(self, db):
        self.updated = db


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    db = eb.ErrorDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db)
    engine = DummyEngine()
    mod = tmp_path / "bot.py"
    mod.write_text("def x():\n    pass\n")
    return db, logger, engine, mod


@pytest.mark.parametrize("scope, src", [("local", None)])
def test_feedback_triggers_patch(tmp_path, monkeypatch, scope, src):
    db, logger, engine, mod = _setup(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    trace = "Traceback...\nKeyError: boom"
    for _ in range(3):
        db.add_telemetry(
            elog.TelemetryEvent(
                error_type=elog.ErrorType.RUNTIME_FAULT,
                stack_trace=trace,
                root_module="bot",
                module="bot",
                module_counts={"bot": 1},
            ),
            source_menace_id=src,
        )
    fb = tf.TelemetryFeedback(logger, engine, threshold=3)
    fb._run_cycle(scope=scope)
    assert engine.calls and engine.calls[0][0] == Path("bot.py")


def test_feedback_threshold(tmp_path, monkeypatch):
    db, logger, engine, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    trace = "Traceback...\nKeyError: boom"
    for _ in range(2):
        db.add_telemetry(
            elog.TelemetryEvent(
                error_type=elog.ErrorType.RUNTIME_FAULT,
                stack_trace=trace,
                root_module="bot",
                module="bot",
                module_counts={"bot": 1},
            )
        )
    fb = tf.TelemetryFeedback(logger, engine, threshold=3)
    fb._run_cycle()
    assert not engine.calls
