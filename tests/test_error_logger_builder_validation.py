import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.error_logger as elog  # noqa: E402
from menace.error_logger import ErrorLogger  # noqa: E402


class DummyBuilder:
    def __init__(self):
        self.refresh_calls = 0

    def refresh_db_weights(self):
        self.refresh_calls += 1


def test_refresh_called_during_init(monkeypatch, tmp_path):
    builder = DummyBuilder()
    db = types.SimpleNamespace(add_telemetry=lambda *a, **k: None)
    logger = ErrorLogger(db=db, context_builder=builder)
    called: list[bool] = []

    def fake_generate_patch(module, manager, *, context_builder):
        assert context_builder is builder
        assert builder.refresh_calls == 1
        called.append(True)
        return 1

    monkeypatch.setattr(elog, "generate_patch", fake_generate_patch)
    monkeypatch.setattr(elog, "propose_fix", lambda metrics, profile: [("mod", "hint")])
    monkeypatch.setattr(elog, "path_for_prompt", lambda module: module)
    monkeypatch.setattr(elog, "cdh", None)

    logger.log_fix_suggestions({}, {}, task_id="t", bot_id="b")
    assert called
