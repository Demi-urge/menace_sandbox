import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault(
    "menace.data_bot", types.SimpleNamespace(MetricsDB=object, DataBot=object)
)

import menace.error_logger as elog  # noqa: E402
from menace.error_logger import ErrorLogger  # noqa: E402
import types
from menace.error_bot import ErrorDB  # noqa: E402


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyManager:
    def __init__(self):
        self.evolution_orchestrator = types.SimpleNamespace(provenance_token="tok", event_bus=None)

    def generate_patch(self, module, description="", context_builder=None, provenance_token="", **kwargs):  # pragma: no cover - stub
        return 1


def test_error_logger_triggers_rule_update(monkeypatch, tmp_path):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    db = ErrorDB(path=tmp_path / "errors.db")
    logger = ErrorLogger(db=db, context_builder=DummyBuilder(), manager=DummyManager())

    called: list[bool] = []

    def fake_update(db_obj, **kwargs):
        called.append(True)

    monkeypatch.setattr(logger.classifier, "learn_error_phrases", fake_update)

    for _ in range(logger._update_threshold):
        logger.log(Exception("boom"), None, None)

    assert called
