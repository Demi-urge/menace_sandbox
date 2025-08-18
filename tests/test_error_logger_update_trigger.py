import menace.error_logger as elog
from menace.error_logger import ErrorLogger
from menace.error_bot import ErrorDB


def test_error_logger_triggers_rule_update(monkeypatch, tmp_path):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    db = ErrorDB(path=tmp_path / "errors.db")
    logger = ErrorLogger(db=db)

    called: list[bool] = []

    def fake_update(db_obj):
        called.append(True)

    monkeypatch.setattr(logger.classifier, "update_rules_from_db", fake_update)

    for _ in range(logger._update_threshold):
        logger.log(Exception("boom"), None, None)

    assert called

