import logging
import os

from pathlib import Path


def test_central_logging_audit(monkeypatch, tmp_path):
    log_file = tmp_path / "central.log"
    monkeypatch.setenv("SANDBOX_CENTRAL_LOGGING", "1")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log_file))
    monkeypatch.delenv("KAFKA_HOSTS", raising=False)

    import logging_utils as lu

    lu.setup_logging()

    logger = logging.getLogger("SelfTestService")
    logger.info("hello")

    assert log_file.exists()
    data = log_file.read_text()
    assert "hello" in data

