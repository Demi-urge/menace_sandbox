import logging
import os

def test_setup_logging_verbose(monkeypatch):
    monkeypatch.delenv("SANDBOX_CENTRAL_LOGGING", raising=False)
    monkeypatch.setenv("SANDBOX_VERBOSE", "1")
    import logging_utils as lu

    lu.setup_logging()
    assert logging.getLogger().level == logging.DEBUG

