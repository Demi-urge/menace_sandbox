import time

import pytest

import audit_utils


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_safe_write_audit_watchdog(monkeypatch):
    def _connect(path, **_kwargs):
        return _DummyConnection()

    monkeypatch.setattr(audit_utils.sqlite3, "connect", _connect)
    monkeypatch.setattr(audit_utils, "configure_audit_sqlite_connection", lambda _c: None)

    def _slow_write(_conn):
        time.sleep(0.2)

    with pytest.raises(TimeoutError):
        audit_utils.safe_write_audit(
            "audit.db", _slow_write, timeout=0.01, watchdog_seconds=0.05
        )
