import importlib
import sys
from pathlib import Path
import sqlite3

audit_logger = importlib.import_module("audit_logger")
bot_registry = importlib.import_module("bot_registry")
sys.modules.setdefault("menace_sandbox.bot_registry", bot_registry)
db_router = importlib.import_module("db_router")


class DummyRouter:
    """Router stub recording requested table names."""

    class _ConnectionWrapper:
        __slots__ = ("_conn", "__weakref__")

        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, item):
            return getattr(self._conn, item)

    def _wrap(self, conn):
        return self._ConnectionWrapper(conn)

    def __init__(self, connect):
        self.connect = connect
        self.calls = []

    def get_connection(self, table_name: str, operation: str = "read"):
        self.calls.append((table_name, operation))
        return self._wrap(self.connect(":memory:"))


def test_audit_logger_uses_router(monkeypatch, tmp_path):
    original_connect = sqlite3.connect
    router = DummyRouter(original_connect)
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", router)
    monkeypatch.setattr(db_router, "init_db_router", lambda *a, **k: router)
    monkeypatch.setattr(audit_logger, "GLOBAL_ROUTER", router)
    monkeypatch.setattr(audit_logger, "init_db_router", lambda *a, **k: router)

    audit_path = tmp_path / "audit.db"
    monkeypatch.setattr(audit_logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(audit_logger, "SQLITE_PATH", audit_path)

    with audit_logger._AUDIT_QUEUE_LOCK:
        audit_logger._AUDIT_QUEUE.clear()
    audit_logger._set_bootstrap_queueing_enabled(True)

    connect_calls: list[Path] = []

    def tracked_connect(path, *args, **kwargs):
        connect_calls.append(Path(path))
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(audit_logger.sqlite3, "connect", tracked_connect)

    jsonl_path = tmp_path / "audit.jsonl"

    audit_logger.log_to_sqlite("test", {}, db_path=None, jsonl_path=jsonl_path)
    audit_logger.flush_queued_events()

    assert router.calls == [("events", "write")]

    # Subsequent writes should go straight to SQLite without re-queuing.
    connect_calls.clear()
    router.calls.clear()
    audit_logger.log_to_sqlite("steady", {}, db_path=audit_path, jsonl_path=jsonl_path)

    assert router.calls == []
    assert connect_calls and connect_calls[0] == audit_path


def test_bot_registry_save_uses_router(monkeypatch, tmp_path):
    original_connect = sqlite3.connect
    router = DummyRouter(original_connect)
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", router)
    monkeypatch.setattr(db_router, "init_db_router", lambda *a, **k: router)
    monkeypatch.setattr(bot_registry, "init_db_router", lambda *a, **k: router)
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sqlite3.connect called")))

    reg = bot_registry.BotRegistry()
    reg.register_interaction("a", "b")
    reg.save(tmp_path / "registry.db")

    assert router.calls == [("bots", "read")]
