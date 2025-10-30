import importlib
from pathlib import Path
import sqlite3

audit_logger = importlib.import_module("audit_logger")
bot_registry = importlib.import_module("menace_sandbox.bot_registry")
db_router = importlib.import_module("db_router")


class DummyRouter:
    """Router stub recording requested table names."""

    def __init__(self, connect):
        self.connect = connect
        self.calls = []

    def get_connection(self, table_name: str, operation: str = "read"):
        self.calls.append((table_name, operation))
        return self.connect(":memory:")


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

    connect_calls: list[Path] = []

    def tracked_connect(path, *args, **kwargs):
        connect_calls.append(Path(path))
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(audit_logger.sqlite3, "connect", tracked_connect)

    audit_logger.log_to_sqlite("test", {}, db_path=audit_path)

    assert router.calls == [("events", "write")]
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
