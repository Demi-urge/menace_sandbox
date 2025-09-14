import sys
import types
import importlib
import menace.error_logger as elog
import types
import menace.sentry_client as sc


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyManager:
    def __init__(self):
        self.evolution_orchestrator = types.SimpleNamespace(provenance_token="tok", event_bus=None)

    def generate_patch(self, module, description="", context_builder=None, provenance_token="", **kwargs):  # pragma: no cover - stub
        return 1


class StubSDK:
    def __init__(self):
        self.captured = []

    def init(self, dsn=None, **kw):
        self.dsn = dsn

    def capture_exception(self, exc):
        self.captured.append(str(exc))


def test_error_logger_with_sentry(monkeypatch):
    sdk = StubSDK()
    monkeypatch.setitem(sys.modules, "sentry_sdk", sdk)
    importlib.reload(sc)
    importlib.reload(elog)
    monkeypatch.setattr(elog, "get_embedder", lambda: None)

    sentry = sc.SentryClient("http://dsn")
    events = []
    db = types.SimpleNamespace(add_telemetry=lambda e: events.append(e))
    logger = elog.ErrorLogger(db, sentry=sentry, context_builder=DummyBuilder(), manager=DummyManager())
    try:
        raise RuntimeError("boom")
    except Exception as exc:
        logger.log(exc, "t1", "bot1")
    assert sdk.dsn == "http://dsn"
    assert sdk.captured and "boom" in sdk.captured[0]
    assert events


class FailingSDK:
    def init(self, dsn=None, **kw):
        self.dsn = dsn

    def capture_exception(self, exc):
        raise RuntimeError("fail")


def test_capture_exception_logs_failure(monkeypatch, caplog):
    sdk = FailingSDK()
    monkeypatch.setitem(sys.modules, "sentry_sdk", sdk)
    importlib.reload(sc)
    caplog.set_level("ERROR")
    sentry = sc.SentryClient("http://dsn")
    sentry.capture_exception(RuntimeError("boom"))
    assert "failed to send exception" in caplog.text


class ListLogger:
    def __init__(self):
        self.logged = []

    def error(self, msg, *args):
        self.logged.append(msg % args)


def test_custom_fallback_logger(monkeypatch):
    sdk = FailingSDK()
    monkeypatch.setitem(sys.modules, "sentry_sdk", sdk)
    importlib.reload(sc)
    logger = ListLogger()
    sentry = sc.SentryClient("http://dsn", fallback_logger=logger)
    sentry.capture_exception(RuntimeError("boom"))
    assert any("failed to send" in m for m in logger.logged)
