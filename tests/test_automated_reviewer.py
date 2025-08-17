import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.automated_reviewer as ar

class DummyEscalation:
    def __init__(self) -> None:
        self.messages = []
    def handle(self, msg, attachments=None):
        self.messages.append(msg)

class DummyDB:
    def __init__(self):
        self.updated = []
    def update_bot(self, bot_id, **fields):
        self.updated.append((bot_id, fields))


def test_escalation_on_critical():
    esc = DummyEscalation()
    db = DummyDB()
    reviewer = ar.AutomatedReviewer(bot_db=db, escalation_manager=esc)
    reviewer.handle({"bot_id": "7", "severity": "critical"})
    assert db.updated and db.updated[0][0] == 7
    assert esc.messages and "review for bot 7" in esc.messages[0]

def test_vector_service_metrics_and_fallback(monkeypatch, caplog):
    from vector_service import FallbackResult
    import vector_service.decorators as dec
    from vector_service.decorators import log_and_measure
    import menace.automated_reviewer as ar

    class Gauge:
        def __init__(self):
            self.inc_calls = 0
            self.set_calls: list[float] = []
        def labels(self, *args):
            return self
        def inc(self):
            self.inc_calls += 1
        def set(self, value):
            self.set_calls.append(value)

    g1, g2, g3 = Gauge(), Gauge(), Gauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", g1)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", g2)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", g3)

    class DummyRetriever:
        @log_and_measure
        def search(self, query, **_):
            return FallbackResult("sentinel_fallback", [])

    class DummyBuilder:
        def __init__(self):
            self.calls = []
            self.retriever = DummyRetriever()
        def build(self, query):
            self.calls.append(query)
            return self.retriever.search(query, session_id="s")

    builder = DummyBuilder()
    monkeypatch.setattr(ar, "ContextBuilder", lambda *a, **k: builder)

    attachments_list: list[str] = []
    class Escalator:
        def handle(self, msg, attachments=None):
            if attachments:
                attachments_list.extend(attachments)
    class DB:
        def update_bot(self, *a, **k):
            pass

    reviewer = ar.AutomatedReviewer(bot_db=DB(), escalation_manager=Escalator())
    caplog.set_level("ERROR")
    reviewer.handle({"bot_id": "1", "severity": "critical"})
    assert builder.calls
    assert g1.inc_calls == 1
    assert attachments_list == [""]
    assert "context build failed" in caplog.text
