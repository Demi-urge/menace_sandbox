import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import pytest
import json


class DummyEscalation:
    def __init__(self) -> None:
        self.messages = []
        self.attachments = []
        self.session_ids = []
        self.vector_meta = []

    def handle(
        self, msg, attachments=None, session_id=None, vector_metadata=None
    ):
        self.messages.append(msg)
        self.attachments.append(attachments)
        self.session_ids.append(session_id)
        self.vector_meta.append(vector_metadata)


class DummyDB:
    def __init__(self):
        self.updated = []

    def update_bot(self, bot_id, **fields):
        self.updated.append((bot_id, fields))


def _stub_vector_service(monkeypatch):
    import types
    import sys
    import functools
    import time
    import logging

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

    dec = types.ModuleType("vector_service.decorators")

    def log_and_measure(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__qualname__
            start = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception:
                end = time.time()
                dec._CALL_COUNT.labels(name).inc()
                dec._LATENCY_GAUGE.labels(name).set(end - start)
                logging.getLogger(func.__module__).error("context build failed")
                raise
            end = time.time()
            dec._CALL_COUNT.labels(name).inc()
            dec._LATENCY_GAUGE.labels(name).set(end - start)
            size = len(result) if hasattr(result, "__len__") else 0
            dec._RESULT_SIZE_GAUGE.labels(name).set(size)
            return result

        return wrapper

    dec.log_and_measure = log_and_measure
    dec._CALL_COUNT = Gauge()
    dec._LATENCY_GAUGE = Gauge()
    dec._RESULT_SIZE_GAUGE = Gauge()

    class CognitionLayer:
        def __init__(self, *, context_builder=None, **_):
            self.context_builder = context_builder

        def query(self, prompt, **_):
            return self.context_builder.build(prompt, session_id="s"), "sid"

    class ContextBuilder:
        calls: list[tuple[str, bool]] = []

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def build(self, prompt, session_id=None, include_vectors=False, **_):
            self.__class__.calls.append((prompt, include_vectors))
            if include_vectors:
                return {"snippet": "ctx"}, session_id, [("obj", "vec", 0.1)]
            return {"snippet": "ctx"}

        def refresh_db_weights(self):
             pass

        build_context = build

    vs = types.ModuleType("vector_service")
    vs.CognitionLayer = CognitionLayer
    vs.ContextBuilder = ContextBuilder
    class EmbeddableDBMixin:
        pass

    vs.EmbeddableDBMixin = EmbeddableDBMixin
    vs.decorators = dec
    monkeypatch.setitem(sys.modules, "vector_service", vs)
    monkeypatch.setitem(sys.modules, "vector_service.decorators", dec)

    return dec


def test_escalation_on_critical(monkeypatch):
    _stub_vector_service(monkeypatch)
    import menace.automated_reviewer as ar
    import vector_service

    esc = DummyEscalation()
    db = DummyDB()
    builder = vector_service.ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    reviewer = ar.AutomatedReviewer(
        bot_db=db, escalation_manager=esc, context_builder=builder
    )
    reviewer.handle({"bot_id": "7", "severity": "critical"})
    assert vector_service.ContextBuilder.calls
    assert vector_service.ContextBuilder.calls[0][1] is True
    assert db.updated and db.updated[0][0] == 7
    assert esc.messages and "review for bot 7" in esc.messages[0]
    assert esc.session_ids and esc.session_ids[0]
    assert esc.attachments and "ctx" in json.loads(esc.attachments[0][0])["snippet"]
    assert esc.vector_meta and esc.vector_meta[0] == [("obj", "vec", 0.1)]


def test_vector_service_metrics_and_fallback(monkeypatch, caplog):
    dec = _stub_vector_service(monkeypatch)
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

    import vector_service

    class DummyRetriever:
        @log_and_measure
        def search(self, query, **_):
            raise ValueError("context build failed")

    class DummyBuilder(vector_service.ContextBuilder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []
            self.retriever = DummyRetriever()

        def build(self, query, **_):
            self.calls.append(query)
            return self.retriever.search(query, session_id="s")

    builder = DummyBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )

    attachments_list: list[str] = []

    class Escalator:
        def handle(
            self, msg, attachments=None, session_id=None, vector_metadata=None
        ):
            if attachments:
                attachments_list.extend(attachments)

    class DB:
        def update_bot(self, *a, **k):
            pass

    reviewer = ar.AutomatedReviewer(
        bot_db=DB(), escalation_manager=Escalator(), context_builder=builder
    )
    caplog.set_level("ERROR")
    reviewer.handle({"bot_id": "1", "severity": "critical"})
    assert builder.calls
    assert g1.inc_calls == 1
    assert attachments_list == [""]
    assert "context build failed" in caplog.text


def test_refresh_db_weights_failure(monkeypatch):
    _stub_vector_service(monkeypatch)
    import menace.automated_reviewer as ar
    import vector_service

    class BadBuilder(vector_service.ContextBuilder):
        def refresh_db_weights(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ar.AutomatedReviewer(context_builder=BadBuilder())
