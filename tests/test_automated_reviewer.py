import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import pytest
import json
import types


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

    bus_mod = types.ModuleType("menace.unified_event_bus")

    class UnifiedEventBus:
        def subscribe(self, *a, **k):
            pass

        def publish(self, *a, **k):
            pass

    bus_mod.UnifiedEventBus = UnifiedEventBus
    monkeypatch.setitem(sys.modules, "menace.unified_event_bus", bus_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.unified_event_bus", bus_mod)

    cbi = types.ModuleType("menace.coding_bot_interface")

    def self_coding_managed(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def deco(cls):
            return cls

        return deco

    cbi.self_coding_managed = self_coding_managed
    monkeypatch.setitem(sys.modules, "menace.coding_bot_interface", cbi)
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", cbi)

    db_mod = types.ModuleType("menace.data_bot")

    class DataBot:
        def __init__(self, *a, event_bus=None, **k):
            self.event_bus = event_bus
            self.logged: dict[str, list[tuple[float, float]]] = {}

        def roi(self, bot):
            return self.logged.get(bot, [(0.0, 0.0)])[-1][0]

        def average_errors(self, bot):
            return self.logged.get(bot, [(0.0, 0.0)])[-1][1]

        def record_metrics(self, bot, roi, errors, tests_failed=0.0):
            self.logged.setdefault(bot, []).append((float(roi), float(errors)))

        def check_degradation(self, bot, roi, errors, test_failures=0.0):
            t = self.reload_thresholds(bot)
            degraded = roi <= t.roi_drop or errors >= t.error_threshold
            if degraded and self.event_bus:
                self.event_bus.publish("data:threshold_breach", {"bot": bot})
            return degraded

        def reload_thresholds(self, bot=None):
            from menace.self_coding_thresholds import get_thresholds

            t = get_thresholds(bot)
            return types.SimpleNamespace(
                roi_drop=t.roi_drop,
                error_threshold=t.error_increase,
                test_failure_threshold=t.test_failure_increase,
            )

    db_mod.DataBot = DataBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", db_mod)

    text_pre = types.ModuleType("vector_service.text_preprocessor")

    def _gen(*a, **k):
        return ""

    text_pre.generalise = _gen
    text_pre.get_config = lambda *a, **k: None

    class PreprocessingConfig:
        pass

    text_pre.PreprocessingConfig = PreprocessingConfig
    monkeypatch.setitem(sys.modules, "vector_service.text_preprocessor", text_pre)

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
    manager = types.SimpleNamespace(manager_generate_helper=lambda *a, **k: None)
    reviewer = ar.AutomatedReviewer(
        bot_db=db, escalation_manager=esc, context_builder=builder, manager=manager
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

    manager = types.SimpleNamespace(manager_generate_helper=lambda *a, **k: None)
    reviewer = ar.AutomatedReviewer(
        bot_db=DB(), escalation_manager=Escalator(), context_builder=builder, manager=manager
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

    manager = types.SimpleNamespace(manager_generate_helper=lambda *a, **k: None)
    import context_builder_util as cbu

    def _refresh(builder):
        builder.refresh_db_weights()

    monkeypatch.setattr(cbu, "ensure_fresh_weights", _refresh)
    monkeypatch.setattr(ar, "ensure_fresh_weights", _refresh)
    with pytest.raises(RuntimeError):
        ar.AutomatedReviewer(
            context_builder=BadBuilder(),
            bot_db=DummyDB(),
            escalation_manager=DummyEscalation(),
            manager=manager,
        )


def test_reload_thresholds_autoreviewer(monkeypatch):
    _stub_vector_service(monkeypatch)
    from menace.data_bot import DataBot

    bot = DataBot()
    t = bot.reload_thresholds("AutomatedReviewer")
    assert t.roi_drop == -0.1
    assert t.error_threshold == 1.0
    assert t.test_failure_threshold == 0.0


def test_databot_records_and_breach(monkeypatch, tmp_path):
    _stub_vector_service(monkeypatch)
    import menace.automated_reviewer as ar
    import vector_service
    from menace.data_bot import DataBot

    class Bus:
        def __init__(self):
            self.published: list[tuple[str, dict]] = []

        def publish(self, topic, payload):
            self.published.append((topic, payload))

        def subscribe(self, *a, **k):
            pass

    bus = Bus()
    data_bot = DataBot(event_bus=bus)
    monkeypatch.setattr(ar, "data_bot", data_bot)

    builder = vector_service.ContextBuilder(
        bot_db="bots.db", code_db="code.db", error_db="errors.db", workflow_db="workflows.db"
    )
    manager = types.SimpleNamespace(manager_generate_helper=lambda *a, **k: None)
    reviewer = ar.AutomatedReviewer(
        context_builder=builder, bot_db=DummyDB(), escalation_manager=DummyEscalation(), manager=manager
    )

    monkeypatch.setattr(data_bot, "roi", lambda _b: 10.0)
    monkeypatch.setattr(data_bot, "average_errors", lambda _b: 0.0)
    reviewer.handle({"bot_id": "42", "severity": "critical"})

    monkeypatch.setattr(data_bot, "roi", lambda _b: 0.0)
    monkeypatch.setattr(data_bot, "average_errors", lambda _b: 5.0)
    reviewer.handle({"bot_id": "42", "severity": "critical"})

    assert len(data_bot.logged["42"]) == 2
    assert any(
        topic == "data:threshold_breach" and payload["bot"] == "42"
        for topic, payload in bus.published
    )
