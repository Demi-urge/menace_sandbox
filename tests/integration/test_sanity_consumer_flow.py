import json
import sys
import types

from db_router import init_db_router


def test_watchdog_anomaly_reaches_consumer(monkeypatch, tmp_path):
    """Stripe watchdog anomaly flows through SanityConsumer."""

    # Use temporary database paths and dynamic path resolution
    init_db_router("ba", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))

    dynamic_stub = types.SimpleNamespace(
        resolve_path=lambda p: tmp_path / p,
        resolve_dir=lambda p: (tmp_path / p),
        get_project_root=lambda: tmp_path,
        get_project_roots=lambda: [tmp_path],
    )
    monkeypatch.setitem(sys.modules, "dynamic_path_router", dynamic_stub)

    # Stub heavy discrepancy_db dependency before imports
    discrepancy_stub = types.ModuleType("discrepancy_db")

    class _DummyDB:
        def __init__(self, *a, **k):
            pass

    discrepancy_stub.DiscrepancyDB = _DummyDB
    discrepancy_stub.DiscrepancyRecord = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "discrepancy_db", discrepancy_stub)

    # Stub additional heavy dependencies for SanityConsumer import
    engine_stub = types.ModuleType("self_coding_engine")

    class _BaseEngine:
        def __init__(self, *a, **k):
            pass

    engine_stub.SelfCodingEngine = _BaseEngine
    monkeypatch.setitem(sys.modules, "self_coding_engine", engine_stub)

    fls_stub = types.ModuleType("failure_learning_system")

    class _FLS:
        def __init__(self, *a, **k):
            pass

    fls_stub.DiscrepancyDB = _FLS
    monkeypatch.setitem(sys.modules, "failure_learning_system", fls_stub)

    feedback_stub = types.ModuleType("sanity_feedback")

    class _Feedback:
        def __init__(self, engine, outcome_db=None):
            self.engine = engine
            self.outcome_db = outcome_db

    feedback_stub.SanityFeedback = _Feedback
    monkeypatch.setitem(sys.modules, "sanity_feedback", feedback_stub)

    # Minimal event bus implementation for menace_sanity_layer import fallback
    (tmp_path / "unified_event_bus.py").write_text(
        "class UnifiedEventBus:\n"
        "    def __init__(self):\n"
        "        self.handlers = []\n"
        "    def subscribe(self, topic, cb):\n"
        "        self.handlers.append((topic, cb))\n"
        "    def publish(self, topic, event):\n"
        "        for t, cb in list(self.handlers):\n"
        "            if t == topic:\n"
        "                cb(topic, event)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    # Reload modules with stubbed paths
    monkeypatch.delitem(sys.modules, "stripe_watchdog", raising=False)
    monkeypatch.delitem(sys.modules, "menace_sanity_layer", raising=False)
    monkeypatch.delitem(sys.modules, "billing.sanity_consumer", raising=False)
    import menace_sanity_layer as msl  # type: ignore
    import stripe_watchdog as sw  # type: ignore
    from billing import sanity_consumer as sc  # type: ignore

    # Stub logging and anomaly trail writes
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)

    # Shared event bus
    bus = msl.UnifiedEventBus()
    msl._EVENT_BUS = bus

    # Capture published events
    published: list[dict] = []
    bus.subscribe("billing.anomaly", lambda _t, e: published.append(e))

    # Stub GPT memory manager
    class DummyMemory:
        def __init__(self) -> None:
            self.entries: list[tuple[str, dict, list[str]]] = []

        def log_interaction(self, instruction, content, *, tags=None):
            self.entries.append((instruction, json.loads(content), tags or []))

    memory = DummyMemory()
    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: memory)

    # Stub discrepancy DB used by SanityConsumer
    records: list[object] = []

    class DummyOutcomeDB:
        def add(self, rec):
            records.append(rec)

    outcome_db = DummyOutcomeDB()

    class DummyRecord:
        def __init__(self, message, metadata):
            self.message = message
            self.metadata = metadata

    monkeypatch.setattr(sc, "DiscrepancyDB", lambda: outcome_db)
    monkeypatch.setattr(sc, "DiscrepancyRecord", DummyRecord)

    # Stub self-coding engine
    class DummyEngine:
        def __init__(self, *a, **k) -> None:
            self.updates: list[dict] = []

        def update_generation_params(self, meta):
            self.updates.append(meta)
    monkeypatch.setattr(sc, "SelfCodingEngine", DummyEngine)

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

    consumer = sc.SanityConsumer(event_bus=bus, context_builder=DummyBuilder())

    # Emit anomaly
    record = {"type": "missing_charge", "id": "ch_1", "amount": 5}
    monkeypatch.setattr(sw, "record_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.menace_sanity_layer, "record_payment_anomaly", lambda *a, **k: None)
    monkeypatch.setattr(sw, "load_api_key", lambda: None)
    sw._emit_anomaly(record, False, False)

    # Sanity layer persisted and published event
    anomalies = msl.list_anomalies()
    assert anomalies and anomalies[0]["event_type"] == "missing_charge"
    assert published and published[0]["event_type"] == "missing_charge"

    # Memory received corrective instruction twice (watchdog + consumer)
    assert len(memory.entries) == 2
    assert memory.entries[0][0] == msl.EVENT_TYPE_INSTRUCTIONS["missing_charge"]
    assert "patch_id" in memory.entries[1][1]["metadata"]

    # Generation parameters updated by consumer
    assert consumer._engine.updates and consumer._engine.updates[0]["id"] == "ch_1"

    # Discrepancy DB recorded the anomaly
    assert records and records[0].message == "missing_charge"
