import json
import sys
import types

sys.modules.setdefault(
    "vector_service", types.SimpleNamespace(CognitionLayer=lambda: None)
)

import menace_sanity_layer as msl  # noqa: E402


def test_record_payment_anomaly_writes_db_and_memory(monkeypatch):
    db_calls: list[tuple[str, float, dict]] = []
    mem_calls: list[tuple[str, dict, list[str]]] = []

    class DummyDB:
        def log_detection(self, event_type, severity, payload):
            db_calls.append((event_type, severity, json.loads(payload)))

    class DummyMemory:
        def log_interaction(self, instruction, content, *, tags=None):
            mem_calls.append((instruction, json.loads(content), tags or []))

    monkeypatch.setattr(msl, "_DISCREPANCY_DB", DummyDB())
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", DummyMemory())
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)

    msl.record_payment_anomaly(
        "test_event",
        {"foo": "bar"},
        "handle it",
        severity=2.5,
        write_codex=False,
        export_training=False,
    )

    assert db_calls == [
        ("test_event", 2.5, {"foo": "bar", "write_codex": False, "export_training": False})
    ]
    assert mem_calls and mem_calls[0][0] == "Avoid handle it."
    assert mem_calls[0][1]["event_type"] == "test_event"
    assert mem_calls[0][1]["metadata"]["foo"] == "bar"


def test_watchdog_anomaly_updates_db_memory_and_event_bus(monkeypatch, tmp_path):
    """Simulate Stripe watchdog anomaly end-to-end."""

    from db_router import init_db_router

    init_db_router("ba", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))

    # Ensure temporary paths are used by the watchdog module
    monkeypatch.setitem(
        sys.modules,
        "dynamic_path_router",
        types.SimpleNamespace(resolve_path=lambda p: tmp_path / p),
    )
    monkeypatch.delitem(sys.modules, "stripe_watchdog", raising=False)
    import stripe_watchdog as sw  # noqa: WPS433

    # Avoid file I/O from watchdog and sanity layer
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(msl, "_DISCREPANCY_DB", None)
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", None)

    events: list[dict] = []
    msl._EVENT_BUS = msl.UnifiedEventBus()

    def _handler(_topic, event):
        events.append(event)

    msl._EVENT_BUS.subscribe("billing.anomaly", _handler)

    class DummyMM:
        def __init__(self) -> None:
            self.stored: list[tuple[str, dict, str]] = []

        def query(self, key, limit):  # noqa: D401
            return []

        def store(self, key, data, tags=""):
            self.stored.append((key, data, tags))

    mm = DummyMM()
    msl._MEMORY_MANAGER = mm

    record = {"type": "overcharge", "id": "ch_1", "amount": 5}
    sw._emit_anomaly(record, False, False)

    anomalies = msl.list_anomalies()
    assert anomalies and anomalies[0]["event_type"] == "overcharge"
    assert mm.stored and mm.stored[0][0] == "billing:overcharge"
    assert events and events[0]["event_type"] == "overcharge"
