import json

import menace_sanity_layer as msl


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
    assert mem_calls and mem_calls[0][0] == "handle it"
    assert mem_calls[0][1]["event_type"] == "test_event"
    assert mem_calls[0][1]["metadata"]["foo"] == "bar"
