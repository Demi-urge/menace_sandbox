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
    assert mem_calls and mem_calls[0][0] == "handle it"
    assert mem_calls[0][1]["event_type"] == "test_event"
    assert mem_calls[0][1]["metadata"]["foo"] == "bar"


def test_record_billing_anomaly_persists_and_publishes(monkeypatch, tmp_path):
    from db_router import init_db_router

    init_db_router("ba", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    published: list[dict] = []
    monkeypatch.setattr(msl, "publish_anomaly", lambda ev: published.append(ev))

    msl.record_billing_anomaly(
        "overcharge", {"amount": 5}, severity=3.0, source_workflow="wf-1"
    )

    anomalies = msl.list_anomalies()
    assert anomalies and anomalies[0]["event_type"] == "overcharge"
    assert published and published[0]["event_type"] == "overcharge"
