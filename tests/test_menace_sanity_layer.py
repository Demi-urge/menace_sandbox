import json
import os
import sys
import types
from typing import Any

import ast
from pathlib import Path

import pytest
from dynamic_path_router import resolve_path

sys.modules.setdefault(
    "vector_service", types.SimpleNamespace(CognitionLayer=lambda: None)
)

import menace_sanity_layer as msl  # noqa: E402


def _severity_map() -> dict[str, float]:
    path = Path(resolve_path("stripe_watchdog.py"))
    mod = ast.parse(path.read_text())
    for node in mod.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SEVERITY_MAP":
            return ast.literal_eval(node.value)
    return {}


SEVERITY_MAP = _severity_map()
SEVERITY_KEYS = sorted(SEVERITY_MAP.keys())


@pytest.mark.parametrize("event_type", SEVERITY_KEYS)
def test_anomaly_instruction_returns_mapping(event_type):
    expected = msl.EVENT_TYPE_INSTRUCTIONS[event_type]
    instruction = msl._anomaly_instruction(event_type, {}, expected)
    assert instruction == expected


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


@pytest.mark.parametrize("event_type", SEVERITY_KEYS)
def test_record_event_uses_mapped_instruction(event_type, monkeypatch):
    calls: list[tuple[str, dict, list[str]]] = []

    class DummyMemory:
        def log_interaction(self, instruction, content, *, tags=None):
            calls.append((instruction, json.loads(content), tags or []))

    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: DummyMemory())

    msl.record_event(event_type, {})

    assert calls, "memory logging not invoked"
    instruction, payload, tags = calls[0]
    assert instruction == msl.EVENT_TYPE_INSTRUCTIONS[event_type]
    assert msl.FEEDBACK in tags and msl.ERROR_FIX in tags and event_type in tags


@pytest.mark.parametrize("event_type", SEVERITY_KEYS)
def test_record_payment_anomaly_uses_mapped_instruction(event_type, monkeypatch):
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

    instruction = msl.EVENT_TYPE_INSTRUCTIONS[event_type]
    msl.record_payment_anomaly(
        event_type,
        {"foo": "bar"},
        instruction,
        severity=SEVERITY_MAP[event_type],
    )

    assert db_calls and db_calls[0][0] == event_type
    assert mem_calls and mem_calls[0][0] == instruction


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

    class DummyBus:
        def __init__(self) -> None:
            self.handlers: list[tuple[str, object]] = []

        def subscribe(self, topic: str, callback):  # noqa: D401
            self.handlers.append((topic, callback))

        def publish(self, topic: str, event: object) -> None:  # noqa: D401
            for t, cb in self.handlers:
                if t == topic:
                    cb(topic, event)

    bus = DummyBus()
    msl._EVENT_BUS = bus

    def _handler(_topic, event):
        events.append(event)

    bus.subscribe("billing.anomaly", _handler)

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


def test_record_billing_event_persists_and_logs(monkeypatch, tmp_path):
    calls: dict[str, Any] = {}

    class DummyDB:
        def add(self, rec):  # noqa: D401
            calls["rec"] = rec

    class DummyMemory:
        def log_interaction(self, prompt, response, *, tags=None):
            calls["mem"] = (prompt, json.loads(response), tags or [])

        def retrieve(self, *_args, **_kwargs):  # pragma: no cover - unused
            return []

    class DummyEngine:
        def __init__(self) -> None:
            self.updated: dict | None = None

        def update_generation_params(self, meta):
            self.updated = meta

    resolve_path("cfg.json")
    cfg = tmp_path / f"cfg{os.extsep}json"
    cfg.write_text("{}")

    monkeypatch.setattr(msl, "_BILLING_EVENT_DB", DummyDB())
    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: DummyMemory())
    monkeypatch.setattr(msl, "DiscrepancyRecord", lambda **kw: types.SimpleNamespace(**kw))

    engine = DummyEngine()

    msl.record_billing_event(
        "overcharge",
        {"amount": 7, "config_updates": {"threshold": 2}},
        "Avoid duplicate charges",
        config_path=cfg,
        self_coding_engine=engine,
    )

    assert calls["rec"].message == "overcharge"
    assert calls["mem"][0] == "Avoid duplicate charges"
    assert msl.FEEDBACK in calls["mem"][2]
    assert json.loads(cfg.read_text())["threshold"] == 2
    assert engine.updated and engine.updated["amount"] == 7


def test_fetch_recent_billing_issues(monkeypatch):
    class DummyRecord:
        def __init__(self, prompt):
            self.prompt = prompt

    class DummyMemory:
        def retrieve(self, *_args, **_kwargs):
            return [DummyRecord("A"), DummyRecord("B")]

    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: DummyMemory())
    assert msl.fetch_recent_billing_issues(2) == ["A", "B"]


def test_anomaly_deduplication(monkeypatch):
    """Rapid duplicate anomalies are forwarded only once per window."""

    events: list[dict] = []

    class DummyBus:
        def publish(self, _topic, event):
            events.append(event)

    monkeypatch.setattr(msl, "_EVENT_BUS", DummyBus())
    monkeypatch.setattr(
        msl.db_router,
        "GLOBAL_ROUTER",
        types.SimpleNamespace(execute_and_log=lambda *a, **k: []),
    )
    msl._ANOMALY_CACHE.clear()
    msl._SUPPRESSION_SETTINGS.update(
        {"window_seconds": 60.0, "max_occurrences": 1.0, "severity_threshold": 0.0}
    )

    for _ in range(3):
        msl.record_billing_anomaly("dup_event", {"id": 1}, severity=1.0)

    assert len(events) == 1
