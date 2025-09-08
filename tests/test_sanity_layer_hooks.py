import json
import sys
import types

import pytest
import yaml


sys.modules.setdefault(
    "vector_service", types.SimpleNamespace(CognitionLayer=lambda: None)
)

import menace_sanity_layer as msl  # noqa: E402


def test_emit_anomaly_triggers_record_event(monkeypatch):
    import stripe_watchdog as sw

    calls = []

    def fake_record_event(event_type, metadata, **kwargs):
        calls.append((event_type, metadata, kwargs))

    monkeypatch.setattr(msl, "record_event", fake_record_event)
    monkeypatch.setattr(sw, "record_event", fake_record_event)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(sw, "record_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.menace_sanity_layer, "record_payment_anomaly", lambda *a, **k: None)
    monkeypatch.setattr(sw, "load_api_key", lambda: None)

    engine = object()
    telemetry = object()
    sw._emit_anomaly(
        {"type": "missing_charge", "charge_id": "ch_1"},
        False,
        False,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
        context_builder=types.SimpleNamespace(build=lambda *a, **k: ""),
    )

    assert calls and calls[0][0] == "missing_charge"
    assert calls[0][1]["charge_id"] == "ch_1"
    assert calls[0][2]["self_coding_engine"] is engine
    assert calls[0][2]["telemetry_feedback"] is telemetry


def test_record_payment_anomaly_telemetry_feedback(monkeypatch):
    class DummyTelemetry:
        def __init__(self):
            self.events: list[tuple[str, dict]] = []
            self.checked = 0

        def record_event(self, event_type, metadata):
            self.events.append((event_type, metadata))

        def check(self):
            self.checked += 1

    telemetry = DummyTelemetry()

    monkeypatch.setattr(msl, "_DISCREPANCY_DB", None)
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", None)
    monkeypatch.setattr(msl, "_get_memory_manager", lambda: None)
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)

    msl.record_payment_anomaly(
        "missing_charge",
        {"charge_id": "ch_1"},
        telemetry_feedback=telemetry,
    )

    assert telemetry.events and telemetry.events[0][0] == "missing_charge"
    assert telemetry.events[0][1]["charge_id"] == "ch_1"
    assert telemetry.checked == 1


def _stub_unified_event_bus(monkeypatch, tmp_path):
    import dynamic_path_router
    from dynamic_path_router import resolve_path

    module_name = resolve_path("unified_event_bus.py").name
    stub_path = tmp_path / module_name
    stub_path.write_text("class UnifiedEventBus:\n    pass\n")

    orig_resolve = resolve_path

    def fake_resolve(name, root=None):
        if name == module_name:
            return stub_path
        try:
            return orig_resolve(name, root)
        except TypeError:
            return orig_resolve(name)

    monkeypatch.setattr(dynamic_path_router, "resolve_path", fake_resolve)


def test_alert_mismatch_invokes_record_event(monkeypatch, tmp_path):
    from tests.test_stripe_billing_router_logging import _import_module

    _stub_unified_event_bus(monkeypatch, tmp_path)
    sbr = _import_module(monkeypatch, tmp_path)

    calls: list[tuple[str, dict]] = []

    def fake_record_event(event_type, metadata, **kwargs):
        calls.append((event_type, metadata))

    monkeypatch.setattr(msl, "record_event", fake_record_event)
    monkeypatch.setattr(sbr, "record_payment_anomaly", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "record_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "log_critical_discrepancy", lambda *a, **k: None)
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "log_billing_event", lambda *a, **k: None)
    import evolution_lock_flag

    monkeypatch.setattr(evolution_lock_flag, "trigger_lock", lambda *a, **k: None)

    sbr._alert_mismatch("bot1", "acct_mismatch")

    assert calls == [
        (
            "account_mismatch",
            {"bot_id": "bot1", "destination_account": "acct_mismatch", "amount": None},
        )
    ]


def test_record_event_logs_instruction_and_tags(monkeypatch):
    calls = []

    class DummyMemory:
        def log_interaction(self, instruction, content, *, tags=None):
            calls.append((instruction, json.loads(content), tags or []))

    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: DummyMemory())

    msl.record_event("missing_charge", {"charge_id": "ch_2"})

    assert calls
    instruction, payload, tags = calls[0]
    assert msl.EVENT_TYPE_INSTRUCTIONS["missing_charge"] in instruction
    assert payload == {
        "event_type": "missing_charge",
        "metadata": {"charge_id": "ch_2"},
    }
    assert msl.FEEDBACK in tags and msl.ERROR_FIX in tags and "missing_charge" in tags


@pytest.mark.parametrize(
    "event_type, metadata, hint",
    [
        ("missing_charge", {"charge_id": "ch"}, {"block_unlogged_charges": True}),
        ("missing_refund", {"refund_id": "re"}, {"block_unlogged_refunds": True}),
        (
            "missing_failure_log",
            {"event_id": "evt"},
            {"log_stripe_failures": True},
        ),
        (
            "unapproved_workflow",
            {"bot_id": "bot"},
            {"enforce_workflow_approval": True},
        ),
        (
            "unknown_webhook",
            {"webhook_id": "wh"},
            {"register_stripe_webhooks": True},
        ),
        (
            "disabled_webhook",
            {"webhook_id": "whd"},
            {"reactivate_stripe_webhook": True},
        ),
        (
            "revenue_mismatch",
            {"expected": 10, "actual": 5},
            {"reconcile_revenue": True},
        ),
        (
            "account_mismatch",
            {"account_id": "acct"},
            {"verify_stripe_account": True},
        ),
        (
            "unauthorized_charge",
            {"charge_id": "ch_unauth"},
            {"block_unauthorized_charges": True},
        ),
        (
            "unauthorized_refund",
            {"refund_id": "re_unauth"},
            {"block_unauthorized_refunds": True},
        ),
        (
            "unauthorized_failure",
            {"event_id": "evt_unauth"},
            {"block_unauthorized_failures": True},
        ),
    ],
)
def test_repeated_anomalies_trigger_param_update(event_type, metadata, hint, monkeypatch):
    mm_storage = {}

    class DummyMM:
        def query(self, key, limit):
            if key in mm_storage:
                return [types.SimpleNamespace(data=json.dumps(mm_storage[key]))]
            return []

        def store(self, key, data, tags=""):
            mm_storage[key] = data

    class DummyEngine:
        def __init__(self):
            self.calls: list[dict] = []

        def update_generation_params(self, meta):
            self.calls.append(meta)

    monkeypatch.setattr(msl, "_get_memory_manager", lambda: DummyMM())
    monkeypatch.setattr(msl, "_DISCREPANCY_DB", None)
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", None)
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)

    msl.refresh_billing_instructions()
    msl._load_instruction_overrides()

    engine = DummyEngine()
    threshold = msl.ANOMALY_THRESHOLDS.get(
        event_type, msl.PAYMENT_ANOMALY_THRESHOLD
    )
    for _ in range(threshold):
        msl.record_payment_anomaly(
            event_type,
            metadata,
            self_coding_engine=engine,
        )

    key = f"billing:{event_type}"
    assert mm_storage[key]["count"] == threshold
    assert engine.calls == [{**hint, "event_type": event_type}]


def test_custom_threshold_and_hint_override(monkeypatch, tmp_path):
    mm_storage: dict[str, dict] = {}

    class DummyMM:
        def query(self, key, limit):
            if key in mm_storage:
                return [types.SimpleNamespace(data=json.dumps(mm_storage[key]))]
            return []

        def store(self, key, data, tags=""):
            mm_storage[key] = data

    class DummyEngine:
        def __init__(self):
            self.calls: list[dict] = []

        def update_generation_params(self, meta):
            self.calls.append(meta)

    cfg = {
        "payment_anomaly_threshold": 5,
        "anomaly_thresholds": {"missing_charge": 2},
        "anomaly_hints": {"missing_charge": {"custom_hint": True}},
    }
    cfg_path = tmp_path / "billing.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    original = msl._INSTRUCTION_PATH
    msl.refresh_billing_instructions(cfg_path)

    monkeypatch.setattr(msl, "_get_memory_manager", lambda: DummyMM())
    monkeypatch.setattr(msl, "_DISCREPANCY_DB", None)
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", None)
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)

    engine = DummyEngine()
    for _ in range(2):
        msl.record_payment_anomaly(
            "missing_charge",
            {"charge_id": "c"},
            self_coding_engine=engine,
        )

    key = "billing:missing_charge"
    assert mm_storage[key]["count"] == 2
    assert engine.calls == [{"custom_hint": True, "event_type": "missing_charge"}]
    msl.refresh_billing_instructions(original)


@pytest.mark.parametrize(
    "event_type, record",
    [
        ("missing_charge", {"type": "missing_charge", "id": "ch_x"}),
        ("missing_refund", {"type": "missing_refund", "refund_id": "re_x"}),
        (
            "missing_failure_log",
            {"type": "missing_failure_log", "event_id": "evt_x"},
        ),
        ("unapproved_workflow", {"type": "unapproved_workflow", "bot_id": "bot"}),
        ("unknown_webhook", {"type": "unknown_webhook", "webhook_id": "wh_x"}),
        ("disabled_webhook", {"type": "disabled_webhook", "webhook_id": "wh_d"}),
        (
            "revenue_mismatch",
            {"type": "revenue_mismatch", "expected": 10, "actual": 5},
        ),
    ],
)
def test_emit_anomaly_records_all_outputs(event_type, record, monkeypatch, tmp_path):
    import stripe_watchdog as sw
    import db_router

    bus_events: list[tuple[str, dict]] = []

    class DummyBus:
        def publish(self, topic: str, event: dict) -> None:  # pragma: no cover - simple
            bus_events.append((topic, event))

    memory_calls: list[tuple[str, dict, list[str]]] = []

    class DummyMemory:
        def log_interaction(self, instruction, content, *, tags=None):
            memory_calls.append((instruction, json.loads(content), tags or []))

    router = db_router.DBRouter("alpha", str(tmp_path), str(tmp_path / "shared.db"))

    def fake_payment(event_type, metadata, instruction, *, severity, **kwargs):
        payment_calls.append((event_type, metadata, instruction, severity))

    payment_calls: list[tuple[str, dict, str, float]] = []

    monkeypatch.setattr(msl.db_router, "GLOBAL_ROUTER", router)
    monkeypatch.setattr(msl, "_EVENT_BUS", DummyBus())
    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: DummyMemory())
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", None)
    monkeypatch.setattr(msl, "_DISCREPANCY_DB", None)
    monkeypatch.setattr(sw.menace_sanity_layer, "record_payment_anomaly", fake_payment)
    monkeypatch.setattr(sw, "record_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(sw, "load_api_key", lambda: None)

    sw._emit_anomaly(
        record, False, False, context_builder=types.SimpleNamespace(build=lambda *a, **k: "")
    )

    rows = router.local_conn.execute(
        "SELECT event_type, metadata, severity FROM billing_anomalies"
    ).fetchall()
    assert rows and rows[0][0] == event_type
    assert json.loads(rows[0][1]) == record
    assert rows[0][2] == sw.SEVERITY_MAP[event_type]

    assert bus_events and bus_events[0][0] == "billing.anomaly"
    event = bus_events[0][1]
    assert event["event_type"] == event_type
    for k, v in record.items():
        assert event["metadata"].get(k) == v
    assert event["severity"] == sw.SEVERITY_MAP[event_type]

    assert memory_calls
    instr, payload, _tags = memory_calls[0]
    assert msl.EVENT_TYPE_INSTRUCTIONS[event_type] in instr
    expected_meta = dict(record)
    expected_meta.pop("type")
    assert payload["event_type"] == event_type
    for k, v in expected_meta.items():
        assert payload["metadata"].get(k) == v

    assert payment_calls and payment_calls[0][2] == msl.EVENT_TYPE_INSTRUCTIONS[event_type]
    assert payment_calls[0][3] == sw.SEVERITY_MAP[event_type]

    router.close()
