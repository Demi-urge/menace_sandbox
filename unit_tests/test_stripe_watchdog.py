"""Unit tests for the simplified Stripe watchdog."""

from __future__ import annotations

import json
from types import SimpleNamespace
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.modules.setdefault("vector_service", SimpleNamespace(CognitionLayer=lambda: None))

import pytest  # noqa: E402

import stripe_watchdog as sw  # noqa: E402


@pytest.fixture
def capture(monkeypatch, tmp_path):
    """Capture audit events and ensure logs go to a temp file."""

    events: list = []
    monkeypatch.setattr(
        sw.audit_logger, "log_event", lambda et, data: events.append((et, data))
    )
    monkeypatch.setattr(
        sw.alert_dispatcher,
        "dispatch_alert",
        lambda *a, **k: events.append(("alert", a, k)),
    )
    monkeypatch.setattr(
        sw.menace_sanity_layer, "record_payment_anomaly", lambda *a, **k: None
    )
    monkeypatch.setattr(sw, "record_billing_event", lambda *a, **k: None)

    class DummyTrail:
        def __init__(self, path):
            self.path = path

        def record(self, entry):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setattr(sw, "ANOMALY_LOG", audit_file)
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", DummyTrail(audit_file))
    return events


def test_orphan_charge_logged(capture, monkeypatch):
    events = capture

    record_calls: list = []

    def fake_record_event(event_type, metadata, **kwargs):
        record_calls.append((event_type, metadata, kwargs))

    monkeypatch.setattr(sw, "record_event", fake_record_event)
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)
    billing_calls: list[tuple[str, float]] = []
    payment_calls: list[tuple[str, float]] = []

    def fake_billing(event_type, metadata, *, severity=1.0, **kwargs):
        billing_calls.append((event_type, severity))

    def fake_payment(event_type, metadata, instruction=None, *, severity=1.0, **kwargs):
        payment_calls.append((event_type, severity))

    monkeypatch.setattr(sw, "record_billing_anomaly", fake_billing)
    monkeypatch.setattr(sw.menace_sanity_layer, "record_payment_anomaly", fake_payment)

    charges = [
        {"id": "ch_known", "created": 1, "amount": 1000, "status": "succeeded"},
        {"id": "ch_orphan", "created": 2, "amount": 2000, "status": "succeeded"},
    ]
    ledger = [{"id": "ch_known", "timestamp_ms": 1000}]

    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda api_key, s, e: charges)
    monkeypatch.setattr(sw, "load_local_ledger", lambda s, e: ledger)
    monkeypatch.setattr(sw, "load_billing_logs", lambda s, e, action="charge": [])
    monkeypatch.setattr(sw, "check_webhook_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(sw, "DiscrepancyDB", None)
    monkeypatch.setattr(sw, "DiscrepancyRecord", None)

    engine = object()
    telemetry = object()
    anomalies = sw.check_events(
        self_coding_engine=engine, telemetry_feedback=telemetry
    )

    assert anomalies and anomalies[0]["id"] == "ch_orphan"
    assert events and events[0][0] == "stripe_anomaly"
    assert record_calls and record_calls[0][2]["self_coding_engine"] is engine
    assert record_calls[0][2]["telemetry_feedback"] is telemetry
    assert billing_calls and billing_calls[0][1] == sw.SEVERITY_MAP["missing_charge"]
    assert payment_calls and payment_calls[0][1] == sw.SEVERITY_MAP["missing_charge"]


def test_record_billing_event_called(capture, monkeypatch):
    charges = [
        {"id": "ch_new", "created": 1, "amount": 1000, "status": "succeeded"}
    ]
    ledger: list[dict] = []

    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda api_key, s, e: charges)
    monkeypatch.setattr(sw, "load_local_ledger", lambda s, e: ledger)
    monkeypatch.setattr(sw, "load_billing_logs", lambda s, e, action="charge": [])
    monkeypatch.setattr(sw, "check_webhook_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(sw, "DiscrepancyDB", None)
    monkeypatch.setattr(sw, "DiscrepancyRecord", None)

    fake_stripe = SimpleNamespace(
        Account=SimpleNamespace(retrieve=lambda api_key=None: {"id": "acct_test"})
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)

    calls: list = []

    def fake_record(event_type, metadata, instruction, **kwargs):
        calls.append((event_type, metadata, instruction))

    monkeypatch.setattr(sw, "record_billing_event", fake_record)

    sw.check_events()

    assert calls, "record_billing_event was not called"
    event_type, metadata, instruction = calls[0]
    assert event_type == "missing_charge"
    assert metadata.get("stripe_account") == "acct_test"
    assert "timestamp" in metadata
    assert (
        instruction
        == sw.menace_sanity_layer.EVENT_TYPE_INSTRUCTIONS["missing_charge"]
    )


def test_distinct_instructions_per_anomaly_type(capture, monkeypatch):
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda *a, **k: [])
    monkeypatch.setattr(sw, "fetch_recent_refunds", lambda *a, **k: [])
    monkeypatch.setattr(sw, "fetch_recent_events", lambda *a, **k: [])
    monkeypatch.setattr(sw, "load_local_ledger", lambda *a, **k: [])
    monkeypatch.setattr(sw, "load_billing_logs", lambda *a, **k: [])
    monkeypatch.setattr(sw, "check_webhook_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(sw, "compare_revenue_window", lambda *a, **k: None)
    monkeypatch.setattr(sw, "stripe", None)
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(sw, "DiscrepancyDB", None)
    monkeypatch.setattr(sw, "DiscrepancyRecord", None)
    monkeypatch.setattr(sw, "load_approved_workflows", lambda: set())

    monkeypatch.setattr(
        sw,
        "detect_missing_charges",
        lambda *a, **k: [
            {"type": "missing_charge", "id": "ch"},
            {"type": "missing_refund", "id": "rf"},
        ],
    )
    monkeypatch.setattr(sw, "detect_failed_events", lambda *a, **k: [])

    calls: list[tuple[str, str]] = []

    def fake_record(event_type, metadata, instruction, **kwargs):
        calls.append((event_type, instruction))

    monkeypatch.setattr(sw, "record_billing_event", fake_record)

    sw.check_events()

    assert len(calls) == 2
    instr1 = sw.menace_sanity_layer.EVENT_TYPE_INSTRUCTIONS["missing_charge"]
    instr2 = sw.menace_sanity_layer.EVENT_TYPE_INSTRUCTIONS["missing_refund"]
    assert calls[0] == ("missing_charge", instr1)
    assert calls[1] == ("missing_refund", instr2)
    assert instr1 != instr2


def test_webhook_endpoint_validation(capture, monkeypatch):
    events = capture

    endpoints = [
        {"id": "we_allowed", "url": "https://ok.example.com", "status": "enabled"},
        {"id": "we_bad", "url": "https://bad.example.com", "status": "enabled"},
    ]
    fake_stripe = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda api_key: endpoints)
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)

    unknown = sw.check_webhook_endpoints(api_key="sk", approved=["we_allowed"])

    assert unknown == ["we_bad"]
    assert events and events[0][1]["type"] == "unknown_webhook"


def test_config_updates_written(monkeypatch, tmp_path):
    """record_billing_event persists config updates when anomalies occur."""

    import menace_sanity_layer as msl

    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)
    config_file = tmp_path / "cfg.json"
    monkeypatch.setattr(sw, "CONFIG_PATH", config_file)

    # Ensure record_billing_event writes to our temp file without touching DBs
    monkeypatch.setattr(msl, "_BILLING_EVENT_DB", None)
    monkeypatch.setattr(msl, "_get_gpt_memory", lambda: None)

    anomaly = {
        "type": "missing_charge",
        "id": "ch1",
        "config_updates": {"max_retries": 5},
    }
    monkeypatch.setattr(sw, "detect_missing_charges", lambda *a, **k: [anomaly])
    monkeypatch.setattr(sw, "detect_missing_refunds", lambda *a, **k: [])
    monkeypatch.setattr(sw, "detect_failed_events", lambda *a, **k: [])
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda *a, **k: [])
    monkeypatch.setattr(sw, "fetch_recent_refunds", lambda *a, **k: [])
    monkeypatch.setattr(sw, "fetch_recent_events", lambda *a, **k: [])
    monkeypatch.setattr(sw, "load_local_ledger", lambda *a, **k: [])
    monkeypatch.setattr(sw, "load_billing_logs", lambda *a, **k: [])
    monkeypatch.setattr(sw, "check_webhook_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(sw, "compare_revenue_window", lambda *a, **k: None)
    monkeypatch.setattr(sw, "DiscrepancyDB", None)
    monkeypatch.setattr(sw, "DiscrepancyRecord", None)
    fake_stripe = SimpleNamespace(Account=SimpleNamespace(retrieve=lambda **k: {"id": "acct"}))
    monkeypatch.setattr(sw, "stripe", fake_stripe)

    engine_updates: list = []
    engine = SimpleNamespace(
        update_generation_params=lambda meta: engine_updates.append(meta)
    )

    sw.check_events(self_coding_engine=engine)

    assert json.loads(config_file.read_text()) == {"max_retries": 5}
    assert engine_updates and engine_updates[0]["config_updates"]["max_retries"] == 5


def test_main_updates_last_run(monkeypatch, tmp_path):
    last_run = tmp_path / "last.txt"
    last_run.write_text("100")
    monkeypatch.setattr(sw, "_LAST_RUN_FILE", last_run)

    monkeypatch.setattr(sw, "load_api_key", lambda: "sk")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda *a, **k: [])
    monkeypatch.setattr(sw, "fetch_recent_refunds", lambda *a, **k: [])
    monkeypatch.setattr(sw, "fetch_recent_events", lambda *a, **k: [])
    monkeypatch.setattr(sw, "load_local_ledger", lambda *a, **k: [])
    monkeypatch.setattr(sw, "load_billing_logs", lambda *a, **k: [])
    monkeypatch.setattr(sw, "detect_missing_charges", lambda *a, **k: [])
    monkeypatch.setattr(sw, "detect_missing_refunds", lambda *a, **k: [])
    monkeypatch.setattr(sw, "detect_failed_events", lambda *a, **k: [])
    monkeypatch.setattr(sw, "check_webhook_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(sw, "compare_revenue_window", lambda *a, **k: None)

    class DummyTrail:
        def __init__(self, path):
            self.path = path

        def record(self, entry):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setattr(sw, "ANOMALY_LOG", audit_file)
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", DummyTrail(audit_file))

    sw.main([])

    assert int(last_run.read_text()) >= 100


def test_detect_missing_charges_id_matching():
    charges = [
        {"id": "ch1", "created": 1, "amount": 1000},
        {"id": "ch2", "created": 2, "amount": 2000},
        {"id": "ch3", "created": 3, "amount": 3000},
    ]
    ledger = [{"id": "ch1"}]
    billing_logs = [{"stripe_id": "ch2"}]

    anomalies = sw.detect_missing_charges(charges, ledger, billing_logs)
    assert len(anomalies) == 1 and anomalies[0]["id"] == "ch3"


def test_detect_missing_refunds_id_matching():
    refunds = [
        {"id": "re1", "amount": 100, "charge": "c1"},
        {"id": "re2", "amount": 200, "charge": "c2"},
        {"id": "re3", "amount": 300, "charge": "c3"},
    ]
    ledger = [{"id": "re1", "action_type": "refund"}]
    billing_logs = [{"stripe_id": "re2"}]

    anomalies = sw.detect_missing_refunds(refunds, ledger, billing_logs)
    assert len(anomalies) == 1 and anomalies[0]["refund_id"] == "re3"


def test_detect_failed_events_id_matching():
    events = [
        {"id": "ev1", "type": "charge.failed"},
        {"id": "ev2", "type": "charge.failed"},
        {"id": "ev3", "type": "charge.failed"},
    ]
    ledger = [{"id": "ev1", "action_type": "failed"}]
    billing_logs = [{"stripe_id": "ev2"}]

    anomalies = sw.detect_failed_events(events, ledger, billing_logs)
    assert len(anomalies) == 1 and anomalies[0]["event_id"] == "ev3"


def _has_alert(events):
    return any(e[0] == "alert" for e in events)


def test_detect_missing_charges_workflow_approval(capture):
    events = capture
    charges = [{"id": "ch1", "created": 1, "amount": 1000}]
    ledger = [{"id": "ch1"}]
    logs = [{"stripe_id": "ch1", "bot_id": "bot_a"}]

    anomalies = sw.detect_missing_charges(
        charges, ledger, logs, approved_workflows=["bot_b"]
    )
    assert anomalies and anomalies[0]["type"] == "unapproved_workflow"
    assert _has_alert(events)

    events.clear()
    anomalies = sw.detect_missing_charges(
        charges, ledger, logs, approved_workflows=["bot_a"]
    )
    assert anomalies == []
    assert not events


def test_detect_missing_refunds_workflow_approval(capture):
    events = capture
    refunds = [{"id": "re1", "amount": 100, "charge": "c1"}]
    ledger = [{"id": "re1", "action_type": "refund"}]
    logs = [{"stripe_id": "re1", "bot_id": "bot_a"}]

    anomalies = sw.detect_missing_refunds(
        refunds, ledger, logs, approved_workflows=["bot_b"]
    )
    assert anomalies and anomalies[0]["type"] == "unapproved_workflow"
    assert _has_alert(events)

    events.clear()
    anomalies = sw.detect_missing_refunds(
        refunds, ledger, logs, approved_workflows=["bot_a"]
    )
    assert anomalies == []
    assert not events


def test_detect_failed_events_workflow_approval(capture):
    events = capture
    failed_events = [{"id": "ev1", "type": "charge.failed"}]
    ledger = [{"id": "ev1", "action_type": "failed"}]
    logs = [{"stripe_id": "ev1", "bot_id": "bot_a"}]

    anomalies = sw.detect_failed_events(
        failed_events, ledger, logs, approved_workflows=["bot_b"]
    )
    assert anomalies and anomalies[0]["type"] == "unapproved_workflow"
    assert _has_alert(events)

    events.clear()
    anomalies = sw.detect_failed_events(
        failed_events, ledger, logs, approved_workflows=["bot_a"]
    )
    assert anomalies == []
    assert not events


def _capture_record_event(monkeypatch):
    calls: list = []
    monkeypatch.setattr(sw, "record_event", lambda et, md, **kw: calls.append((et, md)))
    monkeypatch.setattr(sw, "SANITY_LAYER_FEEDBACK_ENABLED", True)
    return calls


def test_charge_account_mismatch_triggers_anomaly(capture, monkeypatch):
    events = capture
    calls = _capture_record_event(monkeypatch)
    charges = [{"id": "ch1", "account": "acct_actual"}]
    ledger = [{"id": "ch1"}]
    anomalies = sw.detect_missing_charges(
        charges,
        ledger,
        [],
        expected_account_id="acct_expected",
    )
    assert anomalies and anomalies[0]["type"] == "account_mismatch"
    assert events and events[0][1]["type"] == "account_mismatch"
    assert calls and calls[0][0] == "account_mismatch"
    meta = calls[0][1]
    assert meta["account_id"] == "acct_actual"
    assert meta["expected_account_id"] == "acct_expected"


def test_refund_account_mismatch_triggers_anomaly(capture, monkeypatch):
    events = capture
    calls = _capture_record_event(monkeypatch)
    refunds = [{"id": "re1", "amount": 100, "charge": "c1", "account": "acct_actual"}]
    ledger = [{"id": "re1", "action_type": "refund"}]
    anomalies = sw.detect_missing_refunds(
        refunds,
        ledger,
        [],
        expected_account_id="acct_expected",
    )
    assert anomalies and anomalies[0]["type"] == "account_mismatch"
    assert events and events[0][1]["type"] == "account_mismatch"
    assert calls and calls[0][0] == "account_mismatch"
    meta = calls[0][1]
    assert meta["account_id"] == "acct_actual"
    assert meta["expected_account_id"] == "acct_expected"


def test_event_account_mismatch_triggers_anomaly(capture, monkeypatch):
    events = capture
    calls = _capture_record_event(monkeypatch)
    failed_events = [
        {"id": "ev1", "type": "charge.failed", "account": "acct_actual"}
    ]
    ledger = [{"id": "ev1", "action_type": "failed"}]
    anomalies = sw.detect_failed_events(
        failed_events,
        ledger,
        [],
        expected_account_id="acct_expected",
    )
    assert anomalies and anomalies[0]["type"] == "account_mismatch"
    assert events and events[0][1]["type"] == "account_mismatch"
    assert calls and calls[0][0] == "account_mismatch"
    meta = calls[0][1]
    assert meta["account_id"] == "acct_actual"
    assert meta["expected_account_id"] == "acct_expected"
