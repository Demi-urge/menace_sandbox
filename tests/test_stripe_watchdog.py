"""Tests for :mod:`stripe_watchdog` anomaly detection.

The tests intentionally avoid network access by stubbing out Stripe client
interactions and other optional dependencies. Each scenario verifies that an
anomaly results in an ``audit_logger`` entry and that, when ``write_codex`` is
True, a Codex ``TrainingSample`` is also created.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
import sys
import tempfile
from pathlib import Path

import pytest
from logging.handlers import RotatingFileHandler
import logging
import shutil
import os

from dynamic_path_router import resolve_path
from audit_trail import AuditTrail

# Stub heavy optional dependencies before importing stripe_watchdog
sys.modules.setdefault("vector_service", SimpleNamespace(CognitionLayer=lambda: None))

_TEMP_DIR = tempfile.TemporaryDirectory()
UEB_MODULE = resolve_path("unified_event_bus.py").name
_STUB_UEB = Path(_TEMP_DIR.name) / UEB_MODULE
_STUB_UEB.write_text("class UnifiedEventBus:\n    pass\n")

import dynamic_path_router as _dpr  # noqa: E402
_orig_resolve = _dpr.resolve_path


def _fake_resolve(name, root=None):
    if name == UEB_MODULE:
        return _STUB_UEB
    try:
        return _orig_resolve(name, root)
    except TypeError:
        return _orig_resolve(name)


_dpr.resolve_path = _fake_resolve


import stripe_watchdog as sw  # noqa: E402

BUILDER = SimpleNamespace(build=lambda *a, **k: "")


@pytest.fixture
def capture_anomalies(monkeypatch, tmp_path):
    """Capture emitted audit events, Codex samples and severity info."""

    events: list[tuple[str, dict]] = []
    samples: list[dict] = []
    billing_calls: list[tuple[str, dict, float]] = []
    payment_calls: list[tuple[str, dict, str | None, float]] = []

    monkeypatch.setattr(
        sw.audit_logger,
        "log_event",
        lambda et, data: events.append((et, data)),
    )

    class DummySample:
        def __init__(self, source: str, content: str):
            samples.append({"source": source, "content": content})

    monkeypatch.setattr(sw, "TrainingSample", DummySample)
    log_path = tmp_path / "anomaly.log"
    handler = RotatingFileHandler(str(log_path), maxBytes=1024, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.rotator = lambda src, dst: shutil.move(src, dst)
    handler.namer = lambda name: name
    monkeypatch.setattr(sw, "ANOMALY_LOG", log_path)
    monkeypatch.setattr(sw, "ANOMALY_HANDLER", handler)
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", AuditTrail(str(log_path), handler=handler))
    resolve_path("config/stripe_watchdog.yaml")
    monkeypatch.setattr(sw.alert_dispatcher, "dispatch_alert", lambda *a, **k: None)

    def fake_billing(event_type, metadata, *, severity=1.0, **kwargs):
        billing_calls.append((event_type, metadata, severity))

    def fake_payment(event_type, metadata, instruction, *, severity=1.0, **kwargs):
        payment_calls.append((event_type, metadata, instruction, severity))

    monkeypatch.setattr(sw, "record_billing_anomaly", fake_billing)
    monkeypatch.setattr(sw.menace_sanity_layer, "record_payment_anomaly", fake_payment)
    monkeypatch.setattr(sw, "record_billing_event", lambda *a, **k: None)

    return events, samples, billing_calls, payment_calls


def test_orphan_charge_triggers_audit_and_codex(monkeypatch, capture_anomalies):
    events, samples, billing_calls, payment_calls = capture_anomalies

    ledger = [
        {"id": "ch_logged", "timestamp_ms": 1000},
    ]
    charges = [
        {"id": "ch_logged", "created": 1, "amount": 1000, "status": "succeeded"},
        {"id": "ch_orphan", "created": 2, "amount": 2000, "status": "succeeded"},
    ]

    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda api_key, start, end: charges)
    monkeypatch.setattr(sw, "load_local_ledger", lambda start, end: ledger)
    monkeypatch.setattr(sw, "load_billing_logs", lambda start, end: [])
    monkeypatch.setattr(sw, "check_webhook_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(sw, "DiscrepancyDB", None)
    monkeypatch.setattr(sw, "DiscrepancyRecord", None)

    anomalies = sw.check_events(write_codex=True)

    assert anomalies and anomalies[0]["id"] == "ch_orphan"
    assert anomalies[0]["module"] == sw.BILLING_ROUTER_MODULE
    assert events and events[0][0] == "stripe_anomaly" and events[0][1]["id"] == "ch_orphan"
    assert events[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert samples and json.loads(samples[0]["content"])["id"] == "ch_orphan"
    assert json.loads(samples[0]["content"])["module"] == sw.BILLING_ROUTER_MODULE
    assert billing_calls and billing_calls[0][2] == sw.SEVERITY_MAP["missing_charge"]
    assert billing_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert payment_calls and payment_calls[0][3] == sw.SEVERITY_MAP["missing_charge"]
    assert payment_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "missing_charge"
    assert logged["metadata"]["id"] == "ch_orphan"
    assert logged["metadata"]["module"] == sw.BILLING_ROUTER_MODULE
    assert "timestamp" in logged


def test_unknown_webhook_endpoint(monkeypatch, capture_anomalies):
    events, samples, billing_calls, payment_calls = capture_anomalies

    endpoints = [
        {
            "id": "we_good",
            "url": "https://good.example.com/hook",
            "status": "enabled",
        },
        {
            "id": "we_evil",
            "url": "https://evil.example.com/hook",
            "status": "enabled",
        },
    ]

    fake_stripe = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda api_key: endpoints)
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)

    unknown = sw.check_webhook_endpoints(
        api_key="sk_test",
        approved=["we_good"],
        write_codex=True,
    )

    assert unknown == ["we_evil"]
    assert {e[1]["type"] for e in events} == {"unknown_webhook"}
    assert events[0][1]["module"] == sw.WATCHDOG_MODULE
    assert {
        json.loads(s["content"])["type"] for s in samples
    } == {"unknown_webhook"}
    assert json.loads(samples[0]["content"])["module"] == sw.WATCHDOG_MODULE
    assert billing_calls and billing_calls[0][2] == sw.SEVERITY_MAP["unknown_webhook"]
    assert billing_calls[0][1]["module"] == sw.WATCHDOG_MODULE
    assert payment_calls and payment_calls[0][3] == sw.SEVERITY_MAP["unknown_webhook"]
    assert payment_calls[0][1]["module"] == sw.WATCHDOG_MODULE
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "unknown_webhook"
    assert logged["metadata"]["webhook_id"] == "we_evil"
    assert logged["metadata"]["module"] == sw.WATCHDOG_MODULE


def test_disabled_webhook_endpoint(monkeypatch, capture_anomalies):
    events, samples, billing_calls, payment_calls = capture_anomalies

    endpoints = [
        {
            "id": "we_disabled",
            "url": "https://good.example.com/hook",
            "status": "disabled",
        }
    ]

    fake_stripe = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda api_key: endpoints)
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)

    unknown = sw.check_webhook_endpoints(
        api_key="sk_test",
        approved=["we_disabled", "https://good.example.com/hook"],
        write_codex=True,
    )

    assert unknown == ["we_disabled"]
    assert {e[1]["type"] for e in events} == {"disabled_webhook"}
    assert events[0][1]["module"] == sw.WATCHDOG_MODULE
    assert {
        json.loads(s["content"])["type"] for s in samples
    } == {"disabled_webhook"}
    assert json.loads(samples[0]["content"])["module"] == sw.WATCHDOG_MODULE
    assert billing_calls and billing_calls[0][2] == sw.SEVERITY_MAP["disabled_webhook"]
    assert billing_calls[0][1]["module"] == sw.WATCHDOG_MODULE
    assert payment_calls and payment_calls[0][3] == sw.SEVERITY_MAP["disabled_webhook"]
    assert payment_calls[0][1]["module"] == sw.WATCHDOG_MODULE
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "disabled_webhook"
    assert logged["metadata"]["webhook_id"] == "we_disabled"
    assert logged["metadata"]["module"] == sw.WATCHDOG_MODULE


def test_env_allowed_webhook(monkeypatch):
    endpoints = [
        {
            "id": "we_env",
            "url": "https://env.example.com/hook",
            "status": "enabled",
        }
    ]

    fake_stripe = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda api_key: endpoints)
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)
    monkeypatch.setenv("STRIPE_ALLOWED_WEBHOOKS", "we_env")

    unknown = sw.check_webhook_endpoints(api_key="sk_test", context_builder=BUILDER)

    assert unknown == []


def test_unexpected_refund(capture_anomalies):
    events, samples, _, _ = capture_anomalies

    ledger: list[dict] = []
    refunds = [{"id": "rf_1", "amount": 500, "charge": "ch_1"}]

    anomalies = sw.detect_missing_refunds(
        refunds, ledger, write_codex=True, context_builder=BUILDER
    )

    assert anomalies and anomalies[0]["refund_id"] == "rf_1"
    assert anomalies[0]["module"] == sw.BILLING_ROUTER_MODULE
    assert events and events[0][1]["type"] == "missing_refund"
    assert events[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert samples and json.loads(samples[0]["content"])["refund_id"] == "rf_1"
    assert json.loads(samples[0]["content"])["module"] == sw.BILLING_ROUTER_MODULE
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "missing_refund"
    assert logged["metadata"]["refund_id"] == "rf_1"
    assert logged["metadata"]["module"] == sw.BILLING_ROUTER_MODULE


def test_unauthorized_refund_triggers_audit_and_codex(capture_anomalies):
    events, samples, billing_calls, payment_calls = capture_anomalies

    ledger: list[dict] = []
    refunds = [
        {"id": "rf_orphan", "amount": 700, "charge": "ch_1", "account": "acct"}
    ]
    logs = [{"stripe_id": "rf_orphan", "bot_id": "bot_a"}]

    anomalies = sw.detect_unauthorized_refunds(
        refunds, ledger, logs, ["bot_a"], write_codex=True, context_builder=BUILDER
    )

    assert anomalies and anomalies[0]["refund_id"] == "rf_orphan"
    assert anomalies[0]["module"] == sw.BILLING_ROUTER_MODULE
    assert events and events[0][1]["type"] == "unauthorized_refund"
    assert events[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert samples and json.loads(samples[0]["content"])["refund_id"] == "rf_orphan"
    assert json.loads(samples[0]["content"])["module"] == sw.BILLING_ROUTER_MODULE
    assert billing_calls and billing_calls[0][0] == "unauthorized_refund"
    assert billing_calls[0][2] == sw.SEVERITY_MAP["unauthorized_refund"]
    assert billing_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert payment_calls and payment_calls[0][0] == "unauthorized_refund"
    assert payment_calls[0][3] == sw.SEVERITY_MAP["unauthorized_refund"]
    assert payment_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "unauthorized_refund"
    assert logged["metadata"]["refund_id"] == "rf_orphan"
    assert logged["metadata"]["module"] == sw.BILLING_ROUTER_MODULE


def test_failed_event_missing_logs(capture_anomalies):
    events, samples, billing_calls, payment_calls = capture_anomalies

    ledger: list[dict] = []
    stripe_events = [
        {"id": "evt_1", "type": "charge.failed", "account": "acct"}
    ]

    anomalies = sw.detect_failed_events(
        stripe_events, ledger, write_codex=True, context_builder=BUILDER
    )

    assert anomalies and anomalies[0]["event_id"] == "evt_1"
    assert anomalies[0]["module"] == sw.BILLING_ROUTER_MODULE
    assert events and events[0][1]["event_id"] == "evt_1"
    assert events[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert samples and json.loads(samples[0]["content"])["event_id"] == "evt_1"
    assert json.loads(samples[0]["content"])["module"] == sw.BILLING_ROUTER_MODULE
    assert billing_calls and billing_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert payment_calls and payment_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "missing_failure_log"
    assert logged["metadata"]["event_id"] == "evt_1"
    assert logged["metadata"]["module"] == sw.BILLING_ROUTER_MODULE


def test_unauthorized_failure_triggers_audit_and_codex(capture_anomalies):
    events, samples, billing_calls, payment_calls = capture_anomalies

    ledger: list[dict] = []
    stripe_events = [
        {"id": "evt_orphan", "type": "charge.failed", "account": "acct"}
    ]
    logs = [{"stripe_id": "evt_orphan", "bot_id": "bot_a"}]

    anomalies = sw.detect_unauthorized_failures(
        stripe_events, ledger, logs, ["bot_a"], write_codex=True, context_builder=BUILDER
    )

    assert anomalies and anomalies[0]["event_id"] == "evt_orphan"
    assert anomalies[0]["module"] == sw.BILLING_ROUTER_MODULE
    assert events and events[0][1]["type"] == "unauthorized_failure"
    assert events[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert samples and json.loads(samples[0]["content"])["event_id"] == "evt_orphan"
    assert json.loads(samples[0]["content"])["module"] == sw.BILLING_ROUTER_MODULE
    assert billing_calls and billing_calls[0][0] == "unauthorized_failure"
    assert billing_calls[0][2] == sw.SEVERITY_MAP["unauthorized_failure"]
    assert billing_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    assert payment_calls and payment_calls[0][0] == "unauthorized_failure"
    assert payment_calls[0][3] == sw.SEVERITY_MAP["unauthorized_failure"]
    assert payment_calls[0][1]["module"] == sw.BILLING_ROUTER_MODULE
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "unauthorized_failure"
    assert logged["metadata"]["event_id"] == "evt_orphan"
    assert logged["metadata"]["module"] == sw.BILLING_ROUTER_MODULE


def test_revenue_mismatch(monkeypatch, capture_anomalies):
    events, samples, _, _ = capture_anomalies

    charges = [{"amount": 10000, "status": "succeeded"}]
    refunds: list[dict] = []

    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda api_key, s, e: charges)
    monkeypatch.setattr(sw, "fetch_recent_refunds", lambda api_key, s, e: refunds)
    monkeypatch.setattr(sw, "_projected_revenue_between", lambda s, e: 200.0)

    summary = sw.summarize_revenue_window(
        0, 10, tolerance=0.1, write_codex=True, context_builder=BUILDER
    )

    assert summary["projected_revenue"] == 200.0
    assert summary["charge_total"] == 100.0
    assert events and events[0][1]["type"] == "revenue_mismatch"
    assert samples and json.loads(samples[0]["content"])["type"] == "revenue_mismatch"
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "revenue_mismatch"
    assert logged["metadata"]["net_revenue"] == summary["net_revenue"]


def test_logged_charge_not_flagged(monkeypatch, capture_anomalies):
    events, _samples, _, _ = capture_anomalies

    ledger: list[dict] = [{"id": "ch_logged"}]
    billing_logs = [{"amount": 10.0, "timestamp": 2}]
    charges = [
        {"id": "ch_logged", "created": 2, "amount": 1000, "status": "succeeded"}
    ]

    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda api_key, start, end: charges)
    monkeypatch.setattr(sw, "load_local_ledger", lambda start, end: ledger)
    monkeypatch.setattr(sw, "load_billing_logs", lambda start, end: billing_logs)
    monkeypatch.setattr(sw, "check_webhook_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(sw, "DiscrepancyDB", None)
    monkeypatch.setattr(sw, "DiscrepancyRecord", None)

    anomalies = sw.check_events()

    assert anomalies == []


def test_anomaly_log_rotation(monkeypatch, tmp_path):
    log_path = tmp_path / "anomaly.log"
    handler = RotatingFileHandler(
        str(log_path), maxBytes=200, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    def rotator(src, dst):
        with open(src, "rb") as sf, open(dst, "wb") as df:
            df.write(sf.read())
        os.remove(src)

    handler.rotator = rotator
    handler.namer = lambda name: name
    monkeypatch.setattr(sw, "ANOMALY_LOG", log_path)
    monkeypatch.setattr(sw, "ANOMALY_HANDLER", handler)
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", AuditTrail(str(log_path), handler=handler))

    for _ in range(3):
        sw.ANOMALY_TRAIL.record({"test": "x" * 150})

    assert log_path.exists()
    assert log_path.with_name(log_path.name + ".1").exists()
    assert log_path.with_name(log_path.name + ".2").exists()


def test_emit_anomaly_triggers_sanity_layer(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    def fake_payment(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(sw.menace_sanity_layer, "record_payment_anomaly", fake_payment)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(
        sw,
        "ANOMALY_TRAIL",
        SimpleNamespace(record=lambda entry: None),
    )

    engine = object()
    telemetry = object()
    sw._emit_anomaly(
        {"type": "missing_charge", "id": "ch_test"},
        False,
        False,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
        context_builder=BUILDER,
    )

    assert calls and calls[0][0][0] == "missing_charge"
    assert calls[0][0][1]["charge_id"] == "ch_test"
    expected = sw.menace_sanity_layer.EVENT_TYPE_INSTRUCTIONS["missing_charge"]
    assert calls[0][0][2] == expected
    assert calls[0][1]["self_coding_engine"] is engine
    assert calls[0][1]["telemetry_feedback"] is telemetry


@pytest.mark.parametrize("event_type", list(sw.SEVERITY_MAP.keys()))
def test_emit_anomaly_instruction_varies_by_event_type(monkeypatch, event_type):
    calls: list[tuple[str, dict, str | None]] = []

    monkeypatch.setattr(
        sw.menace_sanity_layer,
        "record_payment_anomaly",
        lambda *a, **k: calls.append(a),
    )
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", SimpleNamespace(record=lambda entry: None))

    sw._emit_anomaly({"type": event_type}, False, False, context_builder=BUILDER)

    expected = sw.menace_sanity_layer.EVENT_TYPE_INSTRUCTIONS[event_type]
    assert calls and calls[0][2] == expected


def test_emit_anomaly_instruction_falls_back_to_generic(monkeypatch):
    calls: list[tuple[str, dict, str | None]] = []

    monkeypatch.setattr(
        sw.menace_sanity_layer,
        "record_payment_anomaly",
        lambda *a, **k: calls.append(a),
    )
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", SimpleNamespace(record=lambda entry: None))

    sw._emit_anomaly(
        {"type": "unhandled_event", "id": "x"}, False, False, context_builder=BUILDER
    )

    assert calls and calls[0][2] == sw.BILLING_EVENT_INSTRUCTION
