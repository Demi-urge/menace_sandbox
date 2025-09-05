"""Tests for :mod:`stripe_watchdog` anomaly detection.

The tests intentionally avoid network access by stubbing out Stripe client
interactions and other optional dependencies. Each scenario verifies that an
anomaly results in an ``audit_logger`` entry and that, when ``write_codex`` is
True, a Codex ``TrainingSample`` is also created.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dynamic_path_router import resolve_path
from audit_trail import AuditTrail

import stripe_watchdog as sw


@pytest.fixture
def capture_anomalies(monkeypatch, tmp_path):
    """Capture emitted audit events and Codex samples."""

    events: list[tuple[str, dict]] = []
    samples: list[dict] = []

    monkeypatch.setattr(
        sw.audit_logger,
        "log_event",
        lambda et, data: events.append((et, data)),
    )

    class DummySample:
        def __init__(self, source: str, content: str):
            samples.append({"source": source, "content": content})

    monkeypatch.setattr(sw, "TrainingSample", DummySample)
    monkeypatch.setattr(sw, "ANOMALY_LOG", tmp_path / "anomaly.log")
    monkeypatch.setattr(sw, "ANOMALY_TRAIL", AuditTrail(str(tmp_path / "anomaly.log")))
    resolve_path("config/stripe_watchdog.yaml")
    monkeypatch.setattr(sw.alert_dispatcher, "dispatch_alert", lambda *a, **k: None)

    return events, samples


def test_orphan_charge_triggers_audit_and_codex(monkeypatch, capture_anomalies):
    events, samples = capture_anomalies

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
    assert events and events[0][0] == "stripe_anomaly" and events[0][1]["id"] == "ch_orphan"
    assert samples and json.loads(samples[0]["content"])["id"] == "ch_orphan"
    with sw.ANOMALY_LOG.open("r", encoding="utf-8") as fh:
        line = fh.readline()
        logged = json.loads(line.split(" ", 1)[1])
    assert logged["type"] == "missing_charge"
    assert logged["metadata"]["id"] == "ch_orphan"
    assert "timestamp" in logged


def test_unknown_webhook_endpoint(monkeypatch, capture_anomalies):
    events, samples = capture_anomalies

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
    assert {
        json.loads(s["content"])["type"] for s in samples
    } == {"unknown_webhook"}


def test_disabled_webhook_endpoint(monkeypatch, capture_anomalies):
    events, samples = capture_anomalies

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
    assert {
        json.loads(s["content"])["type"] for s in samples
    } == {"disabled_webhook"}


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

    unknown = sw.check_webhook_endpoints(api_key="sk_test")

    assert unknown == []


def test_unexpected_refund(capture_anomalies):
    events, samples = capture_anomalies

    ledger: list[dict] = []
    refunds = [{"id": "rf_1", "amount": 500, "charge": "ch_1"}]

    anomalies = sw.detect_missing_refunds(refunds, ledger, write_codex=True)

    assert anomalies and anomalies[0]["refund_id"] == "rf_1"
    assert events and events[0][1]["type"] == "missing_refund"
    assert samples and json.loads(samples[0]["content"])["refund_id"] == "rf_1"


def test_revenue_mismatch(monkeypatch, capture_anomalies):
    events, samples = capture_anomalies

    charges = [{"amount": 10000, "status": "succeeded"}]
    refunds: list[dict] = []

    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sw, "fetch_recent_charges", lambda api_key, s, e: charges)
    monkeypatch.setattr(sw, "fetch_recent_refunds", lambda api_key, s, e: refunds)
    monkeypatch.setattr(sw, "_projected_revenue_between", lambda s, e: 200.0)

    summary = sw.summarize_revenue_window(0, 10, tolerance=0.1, write_codex=True)

    assert summary["projected_revenue"] == 200.0
    assert summary["charge_total"] == 100.0
    assert events and events[0][1]["type"] == "revenue_mismatch"
    assert samples and json.loads(samples[0]["content"])["type"] == "revenue_mismatch"


def test_logged_charge_not_flagged(monkeypatch, capture_anomalies):
    events, _samples = capture_anomalies

    ledger: list[dict] = []
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
