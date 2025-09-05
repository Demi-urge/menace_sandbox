import json
from types import SimpleNamespace

import stripe_watchdog as sw
from dynamic_path_router import resolve_path

resolve_path("docs")  # path-ignore


def _fake_list(items):
    class FakeList(list):
        def auto_paging_iter(self):
            return iter(self)
    return FakeList(items)


def test_check_events_detects_orphan_charge(monkeypatch, tmp_path):
    ledger = tmp_path / sw.LEDGER_FILE.name  # path-ignore
    ledger.write_text(
        json.dumps({"id": "ch_logged", "timestamp_ms": 123000}) + "\n"
        + json.dumps({"timestamp_ms": 456000}) + "\n"
    )
    monkeypatch.setattr(sw, "LEDGER_FILE", ledger)

    charges = _fake_list(
        [
            {
                "id": "ch_logged",
                "created": 123,
                "amount": 1000,
                "receipt_email": "a@example.com",
            },
            {
                "id": "ch_ts",
                "created": 456,
                "amount": 2000,
                "receipt_email": "b@example.com",
            },
            {
                "id": "ch_new",
                "created": 789,
                "amount": 3000,
                "receipt_email": "c@example.com",
            },
        ]
    )
    fake_stripe = SimpleNamespace(
        Charge=SimpleNamespace(list=lambda limit, api_key: charges),
        WebhookEndpoint=SimpleNamespace(list=lambda api_key: _fake_list([])),
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test_dummy")
    monkeypatch.setattr(sw, "_load_authorized_webhooks", lambda path=sw.CONFIG_PATH: set())

    anomalies = sw.check_events()
    assert anomalies == [
        {"id": "ch_new", "amount": 3000, "email": "c@example.com", "timestamp": 789}
    ]


def test_check_events_writes_log_and_summary(monkeypatch, tmp_path):
    ledger = tmp_path / sw.LEDGER_FILE.name  # path-ignore
    ledger.write_text(json.dumps({"id": "ch_logged"}) + "\n")
    monkeypatch.setattr(sw, "LEDGER_FILE", ledger)

    log_file = tmp_path / "stripe_watchdog.log"  # path-ignore
    monkeypatch.setattr(sw, "ANOMALY_LOG", log_file)

    charges = _fake_list([
        {"id": "ch_new", "created": 1, "amount": 100, "receipt_email": "a@x"}
    ])
    fake_stripe = SimpleNamespace(
        Charge=SimpleNamespace(list=lambda limit, api_key: charges),
        WebhookEndpoint=SimpleNamespace(list=lambda api_key: _fake_list([])),
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test_dummy")
    monkeypatch.setattr(sw, "_load_authorized_webhooks", lambda path=sw.CONFIG_PATH: set())

    records = []

    class DummyRecord:
        def __init__(self, message, metadata):
            self.message = message
            self.metadata = metadata

    class DummyDB:
        def add(self, rec):
            records.append(rec)

    monkeypatch.setattr(sw, "DiscrepancyDB", lambda: DummyDB())
    monkeypatch.setattr(sw, "DiscrepancyRecord", DummyRecord)

    anomalies = sw.check_events()
    assert json.loads(log_file.read_text().splitlines()[0]) == anomalies[0]
    assert records and str(len(anomalies)) in records[0].message


def test_check_revenue_projection_detects_mismatch(monkeypatch):
    charges = _fake_list([
        {"amount": 10000, "status": "succeeded"},
    ])
    refunds = _fake_list([
        {"amount": 2000},
    ])
    fake_stripe = SimpleNamespace(
        Charge=SimpleNamespace(list=lambda limit, api_key: charges),
        Refund=SimpleNamespace(list=lambda limit, api_key: refunds),
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test_dummy")

    class DummyDB:
        def projected_revenue(self):
            return 120.0

    monkeypatch.setattr(sw, "ROIResultsDB", lambda: DummyDB())

    calls = []

    def fake_alert(alert_type, severity, message, context=None):
        calls.append((alert_type, severity, json.loads(message)))

    monkeypatch.setattr(sw.alert_dispatcher, "dispatch_alert", fake_alert)

    anomaly = sw.check_revenue_projection(tolerance=0.1)
    assert anomaly == {
        "net_revenue": 80.0,
        "projected_revenue": 120.0,
        "difference": -40.0,
    }
    assert calls and calls[0][0] == "stripe_revenue_mismatch"


def test_check_webhook_endpoints_alerts_unauthorized(monkeypatch, tmp_path):
    cfg = tmp_path / "stripe_watchdog.yaml"  # path-ignore
    cfg.write_text(
        "authorized_webhooks:\n  - https://good.example.com/webhook\n"
    )
    monkeypatch.setattr(sw, "CONFIG_PATH", cfg)

    endpoints = _fake_list(
        [
            {"url": "https://good.example.com/webhook"},
            {"url": "https://bad.example.com/webhook"},
        ]
    )
    fake_stripe = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda api_key: endpoints)
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test_dummy")

    calls = []

    def fake_alert(alert_type, severity, message, context=None):
        calls.append((alert_type, severity, message))

    monkeypatch.setattr(sw.alert_dispatcher, "dispatch_alert", fake_alert)

    unknown = sw.check_webhook_endpoints("sk_test_dummy")
    assert unknown == ["https://bad.example.com/webhook"]
    assert calls and calls[0][0] == "stripe_unknown_endpoint"
