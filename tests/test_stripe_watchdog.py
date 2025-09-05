import json
from types import SimpleNamespace

import stripe_watchdog as sw


def _fake_list(items):
    class FakeList(list):
        def auto_paging_iter(self):
            return iter(self)
    return FakeList(items)


def test_check_events_detects_missing(monkeypatch, tmp_path):
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
    monkeypatch.setattr(sw, "_load_allowed_endpoints", lambda path=sw.CONFIG_PATH: set())

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
    monkeypatch.setattr(sw, "_load_allowed_endpoints", lambda path=sw.CONFIG_PATH: set())

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


def test_check_webhook_endpoints_alerts_unknown(monkeypatch, tmp_path):
    cfg = tmp_path / "stripe_watchdog.yaml"  # path-ignore
    cfg.write_text(
        "allowed_endpoints:\n  - https://good.example.com/webhook\n"
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
