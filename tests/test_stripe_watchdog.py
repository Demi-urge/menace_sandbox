import json
from types import SimpleNamespace

import stripe_watchdog as sw


def _fake_list(items):
    class FakeList(list):
        def auto_paging_iter(self):
            return iter(self)
    return FakeList(items)


def test_check_events_detects_missing(monkeypatch, tmp_path):
    ledger = tmp_path / sw.LEDGER_FILE.name
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
        Charge=SimpleNamespace(list=lambda limit, api_key: charges)
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test_dummy")

    anomalies = sw.check_events()
    assert anomalies == [
        {"id": "ch_new", "amount": 3000, "email": "c@example.com", "timestamp": 789}
    ]
