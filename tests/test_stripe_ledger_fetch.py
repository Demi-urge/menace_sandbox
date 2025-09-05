import types

import pytest

from billing import stripe_ledger


def _setup_ledger(monkeypatch, tmp_path):
    ledger = stripe_ledger.StripeLedger(db_path=tmp_path / "ledger.db")
    monkeypatch.setattr(stripe_ledger, "STRIPE_LEDGER", ledger)
    return ledger


def test_fetch_events(monkeypatch, tmp_path):
    ledger = _setup_ledger(monkeypatch, tmp_path)
    stripe_ledger.log_event(
        "charge", "bot1", 10.0, "usd", "u1@example.com", "acct1", 100, "ch_1"
    )
    stripe_ledger.log_event(
        "refund", "bot2", 5.0, "usd", None, "acct2", 200, "ch_2"
    )

    events = stripe_ledger.get_events(50, 150)
    assert [e["action"] for e in events] == ["charge"]

    all_events = ledger.fetch_events(0, 300)
    assert len(all_events) == 2
    assert {e["action"] for e in all_events} == {"charge", "refund"}
    assert set(all_events[0]) == {
        "id",
        "action",
        "bot_id",
        "amount",
        "currency",
        "user_email",
        "account_id",
        "charge_id",
        "timestamp",
    }


def test_fetch_events_empty(monkeypatch, tmp_path):
    _setup_ledger(monkeypatch, tmp_path)
    stripe_ledger.log_event(
        "charge", "bot1", 10.0, "usd", "u1@example.com", "acct1", 100, "ch_1"
    )

    events = stripe_ledger.get_events(200, 300)
    assert events == []
