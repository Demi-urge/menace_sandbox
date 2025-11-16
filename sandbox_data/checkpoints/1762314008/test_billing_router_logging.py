import os
from unittest.mock import MagicMock
import pytest
# flake8: noqa

# Ensure Stripe keys exist before importing module
os.environ.setdefault("STRIPE_SECRET_KEY", "sk_live_default")
os.environ.setdefault("STRIPE_PUBLIC_KEY", "pk_live_default")




import importlib
import types
import sys
import pathlib

# ensure package root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# stub discrepancy_db to avoid heavy imports
_discrepancy = types.ModuleType("discrepancy_db")
class DiscrepancyDB:
    def add(self, rec):
        pass
class DiscrepancyRecord:
    def __init__(self, message, metadata=None, ts=None, id=0):
        self.message = message
        self.metadata = metadata or {}
        self.ts = ts
        self.id = id
_discrepancy.DiscrepancyDB = DiscrepancyDB
_discrepancy.DiscrepancyRecord = DiscrepancyRecord
sys.modules.setdefault("discrepancy_db", _discrepancy)

# stub vault_secret_provider
_vault = types.ModuleType("vault_secret_provider")
class VaultSecretProvider:
    def get(self, name):
        return ""
_vault.VaultSecretProvider = VaultSecretProvider
sys.modules.setdefault("vault_secret_provider", _vault)


# stub rollback_manager
_rb = types.ModuleType("rollback_manager")
class RollbackManager:
    def rollback(self, *a, **kw):
        pass
_rb.RollbackManager = RollbackManager
sys.modules.setdefault("rollback_manager", _rb)

import menace_sandbox.stripe_billing_router as sbr

SECRET = "sk_live_123"
PUBLIC = "pk_live_123"
ACCOUNT = sbr.STRIPE_MASTER_ACCOUNT_ID

def test_charge_logs_events(monkeypatch):
    bot_id = "stripe:cat:bot"
    route = {
        "secret_key": SECRET,
        "public_key": PUBLIC,
        "account_id": ACCOUNT,
        "currency": "usd",
        "user_email": "user@example.com",
    }
    event = {"id": "pi_1", "amount": 5000, "account": ACCOUNT}

    monkeypatch.setattr(sbr, "_resolve_route", lambda b, overrides=None: route)
    monkeypatch.setattr(sbr, "_verify_route", lambda b, r: None)
    monkeypatch.setattr(sbr, "_get_account_id", lambda k: ACCOUNT)

    class Client:
        PaymentIntent = type("PI", (), {"create": staticmethod(lambda **kw: event)})

    monkeypatch.setattr(sbr, "_client", lambda k: Client())
    monkeypatch.setattr(sbr, "record_payment", MagicMock())

    log_event = MagicMock()
    log_billing = MagicMock()
    ledger_log = MagicMock()
    monkeypatch.setattr(sbr.billing_logger, "log_event", log_event)
    monkeypatch.setattr(sbr, "log_billing_event", log_billing)
    monkeypatch.setattr(sbr._STRIPE_LEDGER, "log_event", ledger_log)
    monkeypatch.setattr(sbr.time, "time", lambda: 1_700_000_000.0)

    sbr.charge(bot_id, amount=50.0)

    kwargs = log_event.call_args.kwargs
    assert kwargs["amount"] == 50.0
    assert kwargs["timestamp_ms"] == 1_700_000_000_000
    assert kwargs["user_email"] == "user@example.com"
    assert kwargs["bot_id"] == bot_id
    assert kwargs["destination_account"] == ACCOUNT

    kw = log_billing.call_args.kwargs
    assert kw["amount"] == 50.0
    assert kw["bot_id"] == bot_id
    assert kw["user_email"] == "user@example.com"
    assert kw["destination_account"] == ACCOUNT

    ledger_args = ledger_log.call_args[0]
    assert ledger_args[0] == "charge"
    assert ledger_args[1] == bot_id
    assert ledger_args[2] == 50.0
    assert ledger_args[4] == "user@example.com"
    assert ledger_args[5] == ACCOUNT
    assert ledger_args[6] == 1_700_000_000_000
    assert ledger_args[7] == "pi_1"

def test_subscription_logs_events(monkeypatch):
    bot_id = "stripe:cat:bot"
    route = {
        "secret_key": SECRET,
        "public_key": PUBLIC,
        "account_id": ACCOUNT,
        "currency": "usd",
        "user_email": "user@example.com",
    }
    event = {"id": "sub_1", "account": ACCOUNT}

    class Client:
        Subscription = type("Sub", (), {"create": staticmethod(lambda **kw: event)})
        Price = type("Price", (), {"retrieve": staticmethod(lambda pid: {"unit_amount": 1234})})

    class StripeStub:
        class Customer:
            @staticmethod
            def retrieve(cid, api_key=None):
                return {"email": "user@example.com"}

    monkeypatch.setattr(sbr, "stripe", StripeStub())
    monkeypatch.setattr(sbr, "_resolve_route", lambda b, overrides=None: route)
    monkeypatch.setattr(sbr, "_verify_route", lambda b, r: None)
    monkeypatch.setattr(sbr, "_get_account_id", lambda k: ACCOUNT)
    monkeypatch.setattr(sbr, "_client", lambda k: Client())
    monkeypatch.setattr(sbr, "record_payment", MagicMock())

    log_event = MagicMock()
    log_billing = MagicMock()
    ledger_log = MagicMock()
    monkeypatch.setattr(sbr.billing_logger, "log_event", log_event)
    monkeypatch.setattr(sbr, "log_billing_event", log_billing)
    monkeypatch.setattr(sbr._STRIPE_LEDGER, "log_event", ledger_log)
    monkeypatch.setattr(sbr.time, "time", lambda: 1_700_000_000.0)

    sbr.create_subscription(bot_id, price_id="price_123", customer_id="cust_123")

    kwargs = log_event.call_args.kwargs
    assert kwargs["bot_id"] == bot_id
    assert kwargs["user_email"] == "user@example.com"
    assert kwargs["destination_account"] == ACCOUNT
    assert kwargs["timestamp_ms"] == 1_700_000_000_000
    assert "amount" in kwargs

    kw = log_billing.call_args.kwargs
    assert kw["amount"] == pytest.approx(12.34)
    assert kw["bot_id"] == bot_id
    assert kw["user_email"] == "user@example.com"
    assert kw["destination_account"] == ACCOUNT

    ledger_args = ledger_log.call_args[0]
    assert ledger_args[0] == "subscription"
    assert ledger_args[1] == bot_id
    assert ledger_args[2] == 0.0
    assert ledger_args[4] == "user@example.com"
    assert ledger_args[5] == ACCOUNT
    assert ledger_args[6] == 1_700_000_000_000
    assert ledger_args[7] == "sub_1"

def test_refund_logs_events(monkeypatch):
    bot_id = "stripe:cat:bot"
    route = {
        "secret_key": SECRET,
        "public_key": PUBLIC,
        "account_id": ACCOUNT,
        "currency": "usd",
        "user_email": "user@example.com",
    }
    event = {"id": "re_1", "amount": 5000, "account": ACCOUNT}

    class Client:
        Refund = type("Refund", (), {"create": staticmethod(lambda **kw: event)})

    monkeypatch.setattr(sbr, "_resolve_route", lambda b, overrides=None: route)
    monkeypatch.setattr(sbr, "_verify_route", lambda b, r: None)
    monkeypatch.setattr(sbr, "_get_account_id", lambda k: ACCOUNT)
    monkeypatch.setattr(sbr, "_client", lambda k: Client())
    monkeypatch.setattr(sbr, "record_payment", MagicMock())

    log_event = MagicMock()
    log_billing = MagicMock()
    ledger_log = MagicMock()
    monkeypatch.setattr(sbr.billing_logger, "log_event", log_event)
    monkeypatch.setattr(sbr, "log_billing_event", log_billing)
    monkeypatch.setattr(sbr._STRIPE_LEDGER, "log_event", ledger_log)
    monkeypatch.setattr(sbr.time, "time", lambda: 1_700_000_000.0)

    sbr.refund(bot_id, "pi_123", amount=50.0)

    kwargs = log_event.call_args.kwargs
    assert kwargs["amount"] == 50.0
    assert kwargs["user_email"] == "user@example.com"
    assert kwargs["bot_id"] == bot_id
    assert kwargs["destination_account"] == ACCOUNT
    assert kwargs["timestamp_ms"] == 1_700_000_000_000

    kw = log_billing.call_args.kwargs
    assert kw["amount"] == 50.0
    assert kw["bot_id"] == bot_id
    assert kw["user_email"] == "user@example.com"
    assert kw["destination_account"] == ACCOUNT

    ledger_args = ledger_log.call_args[0]
    assert ledger_args[0] == "refund"
    assert ledger_args[1] == bot_id
    assert ledger_args[2] == 50.0
    assert ledger_args[4] == "user@example.com"
    assert ledger_args[5] == ACCOUNT
    assert ledger_args[6] == 1_700_000_000_000
    assert ledger_args[7] == "re_1"

def test_checkout_session_logs_events(monkeypatch):
    bot_id = "stripe:cat:bot"
    route = {
        "secret_key": SECRET,
        "public_key": PUBLIC,
        "account_id": ACCOUNT,
        "currency": "usd",
        "user_email": "user@example.com",
    }
    event = {"id": "cs_1", "amount_total": 7000, "account": ACCOUNT}

    class Client:
        checkout = type(
            "checkout",
            (),
            {"Session": type("Session", (), {"create": staticmethod(lambda **kw: event)})},
        )

    monkeypatch.setattr(sbr, "_resolve_route", lambda b, overrides=None: route)
    monkeypatch.setattr(sbr, "_verify_route", lambda b, r: None)
    monkeypatch.setattr(sbr, "_get_account_id", lambda k: ACCOUNT)
    monkeypatch.setattr(sbr, "_client", lambda k: Client())
    monkeypatch.setattr(sbr, "record_payment", MagicMock())

    log_event = MagicMock()
    log_billing = MagicMock()
    ledger_log = MagicMock()
    monkeypatch.setattr(sbr.billing_logger, "log_event", log_event)
    monkeypatch.setattr(sbr, "log_billing_event", log_billing)
    monkeypatch.setattr(sbr._STRIPE_LEDGER, "log_event", ledger_log)
    monkeypatch.setattr(sbr.time, "time", lambda: 1_700_000_000.0)

    sbr.create_checkout_session(bot_id, line_items=[{"price": "p", "quantity": 1}])

    kwargs = log_event.call_args.kwargs
    assert kwargs["amount"] == 70.0
    assert kwargs["user_email"] == "user@example.com"
    assert kwargs["bot_id"] == bot_id
    assert kwargs["destination_account"] == ACCOUNT
    assert kwargs["timestamp_ms"] == 1_700_000_000_000

    kw = log_billing.call_args.kwargs
    assert kw["amount"] == 70.0
    assert kw["bot_id"] == bot_id
    assert kw["user_email"] == "user@example.com"
    assert kw["destination_account"] == ACCOUNT

    ledger_args = ledger_log.call_args[0]
    assert ledger_args[0] == "checkout"
    assert ledger_args[1] == bot_id
    assert ledger_args[2] == 0.0
    assert ledger_args[4] == "user@example.com"
    assert ledger_args[5] == ACCOUNT
    assert ledger_args[6] == 1_700_000_000_000
    assert ledger_args[7] == "cs_1"

def test_alert_mismatch_invalid_key(monkeypatch):
    log_crit = MagicMock(wraps=sbr.log_critical_discrepancy)
    monkeypatch.setattr(sbr, "log_critical_discrepancy", log_crit)
    dispatch = MagicMock()
    monkeypatch.setattr(sbr.alert_dispatcher, "dispatch_alert", dispatch)
    rollback_called = MagicMock()

    class DummyRollback:
        def rollback(self, *a, **kw):
            rollback_called(*a, **kw)

    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: DummyRollback())
    pause = MagicMock()
    monkeypatch.setattr(sbr.sandbox_review, "pause_bot", pause)
    monkeypatch.setattr(sbr.billing_logger, "log_event", MagicMock())
    monkeypatch.setattr(sbr, "log_billing_event", MagicMock())

    import evolution_lock_flag

    lock = MagicMock()
    monkeypatch.setattr(evolution_lock_flag, "trigger_lock", lock)

    sbr._alert_mismatch(
        "stripe:cat:bot", "acct_wrong", message="Stripe key misconfiguration", amount=10.0
    )

    log_crit.assert_called_once_with("stripe:cat:bot", "Stripe key misconfiguration")
    dispatch.assert_called_once_with(
        "critical_discrepancy", 5, "Stripe key misconfiguration", {"bot": "stripe:cat:bot"}
    )
    lock.assert_called_once_with("Stripe account mismatch for stripe:cat:bot", severity=5)
    rollback_called.assert_called_once_with("latest", requesting_bot="stripe:cat:bot")
    pause.assert_called_once_with("stripe:cat:bot")


def test_alert_mismatch_account_mismatch(monkeypatch):
    log_crit = MagicMock(wraps=sbr.log_critical_discrepancy)
    monkeypatch.setattr(sbr, "log_critical_discrepancy", log_crit)
    dispatch = MagicMock()
    monkeypatch.setattr(sbr.alert_dispatcher, "dispatch_alert", dispatch)
    rollback_called = MagicMock()

    class DummyRollback:
        def rollback(self, *a, **kw):
            rollback_called(*a, **kw)

    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: DummyRollback())
    pause = MagicMock()
    monkeypatch.setattr(sbr.sandbox_review, "pause_bot", pause)
    monkeypatch.setattr(sbr.billing_logger, "log_event", MagicMock())
    monkeypatch.setattr(sbr, "log_billing_event", MagicMock())

    import evolution_lock_flag

    lock = MagicMock()
    monkeypatch.setattr(evolution_lock_flag, "trigger_lock", lock)

    sbr._alert_mismatch("stripe:cat:bot", "acct_mismatch", amount=5.0)

    log_crit.assert_called_once_with("stripe:cat:bot", "Stripe account mismatch")
    dispatch.assert_called_once_with(
        "critical_discrepancy", 5, "Stripe account mismatch", {"bot": "stripe:cat:bot"}
    )
    lock.assert_called_once_with("Stripe account mismatch for stripe:cat:bot", severity=5)
    rollback_called.assert_called_once_with("latest", requesting_bot="stripe:cat:bot")
    pause.assert_called_once_with("stripe:cat:bot")

