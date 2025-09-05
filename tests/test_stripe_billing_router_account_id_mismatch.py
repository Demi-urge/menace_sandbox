import pytest

from .test_stripe_billing_router_logging import _import_module

FOREIGN_ACCOUNT = "acct_foreign"


@pytest.fixture
def sbr(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    class DummyClient:
        def __getattr__(self, name):
            raise AssertionError("Stripe client should not be used")

    dummy = DummyClient()
    monkeypatch.setattr(sbr, "_client", lambda api_key: dummy)
    monkeypatch.setattr(sbr, "stripe", dummy)
    monkeypatch.setattr(sbr, "_get_account_id", lambda api_key: FOREIGN_ACCOUNT)
    monkeypatch.setattr(sbr, "record_payment", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "_log_payment", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "log_billing_event", lambda *a, **k: None)
    return sbr


def _setup_alert(monkeypatch, sbr):
    import evolution_lock_flag

    alerts = []
    logs = []
    locks = []

    monkeypatch.setattr(sbr, "log_critical_discrepancy", lambda *a, **k: None)
    monkeypatch.setattr(sbr.sandbox_review, "pause_bot", lambda *a, **k: None)
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **kw: logs.append(kw))
    monkeypatch.setattr(
        evolution_lock_flag,
        "trigger_lock",
        lambda reason, severity: locks.append((reason, severity)),
    )
    real_alert = sbr._alert_mismatch

    def wrapper(bot_id, account_id, message="Stripe account mismatch", amount=None):
        alerts.append((bot_id, account_id, amount))
        return real_alert(bot_id, account_id, message=message, amount=amount)

    monkeypatch.setattr(sbr, "_alert_mismatch", wrapper)
    return alerts, logs, locks


def test_charge_foreign_account(monkeypatch, sbr):
    alerts, logs, locks = _setup_alert(monkeypatch, sbr)
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.charge("finance:finance_router_bot", amount=5.0, description="d")
    assert alerts == [("finance:finance_router_bot", FOREIGN_ACCOUNT, 5.0)]
    assert logs[0]["error"] is True
    assert logs[0]["destination_account"] == FOREIGN_ACCOUNT
    assert locks == [("Stripe account mismatch for finance:finance_router_bot", 5)]


def test_subscription_foreign_account(monkeypatch, sbr):
    alerts, logs, locks = _setup_alert(monkeypatch, sbr)
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.create_subscription("finance:finance_router_bot")
    assert alerts == [("finance:finance_router_bot", FOREIGN_ACCOUNT, None)]
    assert logs[0]["error"] is True
    assert logs[0]["destination_account"] == FOREIGN_ACCOUNT
    assert locks == [("Stripe account mismatch for finance:finance_router_bot", 5)]


def test_refund_foreign_account(monkeypatch, sbr):
    alerts, logs, locks = _setup_alert(monkeypatch, sbr)
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.refund("finance:finance_router_bot", "pi", amount=5.0)
    assert alerts == [("finance:finance_router_bot", FOREIGN_ACCOUNT, 5.0)]
    assert logs[0]["error"] is True
    assert logs[0]["destination_account"] == FOREIGN_ACCOUNT
    assert locks == [("Stripe account mismatch for finance:finance_router_bot", 5)]


def test_checkout_session_foreign_account(monkeypatch, sbr):
    alerts, logs, locks = _setup_alert(monkeypatch, sbr)
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.create_checkout_session(
            "finance:finance_router_bot",
            line_items=[{"price": "price_finance_standard", "quantity": 1}],
        )
    assert alerts == [("finance:finance_router_bot", FOREIGN_ACCOUNT, None)]
    assert logs[0]["error"] is True
    assert logs[0]["destination_account"] == FOREIGN_ACCOUNT
    assert locks == [("Stripe account mismatch for finance:finance_router_bot", 5)]
