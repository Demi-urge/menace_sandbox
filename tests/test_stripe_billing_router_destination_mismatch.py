import types

import pytest

from .test_stripe_billing_router_logging import _import_module


@pytest.fixture
def sbr(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    monkeypatch.setattr(sbr, "_get_account_id", lambda api_key: sbr.STRIPE_REGISTERED_ACCOUNT_ID)
    monkeypatch.setattr(sbr, "record_payment", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "_log_payment", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "log_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "_client", lambda api_key: None)
    return sbr


def _get_action_log(calls, action):
    for call in calls:
        if call.get("action_type") == action:
            return call
    return {}


def test_charge_blocks_on_mismatch(monkeypatch, sbr):
    sbr.ROUTING_TABLE[("stripe", "default", "finance", "finance_router_bot")].pop(
        "price_id", None
    )
    fake_stripe = types.SimpleNamespace(
        PaymentIntent=types.SimpleNamespace(
            create=lambda **kw: {"id": "pi", "on_behalf_of": "acct_bad"}
        )
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    alerts = []
    monkeypatch.setattr(
        sbr,
        "_alert_mismatch",
        lambda bot_id, account_id, message="Stripe account mismatch", amount=None: alerts.append(
            (bot_id, account_id, amount)
        ),
    )
    logs = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **kw: logs.append(kw))
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.charge("finance:finance_router_bot", amount=5.0, description="d")
    assert alerts == [("finance:finance_router_bot", "acct_bad", 5.0)]
    log = _get_action_log(logs, "charge")
    assert log["error"] is True and log["destination_account"] == "acct_bad"


def test_subscription_blocks_on_mismatch(monkeypatch, sbr):
    fake_stripe = types.SimpleNamespace(
        Subscription=types.SimpleNamespace(
            create=lambda **kw: {"id": "sub", "account": "acct_bad"}
        )
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    alerts = []
    monkeypatch.setattr(
        sbr,
        "_alert_mismatch",
        lambda bot_id, account_id, message="Stripe account mismatch", amount=None: alerts.append(
            (bot_id, account_id)
        ),
    )
    logs = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **kw: logs.append(kw))
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.create_subscription("finance:finance_router_bot")
    assert alerts == [("finance:finance_router_bot", "acct_bad")]
    log = _get_action_log(logs, "subscription")
    assert log["error"] is True and log["destination_account"] == "acct_bad"


def test_refund_blocks_on_mismatch(monkeypatch, sbr):
    fake_stripe = types.SimpleNamespace(
        Refund=types.SimpleNamespace(
            create=lambda **kw: {"id": "rf", "amount": 500, "on_behalf_of": "acct_bad"}
        )
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    alerts = []
    monkeypatch.setattr(
        sbr,
        "_alert_mismatch",
        lambda bot_id, account_id, message="Stripe account mismatch", amount=None: alerts.append(
            (bot_id, account_id, amount)
        ),
    )
    logs = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **kw: logs.append(kw))
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.refund("finance:finance_router_bot", "pi", amount=5.0)
    assert alerts == [("finance:finance_router_bot", "acct_bad", 5.0)]
    log = _get_action_log(logs, "refund")
    assert log["error"] is True and log["destination_account"] == "acct_bad"


def test_checkout_session_blocks_on_mismatch(monkeypatch, sbr):
    fake_stripe = types.SimpleNamespace(
        checkout=types.SimpleNamespace(
            Session=types.SimpleNamespace(
                create=lambda **kw: {
                    "id": "cs",
                    "account": "acct_bad",
                    "amount_total": 500,
                }
            )
        )
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    alerts = []
    monkeypatch.setattr(
        sbr,
        "_alert_mismatch",
        lambda bot_id, account_id, message="Stripe account mismatch", amount=None: alerts.append(
            (bot_id, account_id, amount)
        ),
    )
    logs = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **kw: logs.append(kw))
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr.create_checkout_session(
            "finance:finance_router_bot",
            line_items=[{"price": "price_finance_standard", "quantity": 1}],
        )
    assert alerts == [("finance:finance_router_bot", "acct_bad", 5.0)]
    log = _get_action_log(logs, "checkout_session")
    assert log["error"] is True and log["destination_account"] == "acct_bad"
