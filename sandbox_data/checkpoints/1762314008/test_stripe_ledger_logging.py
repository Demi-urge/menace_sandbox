# flake8: noqa

import importlib
import importlib.util
import importlib.machinery
import sqlite3
import sys
import types

import pytest
import yaml
from dynamic_path_router import resolve_path


def _import_module(monkeypatch, tmp_path, secrets=None):
    pkg = types.ModuleType("sbrpkg")
    pkg.__path__ = [str(resolve_path("."))]
    pkg.__spec__ = importlib.machinery.ModuleSpec("sbrpkg", loader=None, is_package=True)
    sys.modules["sbrpkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"sbrpkg.{name}", resolve_path(f"{name}.py")  # path-ignore
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"sbrpkg.{name}"] = module
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    secrets = secrets or {
        "stripe_secret_key": "sk_live_dummy",
        "stripe_public_key": "pk_live_dummy",
        "stripe_allowed_secret_keys": "sk_live_dummy",
    }
    routes = {
        "stripe": {
            "default": {
                "finance": {
                    "finance_router_bot": {
                        "product_id": "prod_finance_router",
                        "price_id": "price_finance_standard",
                        "customer_id": "cus_finance_default",
                    }
                }
            }
        }
    }
    cfg = tmp_path / "routes.yaml"
    cfg.write_text(yaml.safe_dump(routes))
    monkeypatch.setenv("STRIPE_ROUTING_CONFIG", str(cfg))
    vsp = _load("vault_secret_provider")
    rb = types.SimpleNamespace(
        RollbackManager=type(
            "RollbackManager",
            (),
            {"auto_rollback": lambda self, tag, nodes: None},
        )
    )
    sys.modules["rollback_manager"] = rb
    sys.modules["sbrpkg.rollback_manager"] = rb
    arm = types.SimpleNamespace(
        AutomatedRollbackManager=lambda: type(
            "ARM",
            (),
            {"auto_rollback": lambda self, tag, bots: None},
        )()
    )
    sys.modules["advanced_error_management"] = arm
    sys.modules["sbrpkg.advanced_error_management"] = arm
    monkeypatch.setattr(
        vsp.VaultSecretProvider, "get", lambda self, n: secrets.get(n, "")
    )
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("STRIPE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("STRIPE_ALLOWED_SECRET_KEYS", raising=False)
    fake_stripe = types.SimpleNamespace(
        StripeClient=lambda api_key: types.SimpleNamespace(
            Account=types.SimpleNamespace(
                retrieve=lambda: {"id": "acct_1H123456789ABCDEF"}
            )
        ),
        Account=types.SimpleNamespace(
            retrieve=lambda api_key: {"id": "acct_1H123456789ABCDEF"}
        ),
    )
    sys.modules["stripe"] = fake_stripe
    sys.modules["sbrpkg.stripe"] = fake_stripe
    # Stub db_router to avoid filesystem interactions during import
    class _StubRouter:
        def get_connection(self, name: str, mode: str | None = None):  # pragma: no cover - simple stub
            return types.SimpleNamespace(
                execute=lambda *a, **k: None,
                commit=lambda: None,
            )

    dr = types.SimpleNamespace(
        GLOBAL_ROUTER=None,
        LOCAL_TABLES={},
        DBRouter=lambda *a, **k: _StubRouter(),
        init_db_router=lambda *a, **k: _StubRouter(),
    )
    sys.modules["db_router"] = dr
    sys.modules["sbrpkg.db_router"] = dr

    disc = types.SimpleNamespace(
        DiscrepancyDB=type("DiscrepancyDB", (), {"log": lambda self, msg, ctx=None: None})
    )
    sys.modules["discrepancy_db"] = disc
    sys.modules["sbrpkg.discrepancy_db"] = disc

    sbr = _load("stripe_billing_router")
    monkeypatch.setattr(
        sbr, "_get_account_id", lambda api_key: sbr.STRIPE_MASTER_ACCOUNT_ID
    )
    return sbr


@pytest.fixture
def sbr_with_db(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    bl = importlib.reload(sys.modules["billing.billing_logger"])
    monkeypatch.setattr(sbr, "billing_logger", bl)
    conn = sqlite3.connect(":memory:")  # noqa: SQL001

    class Router:
        def get_connection(self, name: str):
            return conn

    router = Router()
    monkeypatch.setattr(sbr.billing_logger, "GLOBAL_ROUTER", router)
    monkeypatch.setattr(sbr.billing_logger, "init_db_router", lambda name: router)
    return sbr, conn


def test_charge_writes_ledger(monkeypatch, sbr_with_db):
    sbr, conn = sbr_with_db

    def fake_item_create(*, api_key, **params):
        return {"id": "ii_test", **params}

    def fake_invoice_create(*, api_key, **params):
        return {"id": "in_test", **params}

    def fake_invoice_pay(invoice_id, *, api_key, **params):
        return {
            "id": invoice_id,
            "amount_paid": 1250,
            "on_behalf_of": sbr.STRIPE_MASTER_ACCOUNT_ID,
        }

    def fake_customer_retrieve(customer_id, *, api_key=None, **_):
        return {"email": "cust@example.com"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        InvoiceItem=types.SimpleNamespace(create=fake_item_create),
        Invoice=types.SimpleNamespace(create=fake_invoice_create, pay=fake_invoice_pay),
        Customer=types.SimpleNamespace(retrieve=fake_customer_retrieve),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    events: list[tuple] = []
    monkeypatch.setattr(
        sbr,
        "log_billing_event",
        lambda *a, **k: events.append((a, k)),
    )
    sbr.charge("finance:finance_router_bot", 12.5, "desc")
    row = conn.execute(
        "SELECT id, action_type, amount, bot_id, error FROM stripe_ledger",
    ).fetchone()
    assert row == ("in_test", "charge", 12.5, "finance:finance_router_bot", 0)
    assert events and events[0][0] == ("charge",)
    assert events[0][1]["amount"] == 12.5
    assert events[0][1]["user_email"] == "cust@example.com"
    assert events[0][1]["destination_account"] == sbr.STRIPE_MASTER_ACCOUNT_ID


def test_create_subscription_writes_ledger(monkeypatch, sbr_with_db):
    sbr, conn = sbr_with_db

    def fake_create(*, api_key, **params):
        return {"id": "sub_test", "on_behalf_of": sbr.STRIPE_MASTER_ACCOUNT_ID}

    def fake_customer_retrieve(customer_id, *, api_key=None, **_):
        return {"email": "cust@example.com"}

    def fake_price_retrieve(price_id, *, api_key=None, **_):
        return {"unit_amount": 1250}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        Subscription=types.SimpleNamespace(create=fake_create),
        Customer=types.SimpleNamespace(retrieve=fake_customer_retrieve),
        Price=types.SimpleNamespace(retrieve=fake_price_retrieve),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    events: list[tuple] = []
    monkeypatch.setattr(
        sbr,
        "log_billing_event",
        lambda *a, **k: events.append((a, k)),
    )
    sbr.create_subscription("finance:finance_router_bot", idempotency_key="sub-key")
    row = conn.execute(
        "SELECT id, action_type, amount, bot_id, error FROM stripe_ledger",
    ).fetchone()
    assert row == ("sub_test", "subscription", None, "finance:finance_router_bot", 0)
    assert events and events[0][0] == ("subscription",)
    assert events[0][1]["amount"] == 12.5
    assert events[0][1]["user_email"] == "cust@example.com"
    assert events[0][1]["destination_account"] == sbr.STRIPE_MASTER_ACCOUNT_ID


def test_refund_writes_ledger(monkeypatch, sbr_with_db):
    sbr, conn = sbr_with_db

    def fake_create(*, api_key, **params):
        return {
            "id": "rf_test",
            "amount": 500,
            "on_behalf_of": sbr.STRIPE_MASTER_ACCOUNT_ID,
        }

    fake_stripe = types.SimpleNamespace(
        api_key="orig", Refund=types.SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    sbr.refund("finance:finance_router_bot", "ch_test", amount=5.0)
    row = conn.execute(
        "SELECT id, action_type, amount, bot_id, error FROM stripe_ledger"
    ).fetchone()
    assert row == ("rf_test", "refund", 5.0, "finance:finance_router_bot", 0)


def test_create_checkout_session_writes_ledger(monkeypatch, sbr_with_db):
    sbr, conn = sbr_with_db

    def fake_create(*, api_key, **params):
        return {
            "id": "cs_test",
            "amount_total": 1000,
            "on_behalf_of": sbr.STRIPE_MASTER_ACCOUNT_ID,
        }

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        checkout=types.SimpleNamespace(Session=types.SimpleNamespace(create=fake_create)),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    line_items = [{"price": "price_finance_standard", "quantity": 1}]
    sbr.create_checkout_session(
        "finance:finance_router_bot",
        line_items,
        amount=10.0,
        success_url="https://example.com/s",
        cancel_url="https://example.com/c",
        mode="payment",
    )
    row = conn.execute(
        "SELECT id, action_type, amount, bot_id, error FROM stripe_ledger"
    ).fetchone()
    assert row == ("cs_test", "checkout_session", 10.0, "finance:finance_router_bot", 0)


def test_mismatched_account_dispatches_alert_and_rollbacks(monkeypatch, sbr_with_db):
    sbr, conn = sbr_with_db

    alerts = []
    rollbacks = []
    monkeypatch.setattr(
        sbr.alert_dispatcher, "dispatch_alert", lambda *a, **k: alerts.append((a, k))
    )

    class RB:
        def auto_rollback(self, tag, nodes):
            rollbacks.append((tag, nodes))

        def rollback(self, *a, **k):  # pragma: no cover - simple stub
            rollbacks.append((a[0] if a else "", k.get("requesting_bot")))

    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", RB)

    def fake_item_create(*, api_key, **params):
        return {"id": "ii_test", **params}

    def fake_invoice_create(*, api_key, **params):
        return {"id": "in_test", **params}

    def fake_invoice_pay(invoice_id, *, api_key, **params):
        return {"id": invoice_id, "on_behalf_of": "acct_bad"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        InvoiceItem=types.SimpleNamespace(create=fake_item_create),
        Invoice=types.SimpleNamespace(create=fake_invoice_create, pay=fake_invoice_pay),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    monkeypatch.setattr(sbr, "_get_account_id", lambda api_key: "acct_bad")
    with pytest.raises(RuntimeError):
        sbr.charge("finance:finance_router_bot", 12.5, "desc")
    assert alerts and rollbacks
    rows = conn.execute(
        "SELECT destination_account, error FROM stripe_ledger"
    ).fetchall()
    assert any(r == ("acct_bad", 1) for r in rows)


def test_charge_logs_on_api_exception(monkeypatch, sbr_with_db):
    sbr, conn = sbr_with_db

    def fake_item_create(*, api_key, **params):
        raise RuntimeError("boom")

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        InvoiceItem=types.SimpleNamespace(create=fake_item_create),
        Invoice=types.SimpleNamespace(create=lambda **kw: {}, pay=lambda *a, **kw: {}),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.charge("finance:finance_router_bot", 12.5, "desc")
    row = conn.execute(
        "SELECT action_type, error FROM stripe_ledger"
    ).fetchone()
    assert row == ("charge", 1)
