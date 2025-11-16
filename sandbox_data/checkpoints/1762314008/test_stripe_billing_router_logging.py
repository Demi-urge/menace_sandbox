import importlib
import importlib.util
import importlib.machinery
import json
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
    # Provide a lightweight discrepancy DB stub to avoid heavy imports
    dd = types.SimpleNamespace(
        DiscrepancyDB=lambda: types.SimpleNamespace(
            log=lambda *a, **k: None, add=lambda *a, **k: None
        )
    )
    sys.modules["discrepancy_db"] = dd
    sys.modules["sbrpkg.discrepancy_db"] = dd
    rb = types.SimpleNamespace(
        RollbackManager=type(
            "RollbackManager",
            (),
            {
                "auto_rollback": lambda self, tag, nodes: None,
                "rollback": lambda self, tag, requesting_bot=None: None,
            },
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
    sbr = _load("stripe_billing_router")
    return sbr


@pytest.fixture
def sbr_file_logger(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    bl = importlib.reload(sys.modules["billing.billing_logger"])
    ledger_mod = importlib.reload(sys.modules["billing.billing_ledger"])
    ledger_file = tmp_path / "stripe_ledger.jsonl"

    class BadRouter:
        def get_connection(self, name):
            raise RuntimeError("no db")

    monkeypatch.setattr(bl, "GLOBAL_ROUTER", BadRouter())
    monkeypatch.setattr(bl, "init_db_router", lambda name: BadRouter())
    monkeypatch.setattr(bl, "_LEDGER_FILE", ledger_file)
    monkeypatch.setattr(ledger_mod, "_LEDGER_FILE", ledger_file)
    monkeypatch.setattr(sbr, "billing_logger", bl)
    monkeypatch.setattr(sbr, "record_payment", ledger_mod.record_payment)
    monkeypatch.setattr(
        sbr, "_get_account_id", lambda api_key: sbr.STRIPE_MASTER_ACCOUNT_ID
    )
    return sbr, ledger_file


@pytest.fixture
def sbr_basic(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sbr, "_get_account_id", lambda api_key: sbr.STRIPE_MASTER_ACCOUNT_ID
    )
    monkeypatch.setattr(sbr, "record_payment", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "_log_payment", lambda *a, **k: None)
    monkeypatch.setattr(sbr, "log_billing_event", lambda *a, **k: None)
    return sbr


def _read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_charge_logs_to_file(monkeypatch, sbr_file_logger):
    sbr, ledger = sbr_file_logger

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

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        InvoiceItem=types.SimpleNamespace(create=fake_item_create),
        Invoice=types.SimpleNamespace(create=fake_invoice_create, pay=fake_invoice_pay),
        PaymentIntent=types.SimpleNamespace(create=lambda **kw: {"id": "pi_test"}),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    calls = []
    orig = sbr.billing_logger.log_event

    def capture(**k):
        calls.append(k)
        return orig(**k)

    monkeypatch.setattr(sbr.billing_logger, "log_event", capture)

    sbr.charge("finance:finance_router_bot", 12.5, "desc")
    records = _read_records(ledger)
    assert any(
        r.get("action") == "charge"
        and r.get("amount") == 12.5
        and r.get("charge_id") == "pi_test"
        for r in records
    )
    assert calls and calls[0].get("error") is False


def test_refund_logs_to_file(monkeypatch, sbr_file_logger):
    sbr, ledger = sbr_file_logger

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
    calls = []
    orig = sbr.billing_logger.log_event

    def capture(**k):
        calls.append(k)
        return orig(**k)

    monkeypatch.setattr(sbr.billing_logger, "log_event", capture)

    sbr.refund("finance:finance_router_bot", "ch_test", amount=5.0)
    records = _read_records(ledger)
    assert any(
        r.get("action") == "refund"
        and r.get("amount") == 5.0
        and r.get("charge_id") == "rf_test"
        for r in records
    )
    assert calls and calls[0].get("error") is False


def test_create_subscription_logs_to_file(monkeypatch, sbr_file_logger):
    sbr, ledger = sbr_file_logger

    def fake_create(*, api_key, **params):
        return {"id": "sub_test", "on_behalf_of": sbr.STRIPE_MASTER_ACCOUNT_ID}

    fake_stripe = types.SimpleNamespace(
        api_key="orig", Subscription=types.SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    calls = []
    orig = sbr.billing_logger.log_event

    def capture(**k):
        calls.append(k)
        return orig(**k)

    monkeypatch.setattr(sbr.billing_logger, "log_event", capture)

    sbr.create_subscription("finance:finance_router_bot", idempotency_key="sub-key")
    records = _read_records(ledger)
    assert any(
        r.get("action") == "subscription" and r.get("charge_id") == "sub_test"
        for r in records
    )
    assert calls and calls[0].get("error") is False


def test_create_checkout_session_logs_to_file(monkeypatch, sbr_file_logger):
    sbr, ledger = sbr_file_logger

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
    calls = []
    orig = sbr.billing_logger.log_event

    def capture(**k):
        calls.append(k)
        return orig(**k)

    monkeypatch.setattr(sbr.billing_logger, "log_event", capture)

    line_items = [{"price": "price_finance_standard", "quantity": 1}]
    sbr.create_checkout_session(
        "finance:finance_router_bot",
        line_items,
        amount=10.0,
        success_url="https://example.com/s",
        cancel_url="https://example.com/c",
        mode="payment",
    )
    records = _read_records(ledger)
    assert any(
        r.get("action") == "checkout_session"
        and r.get("amount") == 10.0
        and r.get("charge_id") == "cs_test"
        for r in records
    )
    assert calls and calls[0].get("error") is False


@pytest.mark.parametrize(
    "setup_stripe, invoke",
    [
        (
            lambda: types.SimpleNamespace(
                api_key="orig",
                InvoiceItem=types.SimpleNamespace(
                    create=lambda *, api_key, **p: {"id": "ii_test"}
                ),
                Invoice=types.SimpleNamespace(
                    create=lambda *, api_key, **p: {"id": "in_test"},
                    pay=lambda invoice_id, *, api_key, **p: {
                        "id": invoice_id,
                        "amount_paid": 1250,
                    },
                ),
                Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
            ),
            lambda sbr: sbr.charge("finance:finance_router_bot", 12.5, "desc"),
        ),
        (
            lambda: types.SimpleNamespace(
                api_key="orig",
                Subscription=types.SimpleNamespace(
                    create=lambda *, api_key, **p: {"id": "sub_test"}
                ),
                Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
                Price=types.SimpleNamespace(
                    retrieve=lambda *a, **k: {"unit_amount": 1250}
                ),
            ),
            lambda sbr: sbr.create_subscription(
                "finance:finance_router_bot", idempotency_key="sub-key"
            ),
        ),
        (
            lambda: types.SimpleNamespace(
                api_key="orig",
                Refund=types.SimpleNamespace(
                    create=lambda *, api_key, **p: {"id": "rf_test", "amount": 500}
                ),
                Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
            ),
            lambda sbr: sbr.refund(
                "finance:finance_router_bot", "pi_test", amount=5.0
            ),
        ),
        (
            lambda: types.SimpleNamespace(
                api_key="orig",
                checkout=types.SimpleNamespace(
                    Session=types.SimpleNamespace(
                        create=lambda *, api_key, **p: {
                            "id": "cs_test",
                            "amount_total": 1000,
                        }
                    )
                ),
                Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
            ),
            lambda sbr: sbr.create_checkout_session(
                "finance:finance_router_bot",
                [{"price": "price_finance_standard", "quantity": 1}],
                amount=10.0,
                success_url="https://example.com/s",
                cancel_url="https://example.com/c",
                mode="payment",
            ),
        ),
    ],
)
def test_logs_use_account_id_no_secret(monkeypatch, sbr_basic, setup_stripe, invoke):
    sbr = sbr_basic
    captured_log_event: list[dict] = []
    captured_record: list[tuple] = []
    captured_billing: list[tuple] = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **k: captured_log_event.append(k))
    monkeypatch.setattr(
        sbr, "record_payment", lambda *a, **k: captured_record.append((a, k))
    )
    monkeypatch.setattr(
        sbr, "log_billing_event", lambda action, **k: captured_billing.append((action, k))
    )
    monkeypatch.setattr(sbr, "stripe", setup_stripe())

    invoke(sbr)

    assert (
        captured_log_event
        and captured_log_event[-1]["destination_account"]
        == sbr.STRIPE_MASTER_ACCOUNT_ID
    )
    assert "sk_live_dummy" not in str(captured_log_event[-1])
    rec_args, rec_kwargs = captured_record[-1]
    assert rec_args[3] == sbr.STRIPE_MASTER_ACCOUNT_ID
    assert "sk_live_dummy" not in (str(rec_args) + str(rec_kwargs))
    action, billing_kwargs = captured_billing[-1]
    assert billing_kwargs["destination_account"] == sbr.STRIPE_MASTER_ACCOUNT_ID
    assert "sk_live_dummy" not in str(billing_kwargs)


def test_charge_logs_error_on_exception(monkeypatch, sbr_basic):
    sbr = sbr_basic
    calls = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **k: calls.append(k))

    def bad_item_create(*, api_key, **params):
        raise RuntimeError("fail")

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        InvoiceItem=types.SimpleNamespace(create=bad_item_create),
        Invoice=types.SimpleNamespace(
            create=lambda *a, **k: {"id": "in"},
            pay=lambda *a, **k: {"id": "in"},
        ),
        Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.charge("finance:finance_router_bot", 12.5, "desc")
    assert calls and calls[0].get("error") is True


def test_refund_logs_error_on_exception(monkeypatch, sbr_basic):
    sbr = sbr_basic
    calls = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **k: calls.append(k))

    def bad_create(*, api_key, **params):
        raise RuntimeError("fail")

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        Refund=types.SimpleNamespace(create=bad_create),
        Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.refund("finance:finance_router_bot", "ch_test", amount=5.0)
    assert calls and calls[0].get("error") is True


def test_create_subscription_logs_error_on_exception(monkeypatch, sbr_basic):
    sbr = sbr_basic
    calls = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **k: calls.append(k))

    def bad_create(*, api_key, **params):
        raise RuntimeError("fail")

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        Subscription=types.SimpleNamespace(create=bad_create),
        Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
        Price=types.SimpleNamespace(retrieve=lambda *a, **k: {"unit_amount": 1000}),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.create_subscription("finance:finance_router_bot", idempotency_key="sub-key")
    assert calls and calls[0].get("error") is True


def test_create_checkout_session_logs_error_on_exception(monkeypatch, sbr_basic):
    sbr = sbr_basic
    calls = []
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **k: calls.append(k))

    def bad_create(*, api_key, **params):
        raise RuntimeError("fail")

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        checkout=types.SimpleNamespace(Session=types.SimpleNamespace(create=bad_create)),
        Customer=types.SimpleNamespace(retrieve=lambda *a, **k: {}),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    line_items = [{"price": "price_finance_standard", "quantity": 1}]
    with pytest.raises(RuntimeError):
        sbr.create_checkout_session(
            "finance:finance_router_bot",
            line_items,
            amount=10.0,
            success_url="https://example.com/s",
            cancel_url="https://example.com/c",
            mode="payment",
        )
    assert calls and calls[0].get("error") is True


def test_charge_mismatched_destination_triggers_alert_and_rollback(
    monkeypatch, sbr_file_logger
):
    sbr, ledger = sbr_file_logger
    alerts: list = []
    monkeypatch.setattr(
        sbr.alert_dispatcher, "dispatch_alert", lambda *a, **k: alerts.append((a, k))
    )

    class RM:
        def __init__(self):
            self.called = False

        def rollback(self, tag, requesting_bot=None):
            self.called = True

        def auto_rollback(self, *args, **kwargs):
            self.rollback("latest", requesting_bot=args[0])

    rm = RM()
    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: rm)

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
    with pytest.raises(RuntimeError):
        sbr.charge("finance:finance_router_bot", 12.5, "desc")
    assert alerts and rm.called
    records = _read_records(ledger)
    assert any(
        r.get("action_type") == "mismatch" and r.get("error") == 1 for r in records
    )


def test_subscription_mismatched_destination_triggers_alert_and_rollback(
    monkeypatch, sbr_file_logger
):
    sbr, ledger = sbr_file_logger
    alerts: list = []
    monkeypatch.setattr(
        sbr.alert_dispatcher, "dispatch_alert", lambda *a, **k: alerts.append((a, k))
    )

    class RM:
        def __init__(self):
            self.called = False

        def rollback(self, tag, requesting_bot=None):
            self.called = True

        def auto_rollback(self, *args, **kwargs):
            self.rollback("latest", requesting_bot=args[0])

    rm = RM()
    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: rm)

    def fake_sub_create(*, api_key, **params):
        return {"id": "sub_test", "on_behalf_of": "acct_bad"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        Subscription=types.SimpleNamespace(create=fake_sub_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.create_subscription("finance:finance_router_bot", idempotency_key="sub-key")
    assert alerts and rm.called
    records = _read_records(ledger)
    assert any(
        r.get("action_type") == "mismatch" and r.get("error") == 1 for r in records
    )


def test_checkout_session_mismatched_destination_triggers_alert_and_rollback(
    monkeypatch, sbr_file_logger
):
    sbr, ledger = sbr_file_logger
    alerts: list = []
    monkeypatch.setattr(
        sbr.alert_dispatcher, "dispatch_alert", lambda *a, **k: alerts.append((a, k))
    )

    class RM:
        def __init__(self):
            self.called = False

        def rollback(self, tag, requesting_bot=None):
            self.called = True

        def auto_rollback(self, *args, **kwargs):
            self.rollback("latest", requesting_bot=args[0])

    rm = RM()
    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: rm)

    def fake_session_create(*, api_key, **params):
        return {"id": "cs_test", "on_behalf_of": "acct_bad", "amount_total": 1000}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        checkout=types.SimpleNamespace(
            Session=types.SimpleNamespace(create=fake_session_create)
        ),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.create_checkout_session(
            "finance:finance_router_bot", line_items=[{"price": "p", "quantity": 1}]
        )
    assert alerts and rm.called
    records = _read_records(ledger)
    assert any(
        r.get("action_type") == "mismatch" and r.get("error") == 1 for r in records
    )


def test_refund_mismatched_destination_triggers_alert_and_rollback(
    monkeypatch, sbr_file_logger
):
    sbr, ledger = sbr_file_logger
    alerts: list = []
    monkeypatch.setattr(
        sbr.alert_dispatcher, "dispatch_alert", lambda *a, **k: alerts.append((a, k))
    )

    class RM:
        def __init__(self):
            self.called = False

        def rollback(self, tag, requesting_bot=None):
            self.called = True

        def auto_rollback(self, *args, **kwargs):
            self.rollback("latest", requesting_bot=args[0])

    rm = RM()
    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: rm)

    def fake_refund_create(*, api_key, **params):
        return {"id": "rf_test", "on_behalf_of": "acct_bad", "amount": 500}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        Refund=types.SimpleNamespace(create=fake_refund_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.refund("finance:finance_router_bot", "ch_test", amount=5.0)
    assert alerts and rm.called
    records = _read_records(ledger)
    assert any(
        r.get("action_type") == "mismatch" and r.get("error") == 1 for r in records
    )


def test_unknown_key_triggers_alert_and_rollback(monkeypatch, sbr_file_logger):
    sbr, ledger = sbr_file_logger
    alerts = []
    monkeypatch.setattr(
        sbr.alert_dispatcher, "dispatch_alert", lambda *a, **k: alerts.append((a, k))
    )

    class RM:
        def __init__(self):
            self.called = False

        def rollback(self, tag, requesting_bot=None):
            self.called = True

        def auto_rollback(self, *args, **kwargs):
            self.rollback("latest", requesting_bot=args[0])

    rm = RM()
    monkeypatch.setattr(sbr.rollback_manager, "RollbackManager", lambda: rm)
    monkeypatch.setattr(sbr, "ALLOWED_SECRET_KEYS", {"sk_other"})
    with pytest.raises(RuntimeError):
        sbr.charge("finance:finance_router_bot", 12.5, "desc")
    assert alerts and rm.called
