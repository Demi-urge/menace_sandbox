import importlib.util
import importlib.machinery
import sys
import types
import threading

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
    dd = types.SimpleNamespace(
        DiscrepancyDB=lambda: types.SimpleNamespace(
            log=lambda *a, **k: None, add=lambda *a, **k: None
        )
    )
    sys.modules["discrepancy_db"] = dd
    sys.modules["sbrpkg.discrepancy_db"] = dd
    monkeypatch.setattr(
        vsp.VaultSecretProvider, "get", lambda self, n: secrets.get(n, "")
    )
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PUBLIC_KEY", raising=False)
    # ``STRIPE_MASTER_ACCOUNT_ID`` is hardcoded in the module but we ensure the
    # environment does not leak an alternative value into tests.
    monkeypatch.delenv("STRIPE_MASTER_ACCOUNT_ID", raising=False)
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
    monkeypatch.setattr(sbr.billing_logger, "log_event", lambda **kw: None)
    monkeypatch.setattr(sbr, "log_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(
        sbr, "_get_account_id", lambda api_key: sbr.STRIPE_MASTER_ACCOUNT_ID
    )
    return sbr


def test_env_file_provides_stripe_keys(monkeypatch, tmp_path):
    pytest.importorskip("dotenv")
    env_file = tmp_path / "custom.env"
    env_file.write_text("STRIPE_SECRET_KEY=sk_live_env\nSTRIPE_PUBLIC_KEY=pk_live_env\n")
    monkeypatch.setenv("MENACE_ENV_FILE", str(env_file))

    class RaisingSecrets(dict):
        def __bool__(self) -> bool:  # pragma: no cover - simple helper
            return True

        def get(self, key, default=None):  # pragma: no cover - simple helper
            raise AssertionError(f"vault lookup attempted for {key}")

    sbr = _import_module(monkeypatch, tmp_path, secrets=RaisingSecrets())
    assert sbr.STRIPE_SECRET_KEY == "sk_live_env"
    assert sbr.STRIPE_PUBLIC_KEY == "pk_live_env"


@pytest.fixture
def sbr_module(monkeypatch, tmp_path):
    """Return a fresh ``stripe_billing_router`` module for each test."""
    return _import_module(monkeypatch, tmp_path)


@pytest.fixture
def mock_customer_api(monkeypatch, sbr_module):
    """Patch ``stripe.Customer.create`` and record parameters."""
    recorded: dict[str, object] = {}

    def fake_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "cus_test"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig", Customer=types.SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(sbr_module, "stripe", fake_stripe)
    return recorded, fake_stripe


@pytest.fixture
def mock_subscription_api(monkeypatch, sbr_module):
    """Patch ``stripe.Subscription.create`` and record parameters."""
    recorded: dict[str, object] = {}

    def fake_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "sub_test"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig", Subscription=types.SimpleNamespace(create=fake_create)
    )
    monkeypatch.setattr(sbr_module, "stripe", fake_stripe)
    return recorded, fake_stripe


def test_successful_route_and_charge(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    item: dict[str, object] = {}
    invoice_create: dict[str, object] = {}
    invoice_pay: dict[str, object] = {}

    monkeypatch.setattr(sbr.time, "time", lambda: 1700000000.0)

    def fake_invoice_item_create(*, api_key: str, **params):
        item.update(params)
        item["api_key"] = api_key
        return {"id": "ii_test", **params}

    def fake_invoice_create(*, api_key: str, **params):
        invoice_create.update(params)
        invoice_create["api_key"] = api_key
        return {"id": "in_test", **params}

    def fake_invoice_pay(invoice_id, *, api_key: str, **params):
        invoice_pay["invoice_id"] = invoice_id
        invoice_pay["api_key"] = api_key
        invoice_pay.update(params)
        return {"id": invoice_id, "status": "paid"}

    fake_stripe = types.SimpleNamespace(
        api_key="original",
        InvoiceItem=types.SimpleNamespace(create=fake_invoice_item_create),
        Invoice=types.SimpleNamespace(create=fake_invoice_create, pay=fake_invoice_pay),
        PaymentIntent=types.SimpleNamespace(
            create=lambda **kw: {
                "id": "pi",
                "status": "paid",
                "on_behalf_of": sbr.STRIPE_MASTER_ACCOUNT_ID,
            }
        ),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    log_record: dict[str, object] = {}
    monkeypatch.setattr(
        sbr, "log_billing_event", lambda action, **kw: log_record.update({"action": action, **kw})
    )

    route = sbr._resolve_route("finance:finance_router_bot")
    assert route["product_id"] == "prod_finance_router"

    res = sbr.charge("finance:finance_router_bot", 12.5, "desc")
    expected_key = "finance:finance_router_bot-12.5-1700000000000"
    assert res["status"] == "paid"
    assert item["price"] == "price_finance_standard"
    assert item["customer"] == "cus_finance_default"
    assert item["api_key"] == "sk_live_dummy"
    assert invoice_create["customer"] == "cus_finance_default"
    assert invoice_create["api_key"] == "sk_live_dummy"
    assert invoice_pay["invoice_id"] == "in_test"
    assert invoice_pay["api_key"] == "sk_live_dummy"
    assert item["idempotency_key"] == expected_key
    assert invoice_create["idempotency_key"] == expected_key
    assert invoice_pay["idempotency_key"] == expected_key
    assert fake_stripe.api_key == "original"
    assert log_record["action"] == "charge"
    assert log_record["bot_id"] == "finance:finance_router_bot"


def test_charge_uses_payment_intent_when_no_price(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    # remove price to force PaymentIntent path
    sbr.ROUTING_TABLE[("stripe", "default", "finance", "finance_router_bot")] = {
        "product_id": "prod_finance_router",
        "customer_id": "cus_finance_default",
    }
    monkeypatch.setattr(sbr.time, "time", lambda: 1700000000.0)

    recorded: dict[str, object] = {}

    def fake_pi_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "pi_test", **params}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        PaymentIntent=types.SimpleNamespace(create=fake_pi_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    res = sbr.charge("finance:finance_router_bot", 12.5, "desc")
    expected_key = "finance:finance_router_bot-12.5-1700000000000"
    assert res["id"] == "pi_test"
    assert recorded["amount"] == 1250
    assert recorded["customer"] == "cus_finance_default"
    assert recorded["api_key"] == "sk_live_dummy"
    assert recorded["currency"] == "usd"
    assert recorded["idempotency_key"] == expected_key
    assert fake_stripe.api_key == "orig"


def test_charge_uses_currency_from_route(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    # remove price to force PaymentIntent path and set non-default currency
    sbr.ROUTING_TABLE[("stripe", "default", "finance", "finance_router_bot")] = {
        "product_id": "prod_finance_router",
        "customer_id": "cus_finance_default",
        "currency": "eur",
    }
    monkeypatch.setattr(sbr.time, "time", lambda: 1700000000.0)

    recorded: dict[str, object] = {}

    def fake_pi_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "pi_test", **params}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        PaymentIntent=types.SimpleNamespace(create=fake_pi_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    res = sbr.charge("finance:finance_router_bot", 10.0, "desc")
    expected_key = "finance:finance_router_bot-10.0-1700000000000"
    assert res["id"] == "pi_test"
    assert recorded["currency"] == "eur"
    assert recorded["api_key"] == "sk_live_dummy"
    assert recorded["idempotency_key"] == expected_key
    assert fake_stripe.api_key == "orig"


def test_charge_amount_validation(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    sbr.ROUTING_TABLE[("stripe", "default", "finance", "finance_router_bot")] = {
        "product_id": "prod_finance_router",
        "customer_id": "cus_finance_default",
    }

    calls: list[object] = []

    def fake_pi_create(*, api_key: str, **params):
        calls.append(params)
        return {"id": "pi_test"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        PaymentIntent=types.SimpleNamespace(create=fake_pi_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    with pytest.raises(ValueError):
        sbr.charge("finance:finance_router_bot", 0)
    with pytest.raises(ValueError):
        sbr.charge("finance:finance_router_bot", -1)
    with pytest.raises(ValueError):
        sbr.charge("finance:finance_router_bot", "bad")  # type: ignore[arg-type]

    assert calls == []


def test_get_balance_error(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    monkeypatch.setattr(sbr, "_client", lambda api_key: None)

    def bad_retrieve(*args, **kwargs):
        raise ValueError("boom")

    fake_stripe = types.SimpleNamespace(Balance=types.SimpleNamespace(retrieve=bad_retrieve))
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    with pytest.raises(RuntimeError):
        sbr.get_balance("finance:finance_router_bot")


def test_missing_keys_or_rule(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    monkeypatch.setattr(sbr, "STRIPE_SECRET_KEY", "")
    monkeypatch.setattr(sbr, "STRIPE_PUBLIC_KEY", "")
    with pytest.raises(RuntimeError):
        sbr._resolve_route("finance:finance_router_bot")

    with pytest.raises(RuntimeError, match="No billing route"):
        sbr._resolve_route("finance:unknown_bot")


def test_region_and_business_overrides(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    sbr.register_route(
        "finance",
        "finance_router_bot",
        {
            "product_id": "prod_finance_eu",
            "price_id": "price_finance_eu",
            "customer_id": "cus_finance_eu",
            "currency": "eur",
        },
        region="eu",
    )
    sbr.register_override(
        {
            "business_category": "finance",
            "bot_name": "finance_router_bot",
            "key": "tier",
            "value": "enterprise",
            "route": {"price_id": "price_finance_enterprise"},
        }
    )
    route = sbr._resolve_route(
        "finance:finance_router_bot", overrides={"region": "eu", "tier": "enterprise"}
    )
    assert route["product_id"] == "prod_finance_eu"
    assert route["customer_id"] == "cus_finance_eu"
    assert route["price_id"] == "price_finance_enterprise"
    assert route["currency"] == "eur"


def test_domain_routing_and_invalid_domain(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    sbr.register_route(
        "finance",
        "finance_router_bot",
        {
            "product_id": "prod_alt",
            "price_id": "price_alt",
            "customer_id": "cus_alt",
        },
        domain="alt",
    )
    route = sbr._resolve_route("alt:finance:finance_router_bot")
    assert route["customer_id"] == "cus_alt"
    with pytest.raises(RuntimeError, match="Unsupported billing domain"):
        sbr._resolve_route("unknown:finance:finance_router_bot")


def test_key_override_errors(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        sbr._resolve_route(
            "finance:finance_router_bot",
            overrides={"secret_key": "sk_live_other"},
        )
    with pytest.raises(RuntimeError):
        sbr._resolve_route(
            "finance:finance_router_bot",
            overrides={"public_key": "pk_live_other"},
        )


def test_register_rejects_api_keys(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        sbr.register_route(
            "finance",
            "finance_router_bot",
            {"secret_key": "sk_live_other"},
        )
    with pytest.raises(ValueError):
        sbr.register_override(
            {
                "business_category": "finance",
                "bot_name": "finance_router_bot",
                "key": "tier",
                "value": "enterprise",
                "route": {"public_key": "pk_live_other"},
            }
        )


def test_load_routing_table_missing_required_keys(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    bad = {
        "stripe": {
            "default": {
                "finance": {
                    "finance_router_bot": {
                        "price_id": "price_finance_standard",
                        "customer_id": "cus_finance_default",
                    }
                }
            }
        }
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(RuntimeError, match="product_id"):
        sbr._load_routing_table(str(path))


def test_load_routing_table_empty_values(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    bad = {
        "stripe": {
            "default": {
                "finance": {
                    "finance_router_bot": {
                        "product_id": "",
                        "price_id": "price_finance_standard",
                        "customer_id": "cus_finance_default",
                    }
                }
            }
        }
    }
    path = tmp_path / "empty.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(RuntimeError, match="non-empty"):
        sbr._load_routing_table(str(path))


def test_concurrent_client_isolation(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    calls: list[str] = []

    def fake_pi_create(*, api_key: str, **params):
        calls.append(api_key)
        return {"id": "pi", **params}

    fake_stripe = types.SimpleNamespace(
        api_key="unchanged",
        PaymentIntent=types.SimpleNamespace(create=fake_pi_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    def fake_resolve(bot_id: str, overrides=None):  # type: ignore[override]
        return {
            "secret_key": f"sk_live_{bot_id}",
            "public_key": "pk_live_dummy",
            "product_id": "prod",
        }

    monkeypatch.setattr(sbr, "_resolve_route", fake_resolve)
    sbr.ALLOWED_SECRET_KEYS.update({"sk_live_bot1", "sk_live_bot2"})

    def worker(bot_id: str) -> None:
        sbr.charge(bot_id, 1.0)

    t1 = threading.Thread(target=worker, args=("bot1",))
    t2 = threading.Thread(target=worker, args=("bot2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert set(calls) == {"sk_live_bot1", "sk_live_bot2"}
    assert fake_stripe.api_key == "unchanged"


def test_create_subscription_success(sbr_module, mock_subscription_api):
    recorded, fake_stripe = mock_subscription_api
    res = sbr_module.create_subscription(
        "finance:finance_router_bot", idempotency_key="sub-key"
    )
    assert res["id"] == "sub_test"
    assert recorded["customer"] == "cus_finance_default"
    assert recorded["items"][0]["price"] == "price_finance_standard"
    assert recorded["api_key"] == "sk_live_dummy"
    assert recorded["idempotency_key"] == "sub-key"
    assert fake_stripe.api_key == "orig"


def test_price_based_charge_without_amount(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    item: dict[str, object] = {}
    invoice_create: dict[str, object] = {}
    invoice_pay: dict[str, object] = {}

    monkeypatch.setattr(sbr.time, "time", lambda: 1700000000.0)

    def fake_invoice_item_create(*, api_key: str, **params):
        item.update(params)
        item["api_key"] = api_key
        return {"id": "ii", **params}

    def fake_invoice_create(*, api_key: str, **params):
        invoice_create.update(params)
        invoice_create["api_key"] = api_key
        return {"id": "in", **params}

    def fake_invoice_pay(invoice_id, *, api_key: str, **params):
        invoice_pay.update(params)
        invoice_pay["invoice_id"] = invoice_id
        invoice_pay["api_key"] = api_key
        return {"id": invoice_id, "status": "paid"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        InvoiceItem=types.SimpleNamespace(create=fake_invoice_item_create),
        Invoice=types.SimpleNamespace(create=fake_invoice_create, pay=fake_invoice_pay),
        PaymentIntent=types.SimpleNamespace(
            create=lambda **kw: {
                "id": "pi",
                "status": "paid",
                "on_behalf_of": sbr.STRIPE_MASTER_ACCOUNT_ID,
            }
        ),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)
    res = sbr.charge("finance:finance_router_bot", amount=5.0)
    expected_key = "finance:finance_router_bot-5.0-1700000000000"
    assert res["status"] == "paid"
    assert item["price"] == "price_finance_standard"
    assert item["customer"] == "cus_finance_default"
    assert item["api_key"] == "sk_live_dummy"
    assert invoice_create["customer"] == "cus_finance_default"
    assert invoice_create["api_key"] == "sk_live_dummy"
    assert invoice_pay["invoice_id"] == "in"
    assert invoice_pay["api_key"] == "sk_live_dummy"
    assert item["idempotency_key"] == expected_key
    assert invoice_create["idempotency_key"] == expected_key
    assert invoice_pay["idempotency_key"] == expected_key
    assert fake_stripe.api_key == "orig"


def test_charge_currency_override(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    sbr.ROUTING_TABLE[("stripe", "default", "finance", "finance_router_bot")] = {
        "product_id": "prod_finance_router",
        "customer_id": "cus_finance_default",
    }
    sbr.register_override(
        {
            "business_category": "finance",
            "bot_name": "finance_router_bot",
            "key": "currency",
            "value": "eur",
            "route": {"currency": "eur"},
        }
    )

    monkeypatch.setattr(sbr.time, "time", lambda: 1700000000.0)

    recorded: dict[str, object] = {}

    def fake_pi_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "pi_test"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        PaymentIntent=types.SimpleNamespace(create=fake_pi_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    res = sbr.charge(
        "finance:finance_router_bot", 5.0, "desc", overrides={"currency": "eur"}
    )
    expected_key = "finance:finance_router_bot-5.0-1700000000000"
    assert res["id"] == "pi_test"
    assert recorded["currency"] == "eur"
    assert recorded["api_key"] == "sk_live_dummy"
    assert recorded["idempotency_key"] == expected_key
    assert fake_stripe.api_key == "orig"


def test_price_route_negative_amount(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    calls: list[object] = []

    def fake_invoice_item_create(*args, **kwargs):
        calls.append("item")

    fake_stripe = types.SimpleNamespace(
        InvoiceItem=types.SimpleNamespace(create=fake_invoice_item_create),
        Invoice=types.SimpleNamespace(
            create=lambda **kw: {"id": "in"},
            pay=lambda *a, **k: {"id": a[0]},
        ),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    with pytest.raises(ValueError):
        sbr.charge("finance:finance_router_bot", -5.0)

    assert calls == []


def test_get_balance_client_error(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    def bad_retrieve():
        raise ValueError("boom")

    class FakeClient:
        def __init__(self, api_key: str):
            self.Balance = types.SimpleNamespace(retrieve=bad_retrieve)

    fake_stripe = types.SimpleNamespace(StripeClient=FakeClient)
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    with pytest.raises(RuntimeError):
        sbr.get_balance("finance:finance_router_bot")


def test_create_customer_success(sbr_module, mock_customer_api):
    recorded, fake_stripe = mock_customer_api
    info = {"email": "bot@example.com", "idempotency_key": "cust-key"}
    res = sbr_module.create_customer("finance:finance_router_bot", info)
    assert res["id"] == "cus_test"
    assert recorded["email"] == "bot@example.com"
    assert recorded["api_key"] == "sk_live_dummy"
    assert recorded["idempotency_key"] == "cust-key"
    assert fake_stripe.api_key == "orig"


def test_create_customer_missing_required_field(monkeypatch, sbr_module):
    def bad_create(*, api_key: str, **params):
        if "email" not in params:
            raise ValueError("email required")
        return {"id": "cus_test"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig", Customer=types.SimpleNamespace(create=bad_create)
    )
    monkeypatch.setattr(sbr_module, "stripe", fake_stripe)
    with pytest.raises(ValueError, match="email required"):
        sbr_module.create_customer("finance:finance_router_bot", {})


def test_create_subscription_missing_fields(monkeypatch, sbr_module):
    key = ("stripe", "default", "finance", "finance_router_bot")
    sbr_module.ROUTING_TABLE[key] = {"product_id": "prod_finance_router"}
    monkeypatch.setattr(sbr_module, "stripe", types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="price_id and customer_id are required"):
        sbr_module.create_subscription("finance:finance_router_bot")


def test_create_subscription_logs_event(sbr_module, mock_subscription_api, monkeypatch):
    recorded, fake_stripe = mock_subscription_api
    log_record: dict[str, object] = {}
    monkeypatch.setattr(
        sbr_module.billing_logger, "log_event", lambda **kw: log_record.update(kw)
    )
    log_db: dict[str, object] = {}
    monkeypatch.setattr(
        sbr_module,
        "log_billing_event",
        lambda action, **kw: log_db.update({"action": action, **kw}),
    )
    res = sbr_module.create_subscription(
        "finance:finance_router_bot", idempotency_key="sub-key"
    )
    assert res["id"] == "sub_test"
    assert log_record["action_type"] == "subscription"
    assert log_record["id"] == "sub_test"
    assert log_db["action"] == "subscription"
    assert log_db["bot_id"] == "finance:finance_router_bot"


def test_refund_success(monkeypatch, sbr_module):
    recorded: dict[str, object] = {}

    def fake_refund_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "rf_test", "amount": 500}

    fake_stripe = types.SimpleNamespace(
        api_key="orig", Refund=types.SimpleNamespace(create=fake_refund_create)
    )
    monkeypatch.setattr(sbr_module, "stripe", fake_stripe)
    log_record: dict[str, object] = {}
    monkeypatch.setattr(
        sbr_module.billing_logger, "log_event", lambda **kw: log_record.update(kw)
    )
    log_db: dict[str, object] = {}
    monkeypatch.setattr(
        sbr_module,
        "log_billing_event",
        lambda action, **kw: log_db.update({"action": action, **kw}),
    )
    ledger_calls: list[tuple] = []
    monkeypatch.setattr(
        sbr_module, "_log_payment", lambda *a: ledger_calls.append(a)
    )
    res = sbr_module.refund(
        "finance:finance_router_bot",
        "ch_test",
        amount=5.0,
        user_email="user@example.com",
        reason="requested_by_customer",
    )
    assert res["id"] == "rf_test"
    assert recorded["payment_intent"] == "ch_test"
    assert recorded["amount"] == 500
    assert recorded["api_key"] == "sk_live_dummy"
    assert log_record["action_type"] == "refund"
    assert log_record["id"] == "rf_test"
    assert log_db["action"] == "refund"
    assert log_db["bot_id"] == "finance:finance_router_bot"
    assert ledger_calls and ledger_calls[0][0] == "refund"


def test_create_checkout_session_success(monkeypatch, sbr_module):
    recorded: dict[str, object] = {}

    def fake_session_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "cs_test", "amount_total": 1000}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        checkout=types.SimpleNamespace(
            Session=types.SimpleNamespace(create=fake_session_create)
        ),
    )
    monkeypatch.setattr(sbr_module, "stripe", fake_stripe)
    log_record: dict[str, object] = {}
    monkeypatch.setattr(
        sbr_module.billing_logger, "log_event", lambda **kw: log_record.update(kw)
    )
    log_db: dict[str, object] = {}
    monkeypatch.setattr(
        sbr_module,
        "log_billing_event",
        lambda action, **kw: log_db.update({"action": action, **kw}),
    )
    ledger_calls: list[tuple] = []
    monkeypatch.setattr(
        sbr_module, "_log_payment", lambda *a: ledger_calls.append(a)
    )
    line_items = [{"price": "price_finance_standard", "quantity": 1}]
    res = sbr_module.create_checkout_session(
        "finance:finance_router_bot",
        line_items,
        mode="payment",
        success_url="https://example.com/s",
        cancel_url="https://example.com/c",
    )
    assert res["id"] == "cs_test"
    assert recorded["line_items"] == line_items
    assert recorded["customer"] == "cus_finance_default"
    assert recorded["api_key"] == "sk_live_dummy"
    assert log_record["action_type"] == "checkout_session"
    assert log_record["id"] == "cs_test"
    assert log_db["action"] == "checkout_session"
    assert log_db["bot_id"] == "finance:finance_router_bot"
    assert ledger_calls and ledger_calls[0][0] == "checkout"


def test_invalid_secret_key_triggers_alert(monkeypatch, sbr_module):
    called: dict[str, object] = {}

    def fake_alert(bot_id: str, account_id: str, **_: object) -> None:
        called["bot_id"] = bot_id
        called["account_id"] = account_id

    monkeypatch.setattr(sbr_module, "_alert_mismatch", fake_alert)

    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr_module._verify_route(
            "finance:finance_router_bot", {"secret_key": "sk_live_bad"}
        )

    assert called == {"bot_id": "finance:finance_router_bot", "account_id": "unknown"}


def test_invalid_account_triggers_alert(monkeypatch, sbr_module):
    called: list[tuple[str, str]] = []

    def fake_alert(bot_id: str, account_id: str, **_: object) -> None:
        called.append((bot_id, account_id))

    monkeypatch.setattr(sbr_module, "_alert_mismatch", fake_alert)

    route = {
        "secret_key": sbr_module.STRIPE_SECRET_KEY,
        "account_id": "acct_bad",
    }
    with pytest.raises(RuntimeError, match="Stripe account mismatch"):
        sbr_module._verify_route("finance:finance_router_bot", route)

    assert called == [("finance:finance_router_bot", "acct_bad")]
