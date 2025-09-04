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
            f"sbrpkg.{name}", resolve_path(f"{name}.py")
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
    monkeypatch.setattr(
        vsp.VaultSecretProvider, "get", lambda self, n: secrets.get(n, "")
    )
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PUBLIC_KEY", raising=False)
    return _load("stripe_billing_router")


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
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

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


def test_create_subscription(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    recorded: dict[str, object] = {}

    def fake_sub_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "sub_test"}

    fake_stripe = types.SimpleNamespace(
        api_key="orig",
        Subscription=types.SimpleNamespace(create=fake_sub_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    res = sbr.create_subscription("finance:finance_router_bot")
    assert res["id"] == "sub_test"
    assert recorded["customer"] == "cus_finance_default"
    assert recorded["items"][0]["price"] == "price_finance_standard"
    assert recorded["api_key"] == "sk_live_dummy"
    assert fake_stripe.api_key == "orig"
