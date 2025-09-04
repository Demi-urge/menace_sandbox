import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _import_module(monkeypatch):
    pkg = types.ModuleType("sbrpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["sbrpkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"sbrpkg.{name}", ROOT / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"sbrpkg.{name}"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    vsp = _load("vault_secret_provider")
    monkeypatch.setattr(
        vsp.VaultSecretProvider,
        "get",
        lambda self, name: {
            "stripe_secret_key": "sk_live_dummy",
            "stripe_public_key": "pk_live_dummy",
        }.get(name, ""),
    )

    return _load("stripe_billing_router")


def test_routing_matches_bot_metadata(monkeypatch):
    sbr = _import_module(monkeypatch)
    route = sbr._resolve_route("finance:finance_router_bot:monetization")
    assert route["product_id"] == "prod_finance_router"
    assert route["price_id"] == "price_finance_standard"
    assert route["customer_id"] == "cus_finance_default"
    assert route["secret_key"] == "sk_live_dummy"
    assert route["public_key"] == "pk_live_dummy"


def test_unsupported_domain_raises(monkeypatch):
    sbr = _import_module(monkeypatch)
    with pytest.raises(RuntimeError, match="Unsupported billing domain"):
        sbr._resolve_route("unknown:bot:category")


def test_unmatched_route_raises(monkeypatch):
    sbr = _import_module(monkeypatch)
    with pytest.raises(RuntimeError, match="No billing route"):
        sbr._resolve_route("finance:unknown_bot:monetization")


def test_missing_keys_raise(monkeypatch):
    sbr = _import_module(monkeypatch)
    monkeypatch.setattr(sbr, "STRIPE_SECRET_KEY", "")
    monkeypatch.setattr(sbr, "STRIPE_PUBLIC_KEY", "")
    with pytest.raises(RuntimeError):
        sbr._resolve_route("finance:finance_router_bot:monetization")


def test_override_updates_route(monkeypatch):
    sbr = _import_module(monkeypatch)
    sbr.register_override(
        "finance",
        "finance_router_bot",
        "monetization",
        key="region",
        value="eu",
        route={"price_id": "price_finance_eu"},
    )
    route = sbr._resolve_route(
        "finance:finance_router_bot:monetization", overrides={"region": "eu"}
    )
    assert route["price_id"] == "price_finance_eu"


def test_charge_and_customer_creation(monkeypatch):
    sbr = _import_module(monkeypatch)

    charge_params: dict[str, object] = {}
    customer_params: dict[str, object] = {}

    def fake_charge_create(**params):
        charge_params.update(params)
        return {"status": "succeeded", **params}

    def fake_customer_create(**params):
        customer_params.update(params)
        return {"id": "cus_test", **params}

    fake_stripe = types.SimpleNamespace(
        api_key="",
        Charge=types.SimpleNamespace(create=fake_charge_create),
        Customer=types.SimpleNamespace(create=fake_customer_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    charge = sbr.init_charge(
        "finance:finance_router_bot:monetization", 12.5, "desc"
    )
    assert charge["status"] == "succeeded"
    assert charge_params["amount"] == 1250
    assert charge_params["customer"] == "cus_finance_default"

    cust = sbr.create_customer(
        "finance:finance_router_bot:monetization", {"email": "a@b"}
    )
    assert cust["id"] == "cus_test"
    assert customer_params == {"email": "a@b"}
