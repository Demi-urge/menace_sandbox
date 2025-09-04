import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _import_module(monkeypatch, secrets=None):
    pkg = types.ModuleType("sbrpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["sbrpkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"sbrpkg.{name}", ROOT / f"{name}.py"
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
    vsp = _load("vault_secret_provider")
    monkeypatch.setattr(
        vsp.VaultSecretProvider, "get", lambda self, n: secrets.get(n, "")
    )
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PUBLIC_KEY", raising=False)
    return _load("stripe_billing_router")


def test_successful_route_and_charge(monkeypatch):
    sbr = _import_module(monkeypatch)

    recorded: dict[str, object] = {}

    def fake_charge_create(**params):
        recorded.update(params)
        return {"id": "ch_test", **params}

    fake_stripe = types.SimpleNamespace(
        api_key="",
        Charge=types.SimpleNamespace(create=fake_charge_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    route = sbr._resolve_route("finance:finance_router_bot")
    assert route["product_id"] == "prod_finance_router"

    res = sbr.charge("finance:finance_router_bot", 12.5, "desc")
    assert res["id"] == "ch_test"
    assert recorded["amount"] == 1250
    assert recorded["customer"] == "cus_finance_default"


def test_missing_keys_or_rule(monkeypatch):
    sbr = _import_module(monkeypatch)

    monkeypatch.setattr(sbr, "STRIPE_SECRET_KEY", "")
    monkeypatch.setattr(sbr, "STRIPE_PUBLIC_KEY", "")
    with pytest.raises(RuntimeError):
        sbr._resolve_route("finance:finance_router_bot")

    with pytest.raises(RuntimeError, match="No billing route"):
        sbr._resolve_route("finance:unknown_bot")


def test_region_and_business_overrides(monkeypatch):
    sbr = _import_module(monkeypatch)
    sbr.register_route(
        "finance",
        "finance_router_bot",
        {
            "product_id": "prod_finance_eu",
            "price_id": "price_finance_eu",
            "customer_id": "cus_finance_eu",
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

