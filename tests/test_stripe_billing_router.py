import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _import_module(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setenv("STRIPE_PUBLIC_KEY", "pk_test")
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

    _load("stripe_handler")
    return _load("stripe_billing_router")


def test_routing_matches_bot_metadata(monkeypatch):
    sbr = _import_module(monkeypatch)
    route = sbr._resolve_route("finance:finance_router_bot:monetization")
    assert route["product_id"] == "prod_finance_router"
    assert route["price_id"] == "price_finance_standard"
    assert route["customer_id"] == "cus_finance_default"
    assert route["secret_key"] == "sk_test"
    assert route["public_key"] == "pk_test"


def test_unmatched_route_raises(monkeypatch):
    sbr = _import_module(monkeypatch)
    with pytest.raises(RuntimeError):
        sbr._resolve_route("unknown:bot:category")


def test_missing_keys_raise(monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PUBLIC_KEY", raising=False)
    pkg = types.ModuleType("sbrpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["sbrpkg"] = pkg
    spec = importlib.util.spec_from_file_location(
        "sbrpkg.stripe_billing_router", ROOT / "stripe_billing_router.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sbrpkg.stripe_billing_router"] = module
    assert spec.loader is not None
    with pytest.raises(RuntimeError):
        spec.loader.exec_module(module)


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
