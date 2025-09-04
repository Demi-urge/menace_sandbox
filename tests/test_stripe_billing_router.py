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

    recorded: dict[str, object] = {}

    def fake_charge_create(*, api_key: str, **params):
        recorded.update(params)
        recorded["api_key"] = api_key
        return {"id": "ch_test", **params}

    fake_stripe = types.SimpleNamespace(
        api_key="original",
        Charge=types.SimpleNamespace(create=fake_charge_create),
    )
    monkeypatch.setattr(sbr, "stripe", fake_stripe)

    route = sbr._resolve_route("stripe:finance:finance_router_bot")
    assert route["product_id"] == "prod_finance_router"

    res = sbr.charge("stripe:finance:finance_router_bot", 12.5, "desc")
    assert res["id"] == "ch_test"
    assert recorded["amount"] == 1250
    assert recorded["customer"] == "cus_finance_default"
    assert recorded["api_key"] == "sk_live_dummy"
    assert fake_stripe.api_key == "original"


def test_missing_keys_or_rule(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)

    monkeypatch.setattr(sbr, "STRIPE_SECRET_KEY", "")
    monkeypatch.setattr(sbr, "STRIPE_PUBLIC_KEY", "")
    with pytest.raises(RuntimeError):
        sbr._resolve_route("stripe:finance:finance_router_bot")

    with pytest.raises(RuntimeError, match="No billing route"):
        sbr._resolve_route("stripe:finance:unknown_bot")


def test_region_and_business_overrides(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    sbr.register_route(
        "stripe",
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
            "domain": "stripe",
            "business_category": "finance",
            "bot_name": "finance_router_bot",
            "key": "tier",
            "value": "enterprise",
            "route": {"price_id": "price_finance_enterprise"},
        }
    )
    route = sbr._resolve_route(
        "stripe:finance:finance_router_bot", overrides={"region": "eu", "tier": "enterprise"}
    )
    assert route["product_id"] == "prod_finance_eu"
    assert route["customer_id"] == "cus_finance_eu"
    assert route["price_id"] == "price_finance_enterprise"


def test_domain_routing_and_invalid_domain(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    sbr.register_route(
        "alt",
        "finance",
        "finance_router_bot",
        {
            "product_id": "prod_alt",
            "price_id": "price_alt",
            "customer_id": "cus_alt",
        },
    )
    route = sbr._resolve_route("alt:finance:finance_router_bot")
    assert route["customer_id"] == "cus_alt"
    with pytest.raises(RuntimeError, match="Unsupported billing domain"):
        sbr._resolve_route("unknown:finance:finance_router_bot")


def test_key_override_errors(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        sbr._resolve_route(
            "stripe:finance:finance_router_bot",
            overrides={"secret_key": "sk_live_other"},
        )
    with pytest.raises(RuntimeError):
        sbr._resolve_route(
            "stripe:finance:finance_router_bot",
            overrides={"public_key": "pk_live_other"},
        )


def test_register_rejects_api_keys(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        sbr.register_route(
            "stripe",
            "finance",
            "finance_router_bot",
            {"secret_key": "sk_live_other"},
        )
    with pytest.raises(ValueError):
        sbr.register_override(
            {
                "domain": "stripe",
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

    def fake_charge_create(*, api_key: str, **params):
        calls.append(api_key)
        return {"id": "ch", **params}

    fake_stripe = types.SimpleNamespace(
        api_key="unchanged",
        Charge=types.SimpleNamespace(create=fake_charge_create),
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
