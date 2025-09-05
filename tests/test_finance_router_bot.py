import importlib.machinery
import importlib.util
import json
import sys
import types

import pytest
import yaml
from dynamic_path_router import resolve_path

# tests previously mocked a deprecated ``stripe_handler`` module; ensure the
# modern ``stripe_billing_router`` is loaded and patched instead.


def _import_finance_router(monkeypatch, tmp_path):
    pkg = types.ModuleType("frbpkg")
    pkg.__path__ = [str(resolve_path("."))]
    pkg.__spec__ = importlib.machinery.ModuleSpec("frbpkg", loader=None, is_package=True)
    sys.modules["frbpkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"frbpkg.{name}", resolve_path(f"{name}.py")  # path-ignore
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"frbpkg.{name}"] = module
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    sys.modules["frbpkg.capital_management_bot"] = types.SimpleNamespace(
        CapitalManagementBot=type(
            "CM", (), {"log_inflow": lambda self, amount, model_id: None}
        )
    )
    sys.modules["frbpkg.unified_event_bus"] = types.SimpleNamespace(
        UnifiedEventBus=type(
            "UEB", (), {"subscribe": lambda *a, **k: None, "publish": lambda *a, **k: None}
        )
    )
    sys.modules["frbpkg.menace_memory_manager"] = types.SimpleNamespace(
        MenaceMemoryManager=type("MMM", (), {"subscribe": lambda *a, **k: None}),
        MemoryEntry=object,
    )

    vsp = _load("vault_secret_provider")
    monkeypatch.setattr(
        vsp.VaultSecretProvider,
        "get",
        lambda self, name: {
            "stripe_secret_key": "sk_live_dummy",
            "stripe_public_key": "pk_live_dummy",
            "stripe_allowed_secret_keys": "sk_live_dummy",
        }.get(name, ""),
    )
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
    dd = types.SimpleNamespace(
        DiscrepancyDB=lambda: types.SimpleNamespace(log=lambda *a, **k: None)
    )
    sys.modules["discrepancy_db"] = dd
    sys.modules["frbpkg.discrepancy_db"] = dd
    arm = types.SimpleNamespace(
        AutomatedRollbackManager=lambda: type(
            "ARM",
            (),
            {"auto_rollback": lambda self, tag, bots: None},
        )()
    )
    sys.modules["advanced_error_management"] = arm
    sys.modules["frbpkg.advanced_error_management"] = arm
    sbr = _load("stripe_billing_router")
    sys.modules["frbpkg.stripe_billing_router"] = sbr
    sys.modules["stripe_billing_router"] = sbr

    return _load("finance_router_bot")


def test_route_and_summary(tmp_path, monkeypatch):
    frb = _import_finance_router(monkeypatch, tmp_path)
    log = tmp_path / "payout.json"
    calls = {}

    def fake_charge(bot_id, amount, description=None, *, overrides=None):
        calls["bot_id"] = bot_id
        calls["amount"] = amount
        return {"status": "succeeded"}

    monkeypatch.setattr(frb.stripe_billing_router, "charge", fake_charge)
    bot = frb.FinanceRouterBot(payout_log_path=log)
    res = bot.route_payment(10.0, "model1")
    assert res
    assert calls == {"bot_id": frb.FinanceRouterBot.BOT_ID, "amount": 10.0}
    data = json.loads(log.read_text())
    assert data and data[0]["model_id"] == "model1"
    summary = bot.report_earnings_summary()
    assert summary["model1"] == 10.0


def test_router_error_propagates(tmp_path, monkeypatch):
    frb = _import_finance_router(monkeypatch, tmp_path)
    log = tmp_path / "payout.json"
    calls = {}

    def bad_charge(bot_id, amount, description=None, *, overrides=None):
        calls["bot_id"] = bot_id
        raise RuntimeError("boom")

    monkeypatch.setattr(frb.stripe_billing_router, "charge", bad_charge)
    bot = frb.FinanceRouterBot(payout_log_path=log)
    res = bot.route_payment(5.0, "model2")
    assert res.startswith("error:")
    assert calls["bot_id"] == frb.FinanceRouterBot.BOT_ID
    data = json.loads(log.read_text())
    assert data and data[0]["result"].startswith("error:")


def test_router_sanitizes_stripe_keys(tmp_path, monkeypatch, caplog):
    frb = _import_finance_router(monkeypatch, tmp_path)
    log = tmp_path / "payout.json"

    def bad_charge(bot_id, amount, description=None, *, overrides=None):
        raise RuntimeError("invalid sk_test_123 pk_live_456")

    monkeypatch.setattr(frb.stripe_billing_router, "charge", bad_charge)
    bot = frb.FinanceRouterBot(payout_log_path=log)
    with caplog.at_level("ERROR"):
        res = bot.route_payment(5.0, "model2")
    assert "sk_test_123" not in res and "pk_live_456" not in res
    assert "[REDACTED]" in res
    data = json.loads(log.read_text())
    assert "sk_test_123" not in data[0]["result"]
    assert "pk_live_456" not in data[0]["result"]
    assert "[REDACTED]" in data[0]["result"]
    joined = " ".join(r.message for r in caplog.records)
    assert "sk_test_123" not in joined
    assert "pk_live_456" not in joined
    assert "[REDACTED]" in joined


def _load_stripe_router(monkeypatch, tmp_path, routes):
    class StripeStub:
        class PaymentIntent:
            last_params = None

            @staticmethod
            def create(*, api_key=None, **params):
                StripeStub.PaymentIntent.last_params = {
                    "api_key": api_key,
                    **params,
                }
                return {"status": "ok"}

        class InvoiceItem:
            last_params = None

            @staticmethod
            def create(*, api_key=None, **params):
                StripeStub.InvoiceItem.last_params = {
                    "api_key": api_key,
                    **params,
                }
                return {"id": "ii_123"}

        class Invoice:
            last_params_create = None
            last_params_pay = None

            @staticmethod
            def create(*, api_key=None, **params):
                StripeStub.Invoice.last_params_create = {
                    "api_key": api_key,
                    **params,
                }
                return {"id": "in_123"}

            @staticmethod
            def pay(invoice_id, *, api_key=None, idempotency_key=None):
                StripeStub.Invoice.last_params_pay = {
                    "invoice_id": invoice_id,
                    "api_key": api_key,
                    "idempotency_key": idempotency_key,
                }
                return {"status": "paid"}

        class Balance:
            @staticmethod
            def retrieve(*, api_key=None):
                return {"available": [{"amount": 5000}]}

    monkeypatch.setitem(sys.modules, "stripe", StripeStub)
    vsp = types.SimpleNamespace(
        VaultSecretProvider=type(
            "VSP",
            (),
            {
                "get": lambda self, name: {
                    "stripe_secret_key": "sk_live_dummy",
                    "stripe_public_key": "pk_live_dummy",
                    "stripe_allowed_secret_keys": "sk_live_dummy",
                }.get(name, ""),
            },
        )
    )
    monkeypatch.setitem(sys.modules, "vault_secret_provider", vsp)
    monkeypatch.setenv("STRIPE_ALLOWED_SECRET_KEYS", "sk_live_dummy")
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PUBLIC_KEY", raising=False)
    cfg = tmp_path / "routes.yaml"
    cfg.write_text(yaml.safe_dump(routes))
    monkeypatch.setenv("STRIPE_ROUTING_CONFIG", str(cfg))
    arm = types.SimpleNamespace(
        AutomatedRollbackManager=lambda: type(
            "ARM",
            (),
            {"auto_rollback": lambda self, tag, bots: None},
        )()
    )
    sys.modules["advanced_error_management"] = arm
    sys.modules.pop("stripe_billing_router", None)
    spec = importlib.util.spec_from_file_location(
        "stripe_billing_router", resolve_path("stripe_billing_router.py")  # path-ignore
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["stripe_billing_router"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "_get_account_id", lambda api_key: module.STRIPE_MASTER_ACCOUNT_ID
    )
    monkeypatch.setattr(module.billing_logger, "log_event", lambda **kw: None)
    monkeypatch.setattr(module, "_verify_route", lambda *a, **k: None)
    return module, StripeStub


def test_stripe_router_charge_and_balance(monkeypatch, tmp_path):
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
    sbr, stripe_stub = _load_stripe_router(monkeypatch, tmp_path, routes)
    sbr.charge("stripe:finance:finance_router_bot", amount=2.5)
    params = stripe_stub.InvoiceItem.last_params
    assert params["api_key"] == "sk_live_dummy"
    assert params["price"] == "price_finance_standard"
    assert params["customer"] == "cus_finance_default"
    assert sbr.get_balance("stripe:finance:finance_router_bot") == 50.0


def test_stripe_router_missing_and_misconfigured(monkeypatch, tmp_path):
    sbr, _ = _load_stripe_router(monkeypatch, tmp_path, {})
    with pytest.raises(RuntimeError):
        sbr.charge("stripe:finance:unknown", amount=1.0)

    routes = {
        "stripe": {
            "default": {
                "finance": {"bad_bot": {"price_id": "price_only"}}
            }
        }
    }
    with pytest.raises(RuntimeError):
        _load_stripe_router(monkeypatch, tmp_path, routes)


def test_sanitize_partially_masked_keys(monkeypatch, tmp_path) -> None:
    frb = _import_finance_router(monkeypatch, tmp_path)
    text = "foo sk_live_1234****5678 bar pk_test_abcd****xyz"
    sanitized = frb._sanitize_stripe_keys(text)
    assert sanitized == "foo [REDACTED] bar [REDACTED]"
