import importlib.machinery
import importlib
import importlib.util
import sys
import types

import pytest
import yaml
from dynamic_path_router import resolve_path

# Tests previously mocked a deprecated ``stripe_handler``.  These tests ensure
# ``stripe_billing_router`` is patched instead.


def _import_investment_engine(monkeypatch, tmp_path):
    pkg = types.ModuleType("iepkg")
    pkg.__path__ = [str(resolve_path("."))]
    pkg.__spec__ = importlib.machinery.ModuleSpec("iepkg", loader=None, is_package=True)
    sys.modules["iepkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"iepkg.{name}", resolve_path(f"{name}.py")  # path-ignore
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"iepkg.{name}"] = module
        sys.modules[name] = module
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
    sys.modules["iepkg.discrepancy_db"] = dd
    arm = types.SimpleNamespace(
        AutomatedRollbackManager=lambda: type(
            "ARM",
            (),
            {"auto_rollback": lambda self, tag, bots: None},
        )()
    )
    sys.modules["advanced_error_management"] = arm
    sys.modules["iepkg.advanced_error_management"] = arm
    sbr = _load("stripe_billing_router")
    sys.modules["iepkg.stripe_billing_router"] = sbr
    sys.modules["stripe_billing_router"] = sbr
    dbm = _load("db_router")
    dbm.GLOBAL_ROUTER = None
    sys.modules["db_router"] = dbm
    return _load("investment_engine")


def test_reinvest_cap(monkeypatch, tmp_path):
    ie = _import_investment_engine(monkeypatch, tmp_path)
    db = ie.InvestmentDB(tmp_path / "i.db")
    engine = ie.PredictiveSpendEngine(db)
    bot = ie.AutoReinvestmentBot(
        cap_percentage=0.5,
        safety_reserve=0.0,
        minimum_threshold=10.0,
        predictor=engine,
        db=db,
    )
    calls = {}

    def fake_balance(bot_id, *a, **k):
        calls["bal"] = bot_id
        return 200.0

    def fake_charge(bot_id, amount, description=None, *, overrides=None):
        calls["charge"] = bot_id
        return {"status": "succeeded"}

    monkeypatch.setattr(ie.stripe_billing_router, "get_balance", fake_balance)
    monkeypatch.setattr(ie.stripe_billing_router, "charge", fake_charge)
    monkeypatch.setattr(engine, "predict", lambda balance, cap: (150.0, 0.2))

    spent = bot.reinvest()
    assert spent == 100.0  # capped at 50% of 200
    assert calls == {"bal": bot.bot_id, "charge": bot.bot_id}
    rows = db.fetch()
    assert rows and rows[0][0] == 100.0


def test_reinvest_error_propagates(monkeypatch, tmp_path):
    ie = _import_investment_engine(monkeypatch, tmp_path)
    db = ie.InvestmentDB(tmp_path / "i.db")
    engine = ie.PredictiveSpendEngine(db)
    bot = ie.AutoReinvestmentBot(predictor=engine, db=db)
    calls = {}

    def fake_balance(bot_id, *a, **k):
        calls["bal"] = bot_id
        return 200.0

    def bad_charge(bot_id, amount, description=None, *, overrides=None):
        calls["charge"] = bot_id
        raise RuntimeError("nope")

    monkeypatch.setattr(ie.stripe_billing_router, "get_balance", fake_balance)
    monkeypatch.setattr(ie.stripe_billing_router, "charge", bad_charge)
    monkeypatch.setattr(engine, "predict", lambda balance, cap: (50.0, 0.2))

    spent = bot.reinvest()
    assert spent == 0.0
    assert calls == {"bal": bot.bot_id, "charge": bot.bot_id}
    assert db.fetch() == []


def test_reinvest_balance_error(monkeypatch, tmp_path):
    ie = _import_investment_engine(monkeypatch, tmp_path)
    db = ie.InvestmentDB(tmp_path / "i.db")
    engine = ie.PredictiveSpendEngine(db)
    bot = ie.AutoReinvestmentBot(predictor=engine, db=db)

    def bad_balance(bot_id, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ie.stripe_billing_router, "get_balance", bad_balance)
    with pytest.raises(RuntimeError):
        bot.reinvest()


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
    del sbr.ROUTING_TABLE[("stripe", "default", "finance", "finance_router_bot")][
        "price_id"
    ]
    del sbr.ROUTING_TABLE[("stripe", "default", "finance", "finance_router_bot")][
        "customer_id"
    ]
    sbr.charge("stripe:finance:finance_router_bot", amount=3.0)
    params = stripe_stub.PaymentIntent.last_params
    assert params["api_key"] == "sk_live_dummy"
    assert params["amount"] == 300
    assert params["description"] == "prod_finance_router"
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
