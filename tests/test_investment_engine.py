import importlib.util
import sys
import types
from pathlib import Path

# Tests previously mocked a deprecated ``stripe_handler``.  These tests ensure
# ``stripe_billing_router`` is patched instead.

ROOT = Path(__file__).resolve().parents[1]


def _import_investment_engine(monkeypatch):
    pkg = types.ModuleType("iepkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["iepkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"iepkg.{name}", ROOT / f"{name}.py"
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
        }.get(name, ""),
    )
    sbr = _load("stripe_billing_router")
    sys.modules["iepkg.stripe_billing_router"] = sbr
    sys.modules["stripe_billing_router"] = sbr
    dbm = _load("db_router")
    dbm.GLOBAL_ROUTER = None
    sys.modules["db_router"] = dbm
    return _load("investment_engine")


def test_reinvest_cap(monkeypatch, tmp_path):
    ie = _import_investment_engine(monkeypatch)
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
    ie = _import_investment_engine(monkeypatch)
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
