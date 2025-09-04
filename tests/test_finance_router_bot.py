import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _import_finance_router(monkeypatch):
    pkg = types.ModuleType("frbpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["frbpkg"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"frbpkg.{name}", ROOT / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"frbpkg.{name}"] = module
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
        }.get(name, ""),
    )
    sbr = _load("stripe_billing_router")
    sys.modules["frbpkg.stripe_billing_router"] = sbr

    return _load("finance_router_bot")


def test_route_and_summary(tmp_path, monkeypatch):
    frb = _import_finance_router(monkeypatch)
    log = tmp_path / "payout.json"
    calls = {}

    def fake_charge(bot_id, amount, description=None, *, overrides=None):
        calls["bot_id"] = bot_id
        calls["amount"] = amount
        return {"status": "succeeded"}

    monkeypatch.setattr(frb.stripe_billing_router, "init_charge", fake_charge)
    bot = frb.FinanceRouterBot(payout_log_path=log)
    res = bot.route_payment(10.0, "model1")
    assert res
    assert calls == {"bot_id": "model1", "amount": 10.0}
    data = json.loads(log.read_text())
    assert data and data[0]["model_id"] == "model1"
    summary = bot.report_earnings_summary()
    assert summary["model1"] == 10.0


def test_router_error_propagates(tmp_path, monkeypatch):
    frb = _import_finance_router(monkeypatch)
    log = tmp_path / "payout.json"
    calls = {}

    def bad_charge(bot_id, amount, description=None, *, overrides=None):
        calls["bot_id"] = bot_id
        raise RuntimeError("boom")

    monkeypatch.setattr(frb.stripe_billing_router, "init_charge", bad_charge)
    bot = frb.FinanceRouterBot(payout_log_path=log)
    res = bot.route_payment(5.0, "model2")
    assert res.startswith("error:")
    assert calls["bot_id"] == "model2"
    data = json.loads(log.read_text())
    assert data and data[0]["result"].startswith("error:")
