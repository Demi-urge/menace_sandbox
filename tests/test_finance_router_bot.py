import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json  # noqa: E402
import menace.finance_router_bot as frb  # noqa: E402


def test_route_and_summary(tmp_path, monkeypatch):
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
