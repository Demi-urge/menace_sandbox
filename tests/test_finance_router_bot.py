import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import os
import json
import menace.finance_router_bot as frb


def test_route_and_summary(tmp_path, monkeypatch):
    log = tmp_path / "payout.json"
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test")
    monkeypatch.setattr(frb.stripe_handler, "charge", lambda *a, **k: "success")
    bot = frb.FinanceRouterBot(payout_log_path=log, test_mode=True)
    res = bot.route_payment(10.0, "model1")
    assert res
    data = json.loads(log.read_text())
    assert data and data[0]["model_id"] == "model1"
    summary = bot.report_earnings_summary()
    assert summary["model1"] == 10.0
