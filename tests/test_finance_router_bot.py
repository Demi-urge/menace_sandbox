import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json  # noqa: E402
import menace.finance_router_bot as frb  # noqa: E402


def test_route_and_summary(tmp_path, monkeypatch):
    log = tmp_path / "payout.json"
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setenv("STRIPE_PUBLIC_KEY", "pk_test")
    monkeypatch.setattr(
        frb.stripe_billing_router, "charge", lambda *a, **k: {"status": "succeeded"}
    )
    bot = frb.FinanceRouterBot(payout_log_path=log)
    res = bot.route_payment(10.0, "model1")
    assert res
    data = json.loads(log.read_text())
    assert data and data[0]["model_id"] == "model1"
    summary = bot.report_earnings_summary()
    assert summary["model1"] == 10.0
