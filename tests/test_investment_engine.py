import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.investment_engine as ie  # noqa: E402


def test_reinvest_cap(monkeypatch, tmp_path):
    db = ie.InvestmentDB(tmp_path / "i.db")
    engine = ie.PredictiveSpendEngine(db)
    bot = ie.AutoReinvestmentBot(
        cap_percentage=0.5,
        safety_reserve=0.0,
        minimum_threshold=10.0,
        predictor=engine,
        db=db,
    )
    monkeypatch.setattr(
        ie.stripe_billing_router, "get_balance", lambda *a, **k: 200.0
    )
    monkeypatch.setattr(
        ie.stripe_billing_router, "charge", lambda *a, **k: {"status": "succeeded"}
    )
    monkeypatch.setattr(engine, "predict", lambda balance, cap: (150.0, 0.2))

    spent = bot.reinvest()
    assert spent == 100.0  # capped at 50% of 200
    rows = db.fetch()
    assert rows and rows[0][0] == 100.0
