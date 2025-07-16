import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import menace.evolution_analysis_bot as eab
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from menace.data_bot import DataBot, MetricsDB, MetricRecord
from menace.capital_management_bot import CapitalManagementBot


def test_train_predict(tmp_path, monkeypatch):
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    hist.add(EvolutionEvent("a", 1.0, 2.0, 1.0))
    hist.add(EvolutionEvent("b", 2.0, 3.0, 2.0))

    mdb = MetricsDB(tmp_path / "m.db")
    mdb.add(
        MetricRecord(
            bot="menace",
            cpu=1.0,
            memory=1.0,
            response_time=0.1,
            disk_io=0.0,
            net_io=0.0,
            errors=0,
        )
    )

    data_bot = DataBot(mdb)
    cap_bot = CapitalManagementBot(data_bot=data_bot)
    bot = eab.EvolutionAnalysisBot(
        hist, data_bot=data_bot, capital_bot=cap_bot
    )

    monkeypatch.setattr(data_bot, "engagement_delta", lambda limit=50: 0.0)
    monkeypatch.setattr(data_bot, "long_term_roi_trend", lambda limit=200: 0.0)
    monkeypatch.setattr(data_bot, "worst_bot", lambda metric="errors", limit=200: "a")
    monkeypatch.setattr(cap_bot, "profit_trend", lambda: 0.0)
    monkeypatch.setattr(
        eab.DataBot, "complexity_score", staticmethod(lambda df: 1.0)
    )

    def roi_train(action):
        return 0.5 if action == "a" else 1.0

    monkeypatch.setattr(cap_bot, "bot_roi", roi_train)
    bot.train()

    monkeypatch.setattr(
        eab.DataBot, "complexity_score", staticmethod(lambda df: 2.0)
    )
    monkeypatch.setattr(cap_bot, "bot_roi", lambda a: 2.0)
    pred1 = bot.predict("a", 1.0)

    monkeypatch.setattr(
        eab.DataBot, "complexity_score", staticmethod(lambda df: 4.0)
    )
    monkeypatch.setattr(cap_bot, "bot_roi", lambda a: 4.0)
    pred2 = bot.predict("a", 1.0)

    assert isinstance(pred1, float) and isinstance(pred2, float)
    assert pred1 != pred2
