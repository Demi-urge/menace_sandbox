import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from menace.ensemble_predictor import EnsemblePredictor
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from menace.data_bot import DataBot, MetricsDB, MetricRecord
from menace.capital_management_bot import CapitalManagementBot


def test_ensemble_predict(tmp_path, monkeypatch):
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

    ens = EnsemblePredictor(2, history_db=hist, data_bot=data_bot, capital_bot=cap_bot)

    monkeypatch.setattr(data_bot, "engagement_delta", lambda limit=50: 0.0)
    monkeypatch.setattr(data_bot, "long_term_roi_trend", lambda limit=200: 0.0)
    monkeypatch.setattr(data_bot, "worst_bot", lambda metric="errors", limit=200: "a")
    monkeypatch.setattr(cap_bot, "profit_trend", lambda: 0.0)
    monkeypatch.setattr(DataBot, "complexity_score", staticmethod(lambda df: 1.0))
    monkeypatch.setattr(cap_bot, "bot_roi", lambda a: 1.0)

    ens.train()
    mean, var = ens.predict("a", 1.0)
    assert isinstance(mean, float) and isinstance(var, float)
    assert var >= 0.0
