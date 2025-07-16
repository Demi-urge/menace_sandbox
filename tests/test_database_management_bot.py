import pytest

pytest.skip("optional dependencies not installed", allow_module_level=True)

from menace.database_management_bot import DatabaseManagementBot
from menace.preliminary_research_bot import PreliminaryResearchBot, BusinessData
import menace.database_manager as dm
from menace.capital_management_bot import CapitalManagementBot


def test_db_management_bot(monkeypatch, tmp_path):
    db = tmp_path / "models.db"
    prelim = PreliminaryResearchBot()
    capital = CapitalManagementBot()

    monkeypatch.setattr(
        prelim,
        "process_model",
        lambda name, urls: BusinessData(model_name=name)
    )
    monkeypatch.setattr(dm, "compute_profitability_score", lambda data: 69.0)

    bot = DatabaseManagementBot(prelim_bot=prelim, capital_bot=capital, db_path=db)

    monkeypatch.setattr(capital, "energy_score", lambda *a, **k: 0.0)
    status = bot.ingest_idea("IdeaBot", tags=["x"], source="idea_bot")
    assert status == "invalid"

    monkeypatch.setattr(capital, "energy_score", lambda *a, **k: 0.5)
    thr = bot.adjust_threshold()
    assert thr < 70.0
    row = dm.search_models("IdeaBot", db_path=db)[0]
    assert row["exploration_status"] == "pending"

    status2 = bot.ingest_idea("IdeaBot", tags=["x"], source="idea_bot")
    assert status2 == "killed"

