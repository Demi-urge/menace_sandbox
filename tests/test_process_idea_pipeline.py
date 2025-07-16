import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.database_manager as dm
import menace.preliminary_research_bot as prb


def test_process_and_threshold(monkeypatch, tmp_path):
    db = tmp_path / "m.db"
    prelim = prb.PreliminaryResearchBot()
    # fake processing returning simple BusinessData
    monkeypatch.setattr(
        prelim,
        "process_model",
        lambda name, urls: prb.BusinessData(model_name=name)
    )
    # profitability score fixed at 69
    monkeypatch.setattr(dm, "compute_profitability_score", lambda data: 69.0)

    status = dm.process_idea(
        "Idea1",
        tags=["x"],
        source="idea_bot",
        urls=[],
        prelim=prelim,
        energy_score=0.0,
        db_path=db,
    )
    assert status == "invalid"
    row = dm.search_models("Idea1", db_path=db)[0]
    assert row["exploration_status"] == "invalid"

    # lowering threshold should flip to greenlit
    dm.update_profitability_threshold(0.5, db_path=db)
    row = dm.search_models("Idea1", db_path=db)[0]
    assert row["exploration_status"] == "pending"

    # duplicate ideas are killed
    status2 = dm.process_idea(
        "Idea1",
        tags=["x"],
        source="idea_bot",
        urls=[],
        prelim=prelim,
        energy_score=0.5,
        db_path=db,
    )
    assert status2 == "killed"


def test_submit_idea(monkeypatch, tmp_path):
    db = tmp_path / "m.db"
    monkeypatch.setattr(
        dm.PreliminaryResearchBot,
        "process_model",
        lambda self, name, urls: prb.BusinessData(model_name=name),
    )
    monkeypatch.setattr(dm, "compute_profitability_score", lambda data: 42.0)
    status = dm.submit_idea("ModelZ", ["tag"], "idea_bot", db_path=db)
    assert status in {"pending", "invalid"}
    rows = dm.search_models("ModelZ", db_path=db)
    assert rows
