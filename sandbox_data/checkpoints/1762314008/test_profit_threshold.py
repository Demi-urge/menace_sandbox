import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.database_manager as dm
import menace.preliminary_research_bot as prb


def test_threshold_update(monkeypatch, tmp_path):
    db = tmp_path / "m.db"
    bot = prb.PreliminaryResearchBot()

    def fake_process(name, urls):
        return prb.BusinessData(
            model_name=name,
            profit_margin=50.0,
            operational_cost=10.0,
            market_saturation=1.0,
            keywords=["x"],
            roi_score=4.0,
        )

    monkeypatch.setattr(bot, "process_model", fake_process)

    status = dm.evaluate_candidate("ModelX", [], bot, threshold=70.0, db_path=db)
    assert status == "invalid"

    dm.apply_threshold(35.0, db_path=db)
    rows = dm.search_models("ModelX", db_path=db)
    assert rows and rows[0]["exploration_status"] == "pending"
    assert abs(rows[0]["current_roi"] - 4.0) < 1e-6
    assert abs(rows[0]["initial_roi_prediction"] - 4.0) < 1e-6
    assert abs(rows[0]["final_roi_prediction"] - 4.0) < 1e-6

    thr = dm.calculate_threshold(0.5)
    assert 60.0 < thr < 70.0

