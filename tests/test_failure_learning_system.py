import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.failure_learning_system as fls
import menace.bot_planning_bot as bpb
import menace.capital_management_bot as cmb


def _record():
    return fls.FailureRecord(
        model_id="m1",
        cause="low demand",
        features="f1,f2",
        demographics="d1",
        profitability=0.1,
        retention=0.2,
        cac=1.0,
        roi=-0.5,
    )


def test_log_and_score(tmp_path):
    db = fls.DiscrepancyDB(tmp_path / "f.db")
    sys = fls.FailureLearningSystem(db)
    rec = _record()
    sys.record_failure(rec)
    sys.record_failure(rec)
    risky = sys.advise_planner(bpb.BotPlanningBot())
    assert "f1" in risky or "f2" in risky
    score = sys.failure_score("m1")
    assert score == 1.0


def test_empty_failure_score(tmp_path):
    sys = fls.FailureLearningSystem(fls.DiscrepancyDB(tmp_path / "f.db"))
    score = sys.failure_score("x")
    assert score == 0.0

