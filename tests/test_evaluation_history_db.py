import pytest

from menace.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord


def test_weights_and_history(tmp_path):
    db = EvaluationHistoryDB(tmp_path / "hist.db")
    db.add(EvaluationRecord(engine="a", cv_score=0.1))
    db.add(EvaluationRecord(engine="a", cv_score=0.3))
    db.add(EvaluationRecord(engine="b", cv_score=0.5))
    db.add(EvaluationRecord(engine="b", cv_score=0.7))

    hist = db.history("a", limit=2)
    assert len(hist) == 2
    assert hist[0][2] == 1
    weights = db.deployment_weights()
    assert weights["b"] == 1.0
    expected_a = ((0.1 + 0.3) / 2) / ((0.5 + 0.7) / 2)
    assert abs(weights["a"] - expected_a) < 1e-6

    db.set_weight("a", 0.2)
    weights = db.deployment_weights()
    assert weights["a"] == 0.2
