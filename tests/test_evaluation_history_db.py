import importlib
import sqlite3

import db_router
import pytest

from menace.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord


def test_weights_and_history(tmp_path):
    router = db_router.DBRouter(
        "hist", str(tmp_path / "hist.db"), str(tmp_path / "hist.db")
    )
    db = EvaluationHistoryDB(router=router)
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


def test_evaluation_history_db_routing(tmp_path):
    local = tmp_path / "local.db"
    shared = tmp_path / "shared.db"
    router = db_router.init_db_router(
        "eval_hist", local_db_path=str(local), shared_db_path=str(shared)
    )
    db = EvaluationHistoryDB(router=router)
    db.add(EvaluationRecord(engine="x", cv_score=0.5))
    conn = router.get_connection("evaluation_history")
    row = conn.execute("SELECT engine FROM evaluation_history").fetchone()
    assert row == ("x",)
    shared_rows = router.shared_conn.execute(
        "SELECT name FROM sqlite_master WHERE name='evaluation_history'"
    ).fetchall()
    assert shared_rows == []


def test_evaluation_history_no_direct_sqlite(tmp_path, monkeypatch):
    local = tmp_path / "local2.db"
    shared = tmp_path / "shared2.db"
    db_router.init_db_router(
        "eval_hist2", local_db_path=str(local), shared_db_path=str(shared)
    )
    importlib.reload(importlib.import_module("menace.evaluation_history_db"))

    calls: list[object] = []

    def bad_connect(*a, **k):  # pragma: no cover - should never run
        calls.append(1)
        raise AssertionError("sqlite3.connect should not be called directly")

    monkeypatch.setattr(sqlite3, "connect", bad_connect)

    db = EvaluationHistoryDB()
    db.add(EvaluationRecord(engine="y", cv_score=0.7))

    assert not calls
