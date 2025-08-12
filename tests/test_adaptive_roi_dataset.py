import numpy as np

import sqlite3

from menace.adaptive_roi_dataset import build_dataset, load_adaptive_roi_dataset
from menace.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
from menace.evolution_history_db import EvolutionEvent, EvolutionHistoryDB
from menace.roi_tracker import ROITracker


def test_dataset_aggregation(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    eval_db_path = tmp_path / "eval.db"
    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(eval_db_path)

    # create two evolution events for one engine
    evo.add(EvolutionEvent(action="engine", before_metric=0.2, after_metric=0.5, roi=0.3))
    evo.add(EvolutionEvent(action="engine", before_metric=0.5, after_metric=0.7, roi=0.4))

    # one evaluation record tied to the engine
    eva.add(EvaluationRecord(engine="engine", cv_score=0.8, passed=True))

    X, y, passed, g = load_adaptive_roi_dataset(evo_db_path, eval_db_path)

    assert X.shape == (1, 2)
    assert y.shape == (1,)
    assert passed.tolist() == [1]
    assert g.tolist() == ["linear"]
    # features should be normalised (mean approximately 0)
    assert np.allclose(X.mean(axis=0), 0.0)
    assert np.allclose(y.mean(), 0.0)


def test_build_dataset(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    eval_db_path = tmp_path / "eval.db"
    roi_db_path = tmp_path / "roi.db"

    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(eval_db_path)

    ts0 = "2024-01-01T00:00:00"
    # evolution event
    evo.add(
        EvolutionEvent(
            action="engine",
            before_metric=0.2,
            after_metric=0.3,
            roi=0.0,
            ts=ts0,
        )
    )

    # evaluation score after event
    eva.add(
        EvaluationRecord(
            engine="engine", cv_score=0.8, passed=True, ts="2024-01-01T00:02:00"
        )
    )

    # create roi.db with one record before and one after event
    conn = sqlite3.connect(roi_db_path)
    conn.execute(
        """
        CREATE TABLE action_roi(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            revenue REAL,
            api_cost REAL,
            cpu_seconds REAL,
            success_rate REAL,
            ts TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
        ("engine", 10.0, 2.0, 1.0, 0.5, "2023-12-31T23:59:00"),
    )
    conn.execute(
        "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
        ("engine", 15.0, 3.0, 2.0, 0.6, "2024-01-01T00:01:00"),
    )
    conn.commit()

    X, y, g = build_dataset(evo_db_path, roi_db_path, eval_db_path)

    tracker = ROITracker()
    base_metrics = set(tracker.metrics_history) | set(tracker.synergy_metrics_history)
    n_features = 6 + len(base_metrics) + 5
    assert X.shape == (1, n_features)
    assert y.tolist() == [12.0]
    assert g.tolist() == ["linear"]
    assert np.allclose(X, 0.0)


def test_build_dataset_horizon(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    eval_db_path = tmp_path / "eval.db"
    roi_db_path = tmp_path / "roi.db"

    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(eval_db_path)

    ts0 = "2024-01-01T00:00:00"
    evo.add(
        EvolutionEvent(
            action="engine",
            before_metric=0.2,
            after_metric=0.3,
            roi=0.0,
            ts=ts0,
        )
    )
    eva.add(
        EvaluationRecord(
            engine="engine", cv_score=0.8, passed=True, ts="2024-01-01T00:02:00"
        )
    )

    conn = sqlite3.connect(roi_db_path)
    conn.execute(
        """
        CREATE TABLE action_roi(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            revenue REAL,
            api_cost REAL,
            cpu_seconds REAL,
            success_rate REAL,
            ts TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
        ("engine", 10.0, 2.0, 1.0, 0.5, "2023-12-31T23:59:00"),
    )
    conn.execute(
        "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
        ("engine", 15.0, 3.0, 2.0, 0.6, "2024-01-01T00:01:00"),
    )
    conn.execute(
        "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
        ("engine", 18.0, 4.0, 3.0, 0.7, "2024-01-01T00:02:00"),
    )
    conn.commit()

    X, y, g = build_dataset(
        evo_db_path, roi_db_path, eval_db_path, horizon=2
    )
    assert y.shape == (1, 2)
    assert y.tolist() == [[12.0, 14.0]]


def test_build_dataset_selected_features(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    eval_db_path = tmp_path / "eval.db"
    roi_db_path = tmp_path / "roi.db"

    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(eval_db_path)

    ts0 = "2024-01-01T00:00:00"
    evo.add(
        EvolutionEvent(
            action="engine",
            before_metric=0.2,
            after_metric=0.3,
            roi=0.0,
            ts=ts0,
        )
    )
    eva.add(
        EvaluationRecord(
            engine="engine", cv_score=0.8, passed=True, ts="2024-01-01T00:02:00"
        )
    )

    conn = sqlite3.connect(roi_db_path)
    conn.execute(
        """
        CREATE TABLE action_roi(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            revenue REAL,
            api_cost REAL,
            cpu_seconds REAL,
            success_rate REAL,
            ts TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
        ("engine", 10.0, 2.0, 1.0, 0.5, "2023-12-31T23:59:00"),
    )
    conn.execute(
        "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
        ("engine", 15.0, 3.0, 2.0, 0.6, "2024-01-01T00:01:00"),
    )
    conn.commit()

    X, y, g, names = build_dataset(
        evo_db_path,
        roi_db_path,
        eval_db_path,
        selected_features=["before_metric", "after_metric"],
        return_feature_names=True,
    )
    assert X.shape[1] == 2
    assert names == ["before_metric", "after_metric"]
