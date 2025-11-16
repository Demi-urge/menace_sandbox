import numpy as np
import json

from menace.adaptive_roi_dataset import build_dataset, load_adaptive_roi_dataset
from menace.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
from menace.evolution_history_db import EvolutionEvent, EvolutionHistoryDB
from menace.roi_tracker import ROITracker
import db_router
import types
import sys


def test_dataset_aggregation(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    router = db_router.DBRouter(
        "eval", str(tmp_path / "db.sqlite"), str(tmp_path / "db.sqlite")
    )
    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(router=router)

    # create two evolution events for one engine
    evo.add(EvolutionEvent(action="engine", before_metric=0.2, after_metric=0.5, roi=0.3))
    evo.add(EvolutionEvent(action="engine", before_metric=0.5, after_metric=0.7, roi=0.4))

    # one evaluation record tied to the engine
    eva.add(EvaluationRecord(engine="engine", cv_score=0.8, passed=True))

    X, y, passed, g = load_adaptive_roi_dataset(evo_db_path, router=router)

    assert X.shape == (1, 2)
    assert y.shape == (1,)
    assert passed.tolist() == [1]
    assert g.tolist() == ["linear"]
    # features should be normalised (mean approximately 0)
    assert np.allclose(X.mean(axis=0), 0.0)
    assert np.allclose(y.mean(), 0.0)


def test_build_dataset(tmp_path, monkeypatch):
    evo_db_path = tmp_path / "evo.db"
    router = db_router.DBRouter(
        "eval2", str(tmp_path / "db2.sqlite"), str(tmp_path / "db2.sqlite")
    )

    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(router=router)

    class DummyEmb:
        def __init__(self, *args, **kwargs):
            pass

        def embed_query(self, text):
            return [1.0, 2.0]

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(OpenAIEmbeddings=DummyEmb),
    )

    eva.conn.execute("ALTER TABLE evaluation_history ADD COLUMN gpt_feedback TEXT")

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

    eva.conn.execute(
        "INSERT INTO evaluation_history(engine, cv_score, passed, error, ts, gpt_feedback) VALUES (?,?,?,?,?,?)",
        ("engine", 0.8, 1, "", "2024-01-01T00:02:00", "good"),
    )
    eva.conn.commit()

    conn = router.get_connection("action_roi")
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
    for i, rev in enumerate([15.0, 20.0, 25.0, 30.0, 35.0], start=1):
        api = 2.0 + i
        conn.execute(
            "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
            (
                "engine",
                rev,
                api,
                float(i + 1),
                0.5 + 0.1 * i,
                f"2024-01-01T00:0{i}:00",
            ),
        )
    conn.commit()

    X, y, g, names = build_dataset(
        evo_db_path, router=router, return_feature_names=True
    )

    assert X.shape == (1, len(names))
    assert any(name.startswith("gpt_feedback_emb_") for name in names)
    assert y.tolist() == [[12.0, 20.0, 28.0, 10.0]]
    assert g.tolist() == ["linear"]
    assert np.allclose(X, 0.0)


def test_build_dataset_horizon(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    router = db_router.DBRouter(
        "horizon", str(tmp_path / "horizon.sqlite"), str(tmp_path / "horizon.sqlite")
    )

    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(router=router)

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

    conn = router.get_connection("action_roi")
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
        ("engine", 20.0, 4.0, 3.0, 0.7, "2024-01-01T00:02:00"),
    )
    conn.commit()

    X, y, g = build_dataset(evo_db_path, router=router, horizons=[1, 2])
    assert y.shape == (1, 3)
    assert y.tolist() == [[12.0, 16.0, 10.0]]


def test_build_dataset_selected_features(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    router = db_router.DBRouter(
        "sel", str(tmp_path / "sel.sqlite"), str(tmp_path / "sel.sqlite")
    )

    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(router=router)

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

    conn = router.get_connection("action_roi")
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
        router=router,
        horizons=[1],
        selected_features=["before_metric", "after_metric"],
        return_feature_names=True,
    )
    assert X.shape[1] == 2
    assert names == ["before_metric", "after_metric"]


def test_build_dataset_auto_selected_features(tmp_path, monkeypatch):
    evo_db_path = tmp_path / "evo.db"
    router = db_router.DBRouter(
        "auto", str(tmp_path / "auto.sqlite"), str(tmp_path / "auto.sqlite")
    )

    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(router=router)

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

    conn = router.get_connection("action_roi")
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

    meta_path = tmp_path / "sandbox_data" / "adaptive_roi.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"selected_features": ["before_metric", "after_metric"]}))
    monkeypatch.chdir(tmp_path)

    X, y, g, names = build_dataset(
        evo_db_path,
        router=router,
        horizons=[1],
        return_feature_names=True,
    )
    assert names == ["before_metric", "after_metric"]
