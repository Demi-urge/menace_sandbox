import json
import sqlite3

import pandas as pd
import menace.retrieval_ranker as rr


def _make_dbs(tmp_path):
    vdb = tmp_path / "vec.db"
    pdb = tmp_path / "patch.db"

    conn = sqlite3.connect(vdb)
    conn.execute(
        "CREATE TABLE vector_metrics (session_id TEXT, vector_id TEXT, db TEXT, age REAL, similarity REAL, contribution REAL, hit INTEGER, ts REAL, event_type TEXT)"
    )
    conn.execute(
        "INSERT INTO vector_metrics VALUES ('s1','v1','db1',0,0,0,1,0,'retrieval')"
    )
    conn.execute(
        "INSERT INTO vector_metrics VALUES ('s2','v2','db1',0,0,0,0,1,'retrieval')"
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(pdb)
    conn.execute(
        "CREATE TABLE patch_outcomes (session_id TEXT, vector_id TEXT, success INTEGER, reverted INTEGER)"
    )
    conn.execute("INSERT INTO patch_outcomes VALUES ('s1','v1',1,0)")
    conn.execute("INSERT INTO patch_outcomes VALUES ('s2','v2',0,0)")
    conn.commit()
    conn.close()
    return vdb, pdb


def test_training_and_serialisation(tmp_path):
    df = pd.DataFrame(
        {
            "session_id": ["a", "b", "c", "d"],
            "vector_id": [1, 2, 3, 4],
            "db_type": ["bot", "bot", "error", "error"],
            "age": [0.1, 0.2, 0.3, 0.4],
            "similarity": [0.9, 0.8, 0.7, 0.6],
            "exec_freq": [0, 1, 2, 3],
            "roi_delta": [0.1, 0.2, 0.3, 0.4],
            "prior_hits": [1, 0, 1, 0],
            "win_rate": [0.5, 0.4, 0.3, 0.2],
            "regret_rate": [0.1, 0.2, 0.3, 0.4],
            "stale_cost": [1.0, 2.0, 3.0, 4.0],
            "sample_count": [10.0, 20.0, 30.0, 40.0],
            "roi": [0.5, 0.4, 0.3, 0.2],
            "label": [1, 0, 1, 0],
        }
    )
    tm, metrics = rr.train(df)
    out = tmp_path / "model.json"
    rr.save_model(tm, out)
    data = rr.load_model(out)
    assert data["features"] == tm.feature_names
    assert "coef" in data or "booster" in data
    assert {
        "age",
        "similarity",
        "exec_freq",
        "roi_delta",
        "prior_hits",
        "win_rate",
        "regret_rate",
        "stale_cost",
        "sample_count",
        "roi",
    }.issubset(set(tm.feature_names))
    assert {"db_bot", "db_error"}.issubset(set(tm.feature_names))
    assert metrics and ("accuracy" in metrics or "auc" in metrics)


def test_cli_training(tmp_path):
    vdb, pdb = _make_dbs(tmp_path)
    out = tmp_path / "model.json"
    rr.main(
        [
            "train",
            "--vector-db",
            str(vdb),
            "--patch-db",
            str(pdb),
            "--model-path",
            str(out),
        ]
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["features"]

