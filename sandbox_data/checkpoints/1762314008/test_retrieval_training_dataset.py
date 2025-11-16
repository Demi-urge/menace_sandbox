import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from vector_metrics_db import VectorMetricsDB
from retrieval_training_dataset import build_dataset


def test_feature_extraction_and_labels(tmp_path):
    vector_db = tmp_path / "vector.db"
    patch_db = tmp_path / "patch.db"

    # Set up vector metrics with two retrieval events for the same vector
    vdb = VectorMetricsDB(vector_db)
    vdb.log_retrieval(
        db="demo",
        tokens=0,
        wall_time_ms=0,
        hit=True,
        rank=1,
        session_id="s1",
        vector_id="v1",
        similarity=0.9,
        age=10.0,
        contribution=0.2,
    )
    vdb.log_retrieval(
        db="other",
        tokens=0,
        wall_time_ms=0,
        hit=False,
        rank=1,
        session_id="s2",
        vector_id="v1",
        similarity=0.8,
        age=20.0,
        contribution=-0.1,
    )

    # Patch outcomes and retriever stats tables
    with sqlite3.connect(patch_db) as conn:
        conn.execute(
            "CREATE TABLE patch_outcomes(session_id TEXT, vector_id TEXT, success INTEGER, reverted INTEGER)"
        )
        conn.execute(
            "CREATE TABLE retriever_stats(origin_db TEXT, win_rate REAL, regret_rate REAL, stale_cost REAL, sample_count REAL, roi REAL)"
        )
        conn.execute(
            "INSERT INTO patch_outcomes(session_id, vector_id, success, reverted) VALUES(?,?,?,0)",
            ("s1", "v1", 1),
        )
        conn.execute(
            "INSERT INTO retriever_stats(origin_db, win_rate, regret_rate, stale_cost, sample_count, roi) VALUES(?,?,?,?,?,?)",
            ("demo", 0.7, 0.2, 5.0, 10.0, 1.2),
        )
        conn.commit()

    df = build_dataset(vector_db, patch_db)

    # There should be two rows corresponding to the two retrieval events
    assert len(df) == 2

    row1 = df[df["session_id"] == "s1"].iloc[0]
    assert row1.label == 1
    assert row1.prior_hits == 0
    assert row1.exec_freq == 0
    assert row1.win_rate == pytest.approx(0.7)
    assert row1.regret_rate == pytest.approx(0.2)
    assert row1.stale_cost == pytest.approx(5.0)
    assert row1.sample_count == pytest.approx(10.0)
    assert row1.roi == pytest.approx(1.2)

    row2 = df[df["session_id"] == "s2"].iloc[0]
    assert row2.label == 0
    assert row2.prior_hits == 1
    assert row2.exec_freq == 1
    # Rows without stats fall back to zeros
    assert row2.win_rate == pytest.approx(0.0)
    assert row2.regret_rate == pytest.approx(0.0)
    assert row2.stale_cost == pytest.approx(0.0)
    assert row2.sample_count == pytest.approx(0.0)
    assert row2.roi == pytest.approx(0.0)
