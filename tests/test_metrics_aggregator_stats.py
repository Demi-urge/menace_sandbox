import os
import menace.data_bot as db
import menace.metrics_aggregator as ma


def test_compute_retriever_stats(tmp_path, monkeypatch):
    mdb_path = tmp_path / "m.db"
    metrics = db.MetricsDB(mdb_path)
    # log patch outcomes
    metrics.log_patch_outcome("p1", True, [("db1", "v1")], session_id="s1")
    metrics.log_patch_outcome("p2", False, [("db1", "v2")], session_id="s2")
    metrics.log_patch_outcome("p3", False, [("db2", "v3")], session_id="s3", reverted=True)
    # log staleness data
    metrics.log_embedding_staleness("db1", "v1", 90000)
    metrics.log_embedding_staleness("db2", "v2", 1000)
    # ensure roi db path doesn't exist
    roi_path = tmp_path / "roi.db"
    res = ma.compute_retriever_stats(mdb_path, roi_path)
    assert abs(res["db1"]["win_rate"] - 0.5) < 1e-6
    assert abs(res["db1"]["regret_rate"] - 0.5) < 1e-6
    assert res["db1"]["stale_cost"] == 90000 - float(os.getenv("EMBEDDING_STALE_THRESHOLD_SECONDS", "86400"))
    assert res["db1"]["sample_count"] == 2
    assert res["db2"]["win_rate"] == 0.0
    assert res["db2"]["regret_rate"] == 1.0
    assert res["db2"]["stale_cost"] == 0.0
    assert res["db2"]["sample_count"] == 1
    latest = metrics.latest_retriever_kpi()
    assert "db1" in latest and abs(latest["db1"]["win_rate"] - 0.5) < 1e-6
    assert latest["db1"]["sample_count"] == 2
