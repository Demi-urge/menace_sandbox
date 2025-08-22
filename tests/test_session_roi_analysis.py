import vector_metrics_db as vdb
from analytics.session_roi import per_origin_stats
from vector_metrics_aggregator import VectorMetricsAggregator


def test_per_origin_stats(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.log_retrieval("code", tokens=0, wall_time_ms=0.0, hit=True, rank=1, session_id="s1", vector_id="v1")
    db.log_retrieval("docs", tokens=0, wall_time_ms=0.0, hit=True, rank=1, session_id="s2", vector_id="v2")
    db.update_outcome("s1", [("code", "v1")], contribution=1.0, win=True)
    db.update_outcome("s2", [("docs", "v2")], contribution=-0.5, win=False)

    stats = per_origin_stats(db)
    assert stats["code"]["success_rate"] == 1.0
    assert stats["code"]["roi_delta"] == 1.0
    assert stats["docs"]["success_rate"] == 0.0
    assert stats["docs"]["roi_delta"] == -0.5

    agg = VectorMetricsAggregator(tmp_path / "v.db")
    assert agg.origin_stats() == stats
