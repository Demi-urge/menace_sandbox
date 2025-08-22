import vector_metrics_db as vdb
from analytics.session_roi import origin_roi, per_origin_stats
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


def test_origin_roi(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.log_retrieval("bots", tokens=0, wall_time_ms=0.0, hit=True, rank=1, session_id="b", vector_id="vb")
    db.log_retrieval("errors", tokens=0, wall_time_ms=0.0, hit=True, rank=1, session_id="e", vector_id="ve")
    db.update_outcome("b", [("bots", "vb")], contribution=0.3, win=True)
    db.update_outcome("e", [("errors", "ve")], contribution=-0.2, win=False)

    stats = origin_roi(db)
    assert stats["bots"]["bots"]["roi_delta"] == 0.3
    assert stats["errors"]["errors"]["success_rate"] == 0.0
