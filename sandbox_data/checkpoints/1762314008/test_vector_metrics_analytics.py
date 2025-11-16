import menace.vector_metrics_db as vdb
import menace.vector_metrics_analytics as vma
from datetime import datetime, timedelta


def test_stale_embedding_cost(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    now = datetime.utcnow()
    old_ts = (now - timedelta(days=10)).isoformat()
    recent_ts = (now - timedelta(days=2)).isoformat()
    db.add(vdb.VectorMetric(event_type="embedding", db="db1", tokens=10, wall_time_ms=0.0, ts=old_ts))
    db.add(vdb.VectorMetric(event_type="embedding", db="db1", tokens=5, wall_time_ms=0.0, ts=recent_ts))
    cost = vma.stale_embedding_cost(db, timedelta(days=5), now=now)
    assert cost == 5 * 24 * 3600 * 10


def test_roi_by_database(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.log_retrieval("db1", tokens=0, wall_time_ms=0.0, hit=True, rank=1, contribution=1.0, prompt_tokens=0)
    db.log_retrieval("db2", tokens=0, wall_time_ms=0.0, hit=False, rank=1, contribution=0.5, prompt_tokens=0)
    db.log_retrieval("db1", tokens=0, wall_time_ms=0.0, hit=True, rank=2, contribution=0.2, prompt_tokens=0)
    roi = vma.roi_by_database(db)
    assert roi == {"db1": 1.2, "db2": 0.5}


def test_retrieval_training_samples(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.log_retrieval("db1", tokens=0, wall_time_ms=0.0, hit=True, rank=1, contribution=0.3, prompt_tokens=0, similarity=0.9, context_score=0.2, age=1.0)
    db.log_retrieval("db1", tokens=0, wall_time_ms=0.0, hit=False, rank=2, contribution=0.1, prompt_tokens=0, similarity=0.8, context_score=0.1, age=2.0)
    samples = vma.retrieval_training_samples(db, limit=1)
    assert len(samples) == 1
    sample = samples[0]
    assert set(sample) == {"db", "rank", "hit", "contribution", "win", "regret"}
