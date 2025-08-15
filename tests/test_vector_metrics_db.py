import menace.vector_metrics_db as vdb


def test_embedding_tokens_total(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.log_embedding("db1", tokens=10, wall_time_ms=5, store_time_ms=1)
    db.log_embedding("db1", tokens=5, wall_time_ms=3)
    assert db.embedding_tokens_total() == 15
    assert db.embedding_tokens_total("db1") == 15


def test_retrieval_hit_rate(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.log_retrieval(
        "db1",
        tokens=10,
        wall_time_ms=1.0,
        hit=True,
        rank=1,
        contribution=0.5,
        prompt_tokens=5,
        similarity=0.9,
        context_score=0.1,
        age=5.0,
    )
    db.log_retrieval(
        "db1",
        tokens=5,
        wall_time_ms=1.0,
        hit=False,
        rank=2,
        contribution=0.1,
        prompt_tokens=2,
        similarity=0.8,
        context_score=0.2,
        age=3.0,
    )
    assert db.retrieval_hit_rate() == 0.5
    assert db.retrieval_hit_rate("db1") == 0.5
