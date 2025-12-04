import db_router
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


def test_retriever_rates_by_db(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.add(
        vdb.VectorMetric(
            event_type="retrieval",
            db="db1",
            tokens=0,
            wall_time_ms=0.0,
            hit=True,
            rank=1,
            win=True,
            regret=False,
        )
    )
    db.add(
        vdb.VectorMetric(
            event_type="retrieval",
            db="db2",
            tokens=0,
            wall_time_ms=0.0,
            hit=False,
            rank=1,
            win=False,
            regret=True,
        )
    )
    assert db.retriever_win_rate_by_db() == {"db1": 1.0, "db2": 0.0}
    assert db.retriever_regret_rate_by_db() == {"db1": 0.0, "db2": 1.0}


def test_vector_metrics_bootstrap_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(db_router, "_audit_bootstrap_safe_default", False)
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", None)

    vm = vdb.VectorMetricsDB(tmp_path / "vm.db", bootstrap_safe=True)
    vm.conn.execute("CREATE TABLE IF NOT EXISTS demo(x INTEGER)")

    assert vm.router.local_conn.audit_bootstrap_safe is True
    assert vm.router.shared_conn.audit_bootstrap_safe is True


def test_warmup_does_not_create_db_file(tmp_path):
    db_path = tmp_path / "warm.db"
    vm = vdb.VectorMetricsDB(db_path, warmup=True)

    assert not db_path.exists()
    assert vm.planned_path() == db_path
    assert not db_path.exists()


def test_bootstrap_fast_does_not_create_db_file(tmp_path):
    db_path = tmp_path / "fast.db"
    vm = vdb.VectorMetricsDB(db_path, bootstrap_fast=True)

    assert not db_path.exists()
    assert vm.planned_path() == db_path
    assert not db_path.exists()


def test_bootstrap_getter_returns_stub_without_creating_db(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VECTOR_METRICS_BOOTSTRAP_WARMUP", "1")
    monkeypatch.setattr(vdb, "_VECTOR_DB_INSTANCE", None)

    vm = vdb.get_vector_metrics_db(warmup=True)

    assert isinstance(vm, vdb._BootstrapVectorMetricsStub)
    assert not (tmp_path / "vector_metrics.db").exists()


def test_bootstrap_timer_env_returns_stub(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "5")
    monkeypatch.setattr(vdb, "_VECTOR_DB_INSTANCE", None)

    vm = vdb.get_vector_metrics_db()

    assert isinstance(vm, vdb._BootstrapVectorMetricsStub)
    assert not (tmp_path / "vector_metrics.db").exists()


def test_bootstrap_timer_override_allows_activation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "5")
    monkeypatch.setattr(vdb, "_VECTOR_DB_INSTANCE", None)

    vm = vdb.get_vector_metrics_db(
        allow_bootstrap_activation=True, ensure_exists=True, read_only=False
    )

    assert isinstance(vm, vdb.VectorMetricsDB)
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)
    assert (tmp_path / "vector_metrics.db").exists()
