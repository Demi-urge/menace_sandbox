import vector_metrics_db


def test_warmup_and_fast_bootstrap_skip_connection(monkeypatch, tmp_path):
    calls = []

    def fail_prepare(self, init_start=None):  # pragma: no cover - guard
        calls.append("prepare")
        raise AssertionError("_prepare_connection should not run during warmup")

    monkeypatch.setattr(vector_metrics_db.VectorMetricsDB, "_prepare_connection", fail_prepare)

    warm = vector_metrics_db.VectorMetricsDB(tmp_path / "warm.db", warmup=True)
    fast = vector_metrics_db.VectorMetricsDB(tmp_path / "fast.db", bootstrap_fast=True)

    assert warm._conn is None
    assert fast._conn is None
    assert warm._lazy_mode is True
    assert fast._lazy_mode is True
    assert calls == []


def test_normal_operation_exits_lazy_mode(monkeypatch, tmp_path):
    orig_prepare = vector_metrics_db.VectorMetricsDB._prepare_connection
    calls = []

    def record_prepare(self, init_start=None):
        calls.append("prepare")
        return orig_prepare(self, init_start)

    monkeypatch.setattr(vector_metrics_db.VectorMetricsDB, "_prepare_connection", record_prepare)

    vm = vector_metrics_db.VectorMetricsDB(tmp_path / "active.db")
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert calls == ["prepare"]
    assert vm._conn is not None
    assert vm._lazy_mode is False
