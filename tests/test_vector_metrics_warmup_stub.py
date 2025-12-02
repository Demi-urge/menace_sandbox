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


def test_warmup_requires_explicit_activation(monkeypatch, tmp_path):
    orig_prepare = vector_metrics_db.VectorMetricsDB._prepare_connection
    calls = []

    def record_prepare(self, init_start=None):
        calls.append("prepare")
        return orig_prepare(self, init_start)

    monkeypatch.setattr(vector_metrics_db.VectorMetricsDB, "_prepare_connection", record_prepare)

    vm = vector_metrics_db.VectorMetricsDB(tmp_path / "warm_write.db", warmup=True)
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert calls == []
    vm.activate_persistence(reason="warmup_complete")
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert calls == ["prepare"]
    assert vm._conn is not None
    assert vm._boot_stub_active is False
    assert vm._warmup_mode is False


def test_warmup_stub_skips_filesystem(tmp_path):
    path = tmp_path / "warm.db"

    vm = vector_metrics_db.VectorMetricsDB(path, warmup=True)

    assert vm._schema_defaults_initialized is False
    assert vm._schema_cache == {}
    assert vm._default_columns == {}
    assert not path.exists()
    assert vm.ready_probe() == str(path)
    assert not path.exists()


def test_menace_bootstrap_uses_stub(monkeypatch, tmp_path):
    path = tmp_path / "bootstrap.db"

    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")

    vm = vector_metrics_db.VectorMetricsDB(path)

    weights = vm.get_db_weights()

    assert vm._boot_stub_active is True
    assert vm._conn is None
    assert vm.router is None
    assert vm._schema_defaults_initialized is False
    assert weights == {}
    assert not path.exists()

