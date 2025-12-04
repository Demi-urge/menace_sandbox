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


def test_readiness_hook_warmup_skips_io(monkeypatch, tmp_path):
    path = tmp_path / "warm-readiness.db"

    monkeypatch.setenv("VECTOR_SERVICE_WARMUP", "1")

    def explode_default_path(*_args, **_kwargs):  # pragma: no cover - safety guard
        raise AssertionError("default path resolution should be skipped during warmup")

    def explode_router(*_args, **_kwargs):  # pragma: no cover - safety guard
        raise AssertionError("router should not initialise during warmup")

    monkeypatch.setattr(
        vector_metrics_db, "default_vector_metrics_path", explode_default_path
    )
    monkeypatch.setattr(vector_metrics_db, "init_db_router", explode_router)

    vm = vector_metrics_db.VectorMetricsDB(path, warmup=True)

    assert vm.router is None
    assert vm._readiness_hook_registered is False
    assert not path.exists()
    assert vm.ready_probe() == str(path)
    assert not path.exists()


def test_shared_db_uses_stub_during_menace_bootstrap(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    vector_metrics_db._MENACE_BOOTSTRAP_ENV_ACTIVE = None
    vector_metrics_db._VECTOR_DB_INSTANCE = None

    vm = vector_metrics_db.get_shared_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert getattr(vm, "_activation_blocked", False) is True
    assert getattr(vm, "_warmup_guarded_activation", False) is True
    assert getattr(vm, "_delegate", None) is None


def test_lazy_activation_promotes_after_warmup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    vector_metrics_db._MENACE_BOOTSTRAP_ENV_ACTIVE = None
    vector_metrics_db._VECTOR_DB_INSTANCE = None

    vm = vector_metrics_db.get_shared_vector_metrics_db()
    vector_metrics_db.activate_shared_vector_metrics_db(reason="post_warmup")

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert getattr(vm, "_delegate", None) is None

    vm._release_activation_block(reason="warmup_complete", configure_ready=True)

    delegate = getattr(vm, "_delegate", None)
    assert isinstance(delegate, vector_metrics_db.VectorMetricsDB)
    assert delegate._warmup_mode is False

