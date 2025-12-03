import vector_metrics_db


def test_shared_db_stub_defers_activation(monkeypatch):
    calls = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False
            self.weights = {}

        def activate_on_first_write(self):
            calls.append("activate_on_first_write")

        def set_db_weights(self, weights):
            self.weights.update(weights)

        def get_db_weights(self, default=None):
            if self.weights:
                return dict(self.weights)
            return default or {}

        def log_embedding(self, *args, **kwargs):
            calls.append(("log_embedding", args, kwargs))

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("activate_persistence")
            return self

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db(bootstrap_fast=True)

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert calls == []

    vector_metrics_db.ensure_vector_db_weights(
        ["alpha", "beta"], bootstrap_fast=True, warmup=True
    )

    assert calls == []

    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert vector_metrics_db._VECTOR_DB_INSTANCE is vm
    assert calls == []

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        post_warmup=True
    )

    assert isinstance(activated, DummyVectorDB)
    assert vector_metrics_db._VECTOR_DB_INSTANCE is activated
    assert calls[0][0] == "init"


def test_shared_stub_promotes_on_first_write(monkeypatch):
    calls = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False
            self.weights = {}
            self.logged = []

        def activate_on_first_write(self):
            calls.append("activate_on_first_write")

        def set_db_weights(self, weights):  # pragma: no cover - simple passthrough
            self.weights.update(weights)

        def get_db_weights(self, default=None):
            if self.weights:
                return dict(self.weights)
            return default or {}

        def log_embedding(self, *args, **kwargs):
            self.logged.append((args, kwargs))
            calls.append(("log_embedding", args, kwargs))

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("activate_persistence")
            return self

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db(warmup=True)
    vector_metrics_db.ensure_vector_db_weights(["alpha"], warmup=True)

    vm.configure_activation(warmup=False, ensure_exists=True, read_only=False)
    vm.activate_on_first_write()
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, vector_metrics_db._BootstrapVectorMetricsStub)
    assert calls == []

    activated = vector_metrics_db.activate_shared_vector_metrics_db(post_warmup=True)

    assert isinstance(activated, DummyVectorDB)
    assert vector_metrics_db._VECTOR_DB_INSTANCE is activated
    assert calls[0][0] == "init"
    assert calls[1] == "activate_on_first_write"
    activated.log_embedding("default", tokens=1, wall_time_ms=1.0)
    assert any(entry[0] == "log_embedding" for entry in calls if isinstance(entry, tuple))


def test_bootstrap_helper_returns_stub_without_sql(monkeypatch):
    class BoomVectorDB:
        def __init__(self, *args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("should not initialise sqlite during bootstrap")

    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", BoomVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_bootstrap_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vector_metrics_db._VECTOR_DB_INSTANCE is vm


def test_ensure_weights_skips_sqlite_when_warmup(monkeypatch):
    class BoomVectorDB:
        def __init__(self, *args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("should not touch sqlite during warmup")

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", BoomVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vector_metrics_db.ensure_vector_db_weights(["alpha"], warmup=True)

    assert isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, vector_metrics_db._BootstrapVectorMetricsStub)
