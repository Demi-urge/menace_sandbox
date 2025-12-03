import sys
import time
import types

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


def test_stub_remains_active_during_warmup_first_write(monkeypatch):
    calls: list[str] = []

    class SlowVectorDB:
        def __init__(self, *args, **kwargs):  # pragma: no cover - guard
            calls.append("init")
            time.sleep(0.2)

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", SlowVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db(bootstrap_fast=True)

    start = time.perf_counter()
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    assert vector_metrics_db._VECTOR_DB_INSTANCE is vm
    assert calls == []


def test_warmup_stub_flag_blocks_activation_without_env(monkeypatch):
    calls: list[str] = []

    class GuardVectorDB:
        def __init__(self, *args, **kwargs):  # pragma: no cover - guard
            calls.append("init")

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", GuardVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    class DummyReadiness:
        def await_ready(self, timeout=None):  # pragma: no cover - background thread
            raise RuntimeError("skip readiness")

    dummy_module = types.SimpleNamespace(readiness_signal=lambda: DummyReadiness())
    monkeypatch.setitem(sys.modules, "bootstrap_readiness", dummy_module)

    vm = vector_metrics_db.get_vector_metrics_db(warmup_stub=True)

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vm._boot_stub_active
    assert vm._readiness_hook_registered
    assert calls == []

    activated = vm._activate(reason="attribute_access")

    assert isinstance(activated, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vm._boot_stub_active
    assert calls == []


def test_first_write_timeboxed_activation_falls_back(monkeypatch):
    monkeypatch.setenv("VECTOR_METRICS_FIRST_WRITE_BUDGET_SECS", "0.05")
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    class SlowVectorDB:
        def __init__(self, path, *args, **kwargs):
            self.path = path
            self.logged = []
            self._boot_stub_active = False
            if path != ":memory:":
                time.sleep(0.2)

        def activate_on_first_write(self):
            self.logged.append("activate_on_first_write")

        def log_embedding(self, *args, **kwargs):
            self.logged.append((args, kwargs))

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            return None

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            return self

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", SlowVectorDB)

    vm = vector_metrics_db.get_shared_vector_metrics_db(warmup=True)
    vm.activate_on_first_write()

    time.sleep(0.06)
    start = time.perf_counter()
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.15
    delegate = vector_metrics_db._VECTOR_DB_INSTANCE
    assert isinstance(delegate, SlowVectorDB)
    assert delegate.path == ":memory:"
    assert any(entry == "activate_on_first_write" for entry in delegate.logged)
    assert any(call for call in delegate.logged if isinstance(call, tuple))
