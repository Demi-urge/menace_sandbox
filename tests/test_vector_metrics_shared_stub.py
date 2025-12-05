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


def test_bootstrap_helper_memoizes_warmup_stub(monkeypatch):
    hook_calls: list[vector_metrics_db._BootstrapVectorMetricsStub] = []

    def _fake_register(self):
        self._readiness_hook_registered = True
        hook_calls.append(self)

    class BoomVectorDB:
        def __init__(self, *args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("should not initialise sqlite during warmup")

    monkeypatch.setattr(
        vector_metrics_db._BootstrapVectorMetricsStub,
        "register_readiness_hook",
        _fake_register,
    )
    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", BoomVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_BOOTSTRAP_VECTOR_DB_STUB", None)

    vm = vector_metrics_db.get_bootstrap_vector_metrics_db()
    vm_again = vector_metrics_db.get_bootstrap_vector_metrics_db()

    assert vm is vm_again
    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vector_metrics_db._VECTOR_DB_INSTANCE is vm
    assert vm._post_warmup_activation_requested is True
    assert hook_calls == []


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
    assert not vm._readiness_hook_registered
    assert calls == []

    activated = vm._activate(reason="attribute_access")

    assert isinstance(activated, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vm._boot_stub_active
    assert calls == []


def test_warmup_stub_waits_for_post_warmup_activation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    vector_metrics_db._MENACE_BOOTSTRAP_ENV_ACTIVE = None
    vector_metrics_db._VECTOR_DB_INSTANCE = None

    calls = []

    class RecordingVectorDB:
        def __init__(self, path, *args, **kwargs):
            calls.append(("init", path, args, kwargs))
            self._boot_stub_active = False

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            calls.append("activate_on_first_write")

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("activate_persistence")
            return self

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", RecordingVectorDB)

    vm = vector_metrics_db.get_bootstrap_shared_vector_metrics_db()
    db_path = tmp_path / "vector_metrics.db"

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vm._activation_blocked is True
    assert vm._activate_on_first_write is False
    assert not vm._readiness_hook_registered
    assert not db_path.exists()

    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert calls == []
    assert not db_path.exists()

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        post_warmup=True
    )

    assert isinstance(activated, RecordingVectorDB)
    assert vector_metrics_db._VECTOR_DB_INSTANCE is activated
    assert vm._delegate is activated
    assert calls and calls[0][0] == "init"


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
    assert isinstance(delegate, vector_metrics_db._BootstrapVectorMetricsStub)
    assert delegate._boot_stub_active


def test_configure_activation_defers_until_post_warmup(monkeypatch):
    calls = []

    class RecordingVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            calls.append("activate_on_first_write")

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("activate_persistence")
            return self

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", RecordingVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_vector_metrics_db(warmup=True)

    vm.configure_activation(
        warmup=False, ensure_exists=True, read_only=False, path="/tmp/sqlite.db"
    )

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert calls == []
    assert vm._queued_activation_kwargs["ensure_exists"] is True
    assert vm._queued_activation_kwargs["read_only"] is False
    assert vm._queued_activation_kwargs["warmup"] is False

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        post_warmup=True
    )

    assert isinstance(activated, RecordingVectorDB)
    assert calls and calls[0][0] == "init"


def test_bootstrap_callers_keep_stub_until_activation_signal(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    calls = []

    class RecordingVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            calls.append("activate_on_first_write")

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("activate_persistence")
            return self

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", RecordingVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)
    assert calls == []

    monkeypatch.delenv("MENACE_BOOTSTRAP")

    vm_again = vector_metrics_db.get_vector_metrics_db(
        warmup=False, ensure_exists=True, read_only=False
    )

    assert vm_again is vm
    assert calls == []

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        post_warmup=True
    )

    assert isinstance(activated, RecordingVectorDB)
    assert calls and calls[0][0] == "init"
    assert any(entry == "end_warmup" for entry in calls)
    assert any(entry == "activate_persistence" for entry in calls)


def test_bootstrap_thread_first_write_defers_activation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    vector_metrics_db._MENACE_BOOTSTRAP_ENV_ACTIVE = None

    class DummyReadiness:
        def await_ready(self, timeout=None):  # pragma: no cover - background thread
            return False

    dummy_module = types.SimpleNamespace(readiness_signal=lambda: DummyReadiness())
    monkeypatch.setitem(sys.modules, "bootstrap_readiness", dummy_module)

    bootstrap_flags = [True, False]

    def fake_bootstrap_thread_context():
        if bootstrap_flags:
            return bootstrap_flags.pop(0)
        return False

    monkeypatch.setattr(
        vector_metrics_db, "_bootstrap_thread_context", fake_bootstrap_thread_context
    )

    stub = vector_metrics_db._BootstrapVectorMetricsStub(
        path=None,
        bootstrap_fast=False,
        warmup=False,
        ensure_exists=True,
        read_only=False,
    )
    stub.activate_on_first_write()

    stub.log_embedding("default", tokens=1, wall_time_ms=1.0)
    stub.log_embedding("default", tokens=2, wall_time_ms=2.0)

    assert stub._delegate is None
    assert stub._activation_kwargs.get("path") is None
    assert not stub._path_resolution_allowed
    assert not (tmp_path / "vector_metrics.db").exists()
    assert stub._readiness_hook_registered
