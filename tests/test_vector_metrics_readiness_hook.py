import sys
import threading
import types
from pathlib import Path
import sys

import vector_metrics_db


def test_get_shared_defaults_to_stub_in_bootstrap(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vm._activation_kwargs["warmup"] is True
    assert vm._activation_kwargs["ensure_exists"] is False
    assert vm._activation_kwargs["read_only"] is True


def test_get_shared_stub_when_bootstrap_timer_active(monkeypatch):
    calls = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            calls.append("activate_on_first_write")

        def log_embedding(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append(("log_embedding", args, kwargs))

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("activate_persistence")
            return self

    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "60")
    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert calls == []

    vm.activate_on_first_write()
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert calls == []

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        post_warmup=True
    )

    assert isinstance(activated, DummyVectorDB)
    assert calls[0][0] == "init"


def test_timer_stub_promotes_after_readiness(monkeypatch):
    ready = threading.Event()
    gate = threading.Event()

    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - simple gate
            ready.set()
            gate.wait(timeout)

    calls = []

    class DummyVectorDB:
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

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: _FakeSignal()),
    )
    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "15")
    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", False)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert not calls

    gate.set()

    for _ in range(20):
        if isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB):
            break
        threading.Event().wait(0.05)

    assert ready.is_set()
    assert isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB)
    assert calls and calls[0][0] == "init"


def test_get_shared_stub_when_readiness_hook_active(monkeypatch):
    calls = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            calls.append("activate_on_first_write")

        def log_embedding(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append(("log_embedding", args, kwargs))

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("activate_persistence")
            return self

    for env in vector_metrics_db._BOOTSTRAP_TIMER_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", True)

    vm = vector_metrics_db.get_shared_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert calls == []

    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert calls == []

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        post_warmup=True
    )

    assert isinstance(activated, DummyVectorDB)
    assert calls[0][0] == "init"


def test_get_shared_stub_in_bootstrap_even_with_explicit_flags(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP_MODE", "1")
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db(
        bootstrap_fast=False, warmup=False, ensure_exists=True, read_only=False
    )

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vm._activation_kwargs["warmup"] is True
    assert vm._activation_kwargs["ensure_exists"] is False
    assert vm._activation_kwargs["read_only"] is True


def test_activation_short_circuits_until_post_warmup(monkeypatch):
    calls = []

    class DummyVectorDB:
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

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")

    vm = vector_metrics_db.get_shared_vector_metrics_db(warmup=True)

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)

    skipped = vector_metrics_db.activate_shared_vector_metrics_db(
        reason="bootstrap_in_progress"
    )

    assert skipped is vm
    assert calls == []

    vm.configure_activation(warmup=False, ensure_exists=True, read_only=False)
    promoted = vector_metrics_db.activate_shared_vector_metrics_db(
        reason="post_warmup", post_warmup=True
    )

    assert isinstance(promoted, DummyVectorDB)
    assert calls and calls[0][0] == "init"


def test_bootstrap_stub_defers_creation_until_activated(monkeypatch, tmp_path):
    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - gate only
            threading.Event().wait(timeout)

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: _FakeSignal()),
    )
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", False)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)
    monkeypatch.chdir(tmp_path)

    created_paths: list[Path] = []

    class DummyVectorDB:
        def __init__(self, path, *args, **kwargs):
            self._boot_stub_active = False
            created_paths.append(Path(path))
            Path(path).touch()

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            pass

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            pass

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover
            return self

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)

    vm = vector_metrics_db.get_shared_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert not created_paths
    assert not (tmp_path / "vector_metrics.db").exists()

    activated = vector_metrics_db.activate_shared_vector_metrics_db(
        reason="bootstrap_ready", post_warmup=True
    )

    assert isinstance(activated, DummyVectorDB)
    assert [p.resolve() for p in created_paths] == [tmp_path / "vector_metrics.db"]
    assert (tmp_path / "vector_metrics.db").exists()


def test_readiness_hook_replays_buffer_after_bootstrap(monkeypatch, tmp_path):
    ready = threading.Event()
    release = threading.Event()

    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - trivial
            ready.set()
            release.wait(timeout)

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: _FakeSignal()),
    )

    added: list[vector_metrics_db.VectorMetric] = []

    def fake_prepare(self, init_start=None):
        self._resolved_path = tmp_path / "resolved.db"
        self._default_path = self._resolved_path
        self._conn = object()

    def fake_add(rec):  # pragma: no cover - simple collector
        added.append(rec)

    monkeypatch.setattr(vector_metrics_db.VectorMetricsDB, "_prepare_connection", fake_prepare)

    vm = vector_metrics_db.VectorMetricsDB(tmp_path / "warm.db", warmup=True)
    vm.add = fake_add  # type: ignore[assignment]
    vm._stub_buffer = [
        vector_metrics_db.VectorMetric(event_type="ready", db="test", tokens=1)
    ]

    vm.activate_persistence(reason="bootstrap_ready")

    ready.wait(timeout=1)
    release.set()
    vm._flush_stub_buffer()
    # Allow the readiness thread to flush the buffer.
    for _ in range(20):
        if added:
            break
        threading.Event().wait(0.05)

    assert added
    assert vm._resolved_path is None


def test_stub_promotes_after_readiness_signal(monkeypatch, tmp_path):
    gate = threading.Event()

    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - trivial gate
            gate.wait(timeout)

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: _FakeSignal()),
    )

    calls = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))
            self._boot_stub_active = False
            self.weights = {}
            self.logged = []

        def activate_on_first_write(self):
            calls.append("activate_on_first_write")

        def set_db_weights(self, weights):  # pragma: no cover - passthrough
            self.weights.update(weights)

        def get_db_weights(self, default=None):
            if self.weights:
                return dict(self.weights)
            return default or {}

        def log_embedding(self, *args, **kwargs):
            self.logged.append((args, kwargs))
            calls.append(("log_embedding", args, kwargs))

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db(warmup=True)
    vm.activate_on_first_write()
    vm.log_embedding("default", tokens=1, wall_time_ms=1.0)

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert calls == []

    gate.set()
    for _ in range(20):
        if isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB):
            break
        threading.Event().wait(0.05)

    activated = vector_metrics_db._VECTOR_DB_INSTANCE
    assert isinstance(activated, DummyVectorDB)
    assert calls[0][0] == "init"
    assert calls[1] == "activate_on_first_write"
    activated.log_embedding("default", tokens=2, wall_time_ms=2.0)
    assert calls[-1][0] == "log_embedding"
    assert getattr(activated, "logged", None)

