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


def test_bootstrap_factory_always_returns_stub(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP_MODE", "1")
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_BOOTSTRAP_VECTOR_DB_STUB", None)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)

    vm = vector_metrics_db.bootstrap_aware_vector_metrics_db(
        ensure_exists=True, read_only=False
    )

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


def test_promotion_request_queued_during_timer_warmup(monkeypatch):
    gate = threading.Event()

    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - simple gate
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
    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "30")
    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", False)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db(
        ensure_exists=True, read_only=False
    )

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert not calls

    gate.set()

    for _ in range(40):
        if isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB):
            break
        threading.Event().wait(0.05)

    assert isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB)
    init_kwargs = calls[0][2]
    assert init_kwargs["ensure_exists"] is True
    assert init_kwargs["read_only"] is False


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


def test_readiness_timeout_promotes_stub(monkeypatch, tmp_path):
    class _HangingSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - gate only
            threading.Event().wait((timeout or 0.1) + 0.05)
            return False

    calls: list[str] = []
    reasons: list[str] = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            calls.append("init")
            self._boot_stub_active = False

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            calls.append("activate_on_first_write")

        def end_warmup(self, *args, **kwargs):  # pragma: no cover - passthrough
            calls.append("end_warmup")

        def activate_persistence(self, *args, **kwargs):  # pragma: no cover
            calls.append("activate_persistence")
            return self

    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "0.05")
    monkeypatch.setenv("VECTOR_METRICS_FIRST_WRITE_BUDGET_SECS", "0.05")
    monkeypatch.setitem(
        sys.modules, "bootstrap_readiness", types.SimpleNamespace(readiness_signal=lambda: _HangingSignal())
    )
    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_BOOTSTRAP_VECTOR_DB_STUB", None)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", False)
    monkeypatch.setattr(
        vector_metrics_db, "_increment_deferral_metric", lambda reason: reasons.append(reason)
    )

    vm = vector_metrics_db.get_shared_vector_metrics_db(
        ensure_exists=True, read_only=False
    )

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)

    for _ in range(60):
        if isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB):
            break
        threading.Event().wait(0.05)

    activated = vector_metrics_db._VECTOR_DB_INSTANCE
    assert isinstance(activated, DummyVectorDB)
    assert "readiness_timeout" in reasons


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


def test_warmup_stub_defers_filesystem_until_ready(monkeypatch, tmp_path):
    gate = threading.Event()

    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - gate only
            gate.wait(timeout)

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: _FakeSignal()),
    )
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", False)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)
    monkeypatch.chdir(tmp_path)

    created_paths: list[Path] = []
    seen_kwargs: list[dict[str, object]] = []

    class DummyVectorDB:
        def __init__(self, path, *args, **kwargs):
            self._boot_stub_active = False
            seen_kwargs.append(dict(kwargs))
            created_paths.append(Path(path))
            if kwargs.get("ensure_exists"):
                Path(path).touch()

        def activate_on_first_write(self):
            pass  # pragma: no cover - passthrough

        def end_warmup(self, *args, **kwargs):
            pass  # pragma: no cover - passthrough

        def activate_persistence(self, *args, **kwargs):
            return self  # pragma: no cover - passthrough

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)

    vm = vector_metrics_db.get_shared_vector_metrics_db(
        warmup=True, ensure_exists=True, read_only=False
    )

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert not created_paths
    assert not (tmp_path / "vector_metrics.db").exists()

    gate.set()
    for _ in range(40):
        if isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB):
            break
        threading.Event().wait(0.05)

    assert isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB)
    assert created_paths == [tmp_path / "vector_metrics.db"]
    assert seen_kwargs and seen_kwargs[0]["ensure_exists"] is True
    assert seen_kwargs[0]["read_only"] is False
    assert (tmp_path / "vector_metrics.db").exists()


def test_readiness_hook_replays_deferred_weights(monkeypatch, tmp_path):
    gate = threading.Event()

    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - gate only
            gate.wait(timeout)

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: _FakeSignal()),
    )
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", False)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)
    monkeypatch.chdir(tmp_path)

    weight_sets: list[dict[str, float]] = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            self._boot_stub_active = False
            self.weights: dict[str, float] = {}

        def activate_on_first_write(self):
            pass  # pragma: no cover - passthrough

        def end_warmup(self, *args, **kwargs):
            pass  # pragma: no cover - passthrough

        def activate_persistence(self, *args, **kwargs):
            return self  # pragma: no cover - passthrough

        def set_db_weights(self, weights):
            self.weights.update(weights)
            weight_sets.append(dict(weights))

        def get_db_weights(self, default=None):
            if self.weights:
                return dict(self.weights)
            return default or {}

    monkeypatch.setattr(vector_metrics_db, "VectorMetricsDB", DummyVectorDB)

    vector_metrics_db.ensure_vector_db_weights(
        ["alpha", "beta"], warmup=True, ensure_exists=True, read_only=False
    )

    vm = vector_metrics_db.get_shared_vector_metrics_db(
        warmup=True, ensure_exists=True, read_only=False
    )

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert not weight_sets

    gate.set()
    for _ in range(40):
        if isinstance(vector_metrics_db._VECTOR_DB_INSTANCE, DummyVectorDB):
            break
        threading.Event().wait(0.05)

    activated = vector_metrics_db._VECTOR_DB_INSTANCE
    assert isinstance(activated, DummyVectorDB)
    assert weight_sets and weight_sets[-1].get("alpha") == 1.0
    assert weight_sets[-1].get("beta") == 1.0


def test_readiness_hook_avoids_path_resolution(monkeypatch):
    class _FakeSignal:
        def await_ready(self, timeout=None):  # pragma: no cover - immediate
            return True

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: _FakeSignal()),
    )
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)
    monkeypatch.setattr(vector_metrics_db, "_BOOTSTRAP_VECTOR_DB_STUB", None)
    monkeypatch.setattr(vector_metrics_db, "_READINESS_HOOK_ARMED", False)
    monkeypatch.setattr(vector_metrics_db, "_MENACE_BOOTSTRAP_ENV_ACTIVE", None)
    monkeypatch.setattr(vector_metrics_db, "_PENDING_WEIGHTS", {"foo": 0.5})

    promotions: list[object] = []

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            promotions.append("init")
            self._boot_stub_active = False
            self.path = ":memory:"

        def get_db_weights(self, default=None):  # pragma: no cover - simple map
            return {}

        def set_db_weights(self, weights):
            promotions.append(("weights", dict(weights)))

        def activate_on_first_write(self):  # pragma: no cover - passthrough
            promotions.append("activate_on_first_write")

    def _fake_activate(self, *, reason: str, allow_override: bool = False):
        db = DummyVectorDB()
        vector_metrics_db._VECTOR_DB_INSTANCE = db
        self._delegate = db
        return db

    monkeypatch.setattr(
        vector_metrics_db._BootstrapVectorMetricsStub, "_activate", _fake_activate
    )

    vector_metrics_db._arm_shared_readiness_hook()

    assert isinstance(
        vector_metrics_db._VECTOR_DB_INSTANCE, vector_metrics_db._BootstrapVectorMetricsStub
    )

    for _ in range(40):
        if any(isinstance(call, tuple) for call in promotions):
            break
        threading.Event().wait(0.05)

    assert ("weights", {"foo": 0.5}) in promotions

