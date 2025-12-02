import sys
import threading
import types

import vector_metrics_db


def test_get_shared_defaults_to_stub_in_bootstrap(monkeypatch):
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    monkeypatch.setattr(vector_metrics_db, "_VECTOR_DB_INSTANCE", None)

    vm = vector_metrics_db.get_shared_vector_metrics_db()

    assert isinstance(vm, vector_metrics_db._BootstrapVectorMetricsStub)
    assert vm._activation_kwargs["warmup"] is True
    assert vm._activation_kwargs["ensure_exists"] is False
    assert vm._activation_kwargs["read_only"] is True


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
    # Allow the readiness thread to flush the buffer.
    for _ in range(20):
        if added:
            break
        threading.Event().wait(0.05)

    assert added
    assert vm._boot_stub_active is False
    assert vm._persistence_activated is True
    assert vm._resolved_path == tmp_path / "resolved.db"

