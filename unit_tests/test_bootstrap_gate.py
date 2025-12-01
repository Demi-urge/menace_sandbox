import time

import bootstrap_gate
from bootstrap_timeout_policy import (
    _BOOTSTRAP_TIMEOUT_MINIMUMS,
    compute_gate_backoff,
    resolve_bootstrap_gate_timeout,
)


def test_resolve_bootstrap_gate_timeout_honours_floor(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "12")
    monkeypatch.setenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", "24")
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(tmp_path / "state.json"))
    floors = dict(_BOOTSTRAP_TIMEOUT_MINIMUMS)
    monkeypatch.setitem(floors, "MENACE_BOOTSTRAP_WAIT_SECS", 8.0)
    monkeypatch.setitem(floors, "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", 16.0)

    with monkeypatch.context() as ctx:
        ctx.setattr("bootstrap_timeout_policy._BOOTSTRAP_TIMEOUT_MINIMUMS", floors)
        timeout = resolve_bootstrap_gate_timeout(fallback_timeout=2.0)

    assert timeout >= floors["MENACE_BOOTSTRAP_WAIT_SECS"]


def test_compute_gate_backoff_scales_with_queue(monkeypatch):
    slow = compute_gate_backoff(queue_depth=1, attempt=1, remaining=10.0)
    faster = compute_gate_backoff(queue_depth=3, attempt=2, remaining=10.0)
    reentrant = compute_gate_backoff(queue_depth=0, attempt=3, remaining=10.0, reentrant=True)

    assert faster > slow
    assert faster <= 10.0
    assert reentrant < faster


def test_wait_for_bootstrap_gate_retries_with_backoff(monkeypatch):
    attempts = {"count": 0}

    def _wait_until_ready(timeout=None, check=None, description=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("busy")
        return True

    heartbeats = {"queue_depth": 2, "ts": time.time()}
    monotonic_samples = [0.0, 0.05, 0.1]

    monkeypatch.setattr(
        bootstrap_gate.bootstrap_manager, "wait_until_ready", _wait_until_ready
    )
    monkeypatch.setattr(bootstrap_gate, "read_bootstrap_heartbeat", lambda: heartbeats)
    monkeypatch.setattr(bootstrap_gate, "compute_gate_backoff", lambda **_: 0.0)
    monkeypatch.setattr(
        bootstrap_gate, "resolve_bootstrap_gate_timeout", lambda fallback_timeout=0.0: 1.0
    )
    monkeypatch.setattr(bootstrap_gate.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bootstrap_gate.time,
        "monotonic",
        lambda: monotonic_samples.pop(0) if monotonic_samples else 0.1,
    )

    bootstrap_gate.wait_for_bootstrap_gate(timeout=0.5, description="test gate")

    assert attempts["count"] == 2
