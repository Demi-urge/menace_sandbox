from __future__ import annotations

import logging
import time

import pytest

from bootstrap_timeout_policy import (
    SharedTimeoutCoordinator,
    build_progress_signal_hook,
    emit_bootstrap_heartbeat,
    read_bootstrap_heartbeat,
    wait_for_bootstrap_quiet_period,
)


def test_shared_timeout_progress_emits_heartbeat(tmp_path, monkeypatch):
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))

    coordinator = SharedTimeoutCoordinator(
        10.0,
        namespace="integration-test",
        logger=logging.getLogger("bootstrap-test"),
        signal_hook=build_progress_signal_hook(
            namespace="integration-test", run_id="progress-test"
        ),
    )

    coordinator.record_progress(
        "vectorizers",
        elapsed=1.5,
        remaining=4.5,
        metadata={"stage": "prepare_pipeline"},
    )

    heartbeat = read_bootstrap_heartbeat(max_age=5)
    assert heartbeat is not None
    assert heartbeat["label"] == "vectorizers"
    assert heartbeat["namespace"] == "integration-test"
    assert heartbeat["meta.stage"] == "prepare_pipeline"


def test_bootstrap_guard_waits_for_peer_activity(tmp_path, monkeypatch):
    heartbeat_path = tmp_path / "heartbeat.json"
    timeout_state = tmp_path / "timeout_state.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(timeout_state))

    emit_bootstrap_heartbeat(
        {
            "label": "prepare_pipeline",
            "remaining_budget": 30.0,
            "ts": time.time(),
            "pid": 99999,
            "host_load": 2.0,
        }
    )

    start = time.monotonic()
    delay, budget_scale = wait_for_bootstrap_quiet_period(
        logging.getLogger("bootstrap-test"),
        max_delay=0.2,
        poll_interval=0.05,
        load_threshold=0.1,
    )
    elapsed = time.monotonic() - start

    assert delay > 0
    assert elapsed >= delay
    assert budget_scale >= 1.0


def test_bootstrap_guard_blocks_when_queue_saturated(tmp_path, monkeypatch):
    heartbeat_path = tmp_path / "heartbeat.json"
    timeout_state = tmp_path / "timeout_state.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(timeout_state))
    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_QUEUE_CAPACITY", "1")
    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_QUEUE_BLOCK", "1")
    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_MAX_DELAY", "0.1")

    emit_bootstrap_heartbeat(
        {"label": "prepare_pipeline", "remaining_budget": 20.0, "ts": time.time(), "pid": 99999}
    )

    with pytest.raises(TimeoutError):
        wait_for_bootstrap_quiet_period(
            logging.getLogger("bootstrap-test"),
            poll_interval=0.02,
            load_threshold=0.1,
        )


def test_shared_timeout_tracks_gate_windows(monkeypatch):
    coordinator = SharedTimeoutCoordinator(
        5.0,
        component_budgets={"vectorizers": 2.0, "retrievers": 1.0},
        signal_hook=build_progress_signal_hook(
            namespace="integration-test", run_id="window-test"
        ),
    )

    coordinator.start_component_timers(
        {"vectorizers": 2.0, "retrievers": 1.0}, minimum=0.5
    )

    coordinator.record_progress(
        "vectorizers",
        elapsed=0.5,
        remaining=1.25,
        metadata={"stage": "prepare_pipeline"},
    )
    coordinator.record_progress(
        "retrievers",
        elapsed=0.2,
        remaining=0.5,
        metadata={"stage": "prepare_pipeline"},
    )

    snapshot = coordinator.snapshot()
    component_windows = snapshot.get("component_windows", {})

    assert set(component_windows) >= {"vectorizers", "retrievers"}
    assert component_windows["vectorizers"].get("remaining") == pytest.approx(1.25, rel=0.01)
    assert component_windows["retrievers"].get("remaining") == pytest.approx(0.5, rel=0.01)


def test_prepare_pipeline_inherits_guard_scale(monkeypatch, tmp_path):
    import coding_bot_interface as cbi

    heartbeat_path = tmp_path / "heartbeat.json"
    timeout_state = tmp_path / "timeout_state.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(timeout_state))
    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_MAX_DELAY", "0.2")

    emit_bootstrap_heartbeat(
        {"label": "prepare_pipeline", "remaining_budget": 30.0, "ts": time.time(), "pid": 99999, "host_load": 2.0}
    )

    monkeypatch.setattr(cbi, "enforce_bootstrap_timeout_policy", lambda logger=None: {})

    class _Pipeline:
        vector_bootstrap_heavy = True

        def __init__(self, **_kwargs):
            return

    pipeline, promote = cbi._prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=_Pipeline,
        context_builder=object(),
        bot_registry=object(),
        data_bot=object(),
        vectorizer_budget=1.0,
        retriever_budget=1.0,
        db_warmup_budget=1.0,
        bootstrap_guard=True,
    )

    promote(object())

    guard_context = cbi._PREPARE_PIPELINE_WATCHDOG.get("guard_context", {})
    derived_budgets = cbi._PREPARE_PIPELINE_WATCHDOG.get("derived_component_budgets") or {}

    assert guard_context.get("budget_scale", 1.0) > 1.0


def test_guard_context_persisted_for_component_floors(tmp_path, monkeypatch):
    heartbeat_path = tmp_path / "heartbeat.json"
    timeout_state = tmp_path / "timeout_state.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))
    monkeypatch.setenv("MENACE_BOOTSTRAP_TIMEOUT_STATE", str(timeout_state))
    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_MAX_DELAY", "0.3")
    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_QUEUE_CAPACITY", "2")
    monkeypatch.setenv("MENACE_BOOTSTRAP_GUARD_QUEUE_BLOCK", "0")

    emit_bootstrap_heartbeat(
        {
            "label": "prepare_pipeline",
            "remaining_budget": 30.0,
            "ts": time.time(),
            "pid": 99999,
            "host_load": 2.5,
        }
    )

    _, budget_scale = wait_for_bootstrap_quiet_period(
        logging.getLogger("bootstrap-test"),
        max_delay=0.25,
        poll_interval=0.05,
        load_threshold=0.5,
    )

    # Simulate a new process by clearing the in-memory guard cache
    import bootstrap_timeout_policy as btp

    btp._LAST_BOOTSTRAP_GUARD = None  # type: ignore[attr-defined]
    guard_context = btp.get_bootstrap_guard_context()
    component_floors = btp.load_component_timeout_floors()

    base_floor = btp._COMPONENT_TIMEOUT_MINIMUMS["vectorizers"]
    assert guard_context.get("budget_scale", 1.0) >= budget_scale
    assert component_floors["vectorizers"] >= base_floor * guard_context.get("budget_scale", 1.0)
