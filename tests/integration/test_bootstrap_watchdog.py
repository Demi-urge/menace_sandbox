from __future__ import annotations

import logging
import time

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
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))

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
