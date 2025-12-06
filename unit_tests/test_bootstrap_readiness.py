import logging
import time

from bootstrap_readiness import (
    CORE_COMPONENTS,
    ReadinessSignal,
    _minimal_readiness_payload,
    log_component_readiness,
    stop_bootstrap_heartbeat_keepalive,
)
from bootstrap_timeout_policy import emit_bootstrap_heartbeat


def test_readiness_signal_transitions_ready_with_keepalive(tmp_path, monkeypatch):
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))

    stop_bootstrap_heartbeat_keepalive()

    ready_components = {component: "ready" for component in CORE_COMPONENTS}
    emit_bootstrap_heartbeat(
        {"readiness": {"components": ready_components, "ready": True, "online": True}}
    )

    payload = _minimal_readiness_payload()
    emit_bootstrap_heartbeat(payload)

    signal = ReadinessSignal(poll_interval=0.01, max_age=5.0)
    try:
        probe = signal.probe()
    finally:
        stop_bootstrap_heartbeat_keepalive()

    assert payload["readiness"]["components"] == ready_components
    assert payload["readiness"]["ready"] is True
    assert payload["readiness"]["online"] is True
    assert probe.ready is True
    assert probe.lagging_core == ()
    assert probe.degraded_core == ()
    assert probe.degraded_online is False


def test_component_readiness_timestamps_from_keepalive(tmp_path, monkeypatch, caplog):
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))

    stop_bootstrap_heartbeat_keepalive()

    payload = _minimal_readiness_payload()
    emit_bootstrap_heartbeat(payload)

    caplog.set_level(logging.INFO)
    readiness_snapshot = log_component_readiness(
        logger=logging.getLogger("bootstrap-readiness-test"),
        max_age=5.0,
    )

    try:
        assert readiness_snapshot
        assert all(entry.get("ts") for entry in readiness_snapshot.values())
        assert "no bootstrap readiness timestamps available" not in [
            record.message for record in caplog.records
        ]
    finally:
        stop_bootstrap_heartbeat_keepalive()


def test_readiness_signal_ready_from_keepalive_payload(tmp_path, monkeypatch, caplog):
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))

    stop_bootstrap_heartbeat_keepalive()

    caplog.set_level(logging.INFO)

    payload = _minimal_readiness_payload()
    emit_bootstrap_heartbeat(payload)

    signal = ReadinessSignal(poll_interval=0.01, max_age=1.0)
    try:
        probe = signal.probe()
        readiness_snapshot = log_component_readiness(
            logger=logging.getLogger("bootstrap-readiness-test"),
            max_age=1.0,
        )
    finally:
        stop_bootstrap_heartbeat_keepalive()

    assert probe.ready is True
    assert probe.lagging_core == ()
    assert readiness_snapshot
    assert all(entry.get("ts") for entry in readiness_snapshot.values())
    assert "no bootstrap readiness timestamps available" not in [
        record.message for record in caplog.records
    ]


def test_keepalive_grace_transitions_components_ready(tmp_path, monkeypatch):
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("MENACE_BOOTSTRAP_WATCHDOG_PATH", str(heartbeat_path))

    stop_bootstrap_heartbeat_keepalive()

    # Accelerate grace period for test determinism.
    import bootstrap_readiness

    bootstrap_readiness._KEEPALIVE_COMPONENT_GRACE_SECONDS = 0.05
    bootstrap_readiness._KEEPALIVE_GRACE_START = None
    bootstrap_readiness._LAST_COMPONENT_SNAPSHOT = None

    payload = _minimal_readiness_payload()
    emit_bootstrap_heartbeat(payload)

    time.sleep(0.06)

    payload = _minimal_readiness_payload()
    emit_bootstrap_heartbeat(payload)

    signal = ReadinessSignal(poll_interval=0.01, max_age=1.0)
    try:
        assert signal.await_ready(timeout=1.0) is True
        readiness = payload["readiness"]
        assert all(status == "ready" for status in readiness["components"].values())
        assert readiness["ready"] is True
    finally:
        stop_bootstrap_heartbeat_keepalive()
