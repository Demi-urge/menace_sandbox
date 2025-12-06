import logging

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
