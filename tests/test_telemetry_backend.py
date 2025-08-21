from menace_sandbox.telemetry_backend import TelemetryBackend


def test_log_and_fetch(tmp_path):
    db = tmp_path / "tel.db"
    backend = TelemetryBackend(str(db))
    deltas = {"synergy_reliability": 0.9, "synergy_safety_rating": 0.8}
    backend.log_prediction(
        "wf1",
        predicted=1.0,
        actual=1.0,
        confidence=0.7,
        scenario_deltas=deltas,
        drift_flag=True,
        readiness=0.5,
        scenario="test",
    )
    history = backend.fetch_history("wf1")
    assert len(history) == 1
    rec = history[0]
    assert rec["scenario_deltas"] == deltas
    assert rec["drift_flag"] is True
    assert rec["scenario"] == "test"
    assert rec["readiness"] == 0.5
