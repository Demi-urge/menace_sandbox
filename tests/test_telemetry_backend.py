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


def test_bootstrap_with_router_stub_and_list_local_tables(monkeypatch, tmp_path):
    import sqlite3
    import menace_sandbox.telemetry_backend as tb

    conn = sqlite3.connect(str(tmp_path / "tel.db"))
    router = type("Router", (), {"local_conn": conn, "shared_conn": conn, "menace_id": "telemetry", "get_connection": lambda self, *_a, **_k: conn})()

    monkeypatch.setattr(tb, "LOCAL_TABLES", [])

    backend = tb.TelemetryBackend(str(tmp_path / "tel.db"), router=router)
    assert backend.router is router
    assert "roi_telemetry" in tb.LOCAL_TABLES
    assert "roi_prediction_events" in tb.LOCAL_TABLES


def test_select_router_tolerates_missing_local_conn(monkeypatch, tmp_path):
    import menace_sandbox.telemetry_backend as tb

    broken_router = type("Router", (), {"menace_id": "telemetry", "shared_conn": object()})()
    monkeypatch.setattr(tb, "GLOBAL_ROUTER", broken_router)

    fallback = object()

    def fake_init(*_a, **_k):
        return fallback

    monkeypatch.setattr(tb, "init_db_router", fake_init)

    chosen = tb._select_router(str(tmp_path / "tel.db"))
    assert chosen is fallback
