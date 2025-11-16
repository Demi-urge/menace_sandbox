import menace.telemetry_backend as tb
from menace.db_router import DBRouter


def test_fetch_history_scopes(tmp_path):
    path = tmp_path / "tel.db"
    router = DBRouter("one", str(path), str(path))
    backend = tb.TelemetryBackend(str(path), router=router)
    backend.log_prediction(
        "wf", predicted=1.0, actual=1.0, confidence=0.9,
        scenario_deltas={}, drift_flag=False, readiness=0.2,
        ts="2021-01-01T00:00:00",
    )
    router.menace_id = "two"
    backend.log_prediction(
        "wf", predicted=0.5, actual=0.5, confidence=0.7,
        scenario_deltas={}, drift_flag=False, readiness=0.3,
        ts="2021-01-02T00:00:00",
    )
    router.menace_id = "one"

    local_hist = backend.fetch_history(scope="local")
    global_hist = backend.fetch_history(scope="global")
    all_hist = backend.fetch_history(scope="all")

    assert {h["predicted"] for h in local_hist} == {1.0}
    assert {h["predicted"] for h in global_hist} == {0.5}
    assert {h["predicted"] for h in all_hist} == {1.0, 0.5}


def test_fetch_history_filter_and_no_matches(tmp_path):
    path = tmp_path / "tel.db"
    router = DBRouter("one", str(path), str(path))
    backend = tb.TelemetryBackend(str(path), router=router)
    backend.log_prediction(
        "wf", predicted=1.0, actual=1.0, confidence=0.9,
        scenario_deltas={}, drift_flag=False, readiness=0.2,
        ts="2021-01-01T00:00:00",
    )
    router.menace_id = "two"
    backend.log_prediction(
        "wf", predicted=0.5, actual=0.5, confidence=0.7,
        scenario_deltas={}, drift_flag=False, readiness=0.3,
        ts="2021-01-02T00:00:00",
    )
    router.menace_id = "one"

    ts2 = "2021-01-02T00:00:00"

    assert {h["predicted"] for h in backend.fetch_history(workflow_id="wf", scope="local")} == {1.0}
    assert {h["predicted"] for h in backend.fetch_history(workflow_id="wf", scope="global")} == {0.5}
    assert {h["predicted"] for h in backend.fetch_history(workflow_id="wf", scope="all")} == {1.0, 0.5}

    assert backend.fetch_history(start_ts=ts2, scope="local") == []
    assert {h["predicted"] for h in backend.fetch_history(start_ts=ts2, scope="global")} == {0.5}

    assert backend.fetch_history(workflow_id="missing", scope="all") == []

