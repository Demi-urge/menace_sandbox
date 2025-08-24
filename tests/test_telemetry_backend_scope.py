import menace.telemetry_backend as tb
from menace.db_router import DBRouter


def test_fetch_history_scopes(tmp_path):
    local = tmp_path / "tel.db"
    shared = tmp_path / "shared.db"

    router_a = DBRouter("one", str(local), str(shared))
    backend_a = tb.TelemetryBackend(str(local), router=router_a)
    backend_a.log_prediction("wf", predicted=1.0, actual=1.0, confidence=0.9, scenario_deltas={}, drift_flag=False, readiness=0.2)

    router_b = DBRouter("two", str(local), str(shared))
    backend_b = tb.TelemetryBackend(str(local), router=router_b)
    backend_b.log_prediction("wf", predicted=0.5, actual=0.5, confidence=0.7, scenario_deltas={}, drift_flag=False, readiness=0.3)

    local_hist = backend_a.fetch_history(scope="local")
    global_hist = backend_a.fetch_history(scope="global")
    all_hist = backend_a.fetch_history(scope="all")

    assert {h["predicted"] for h in local_hist} == {1.0}
    assert {h["predicted"] for h in global_hist} == {0.5}
    assert {h["predicted"] for h in all_hist} == {1.0, 0.5}
