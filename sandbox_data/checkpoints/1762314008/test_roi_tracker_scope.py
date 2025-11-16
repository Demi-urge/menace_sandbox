import sqlite3
import menace.roi_tracker as rt


class DummyRouter:
    def __init__(self, menace_id: str, path: str):
        self.menace_id = menace_id
        self.path = path

    def get_connection(self, table_name: str):
        return sqlite3.connect(self.path, check_same_thread=False)


def test_fetch_prediction_events_scopes(tmp_path):
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE roi_prediction_events (ts TEXT, workflow_id TEXT, predicted_roi REAL, actual_roi REAL, confidence REAL, scenario_deltas TEXT, drift_flag INTEGER, source_menace_id TEXT)"
    )
    conn.execute(
        "INSERT INTO roi_prediction_events VALUES (?,?,?,?,?,?,?,?)",
        ("2021-01-01", "wf", 1.0, 1.0, 0.9, "{}", 0, "one"),
    )
    conn.execute(
        "INSERT INTO roi_prediction_events VALUES (?,?,?,?,?,?,?,?)",
        ("2021-01-02", "wf", 2.0, 2.0, 0.8, "{}", 0, "two"),
    )
    conn.commit()
    conn.close()

    rt.router = DummyRouter("one", str(db_path))
    tracker = object.__new__(rt.ROITracker)
    local = rt.ROITracker.fetch_prediction_events(tracker, scope="local")
    global_ = rt.ROITracker.fetch_prediction_events(tracker, scope="global")
    all_ = rt.ROITracker.fetch_prediction_events(tracker, scope="all")

    assert {e["predicted_roi"] for e in local} == {1.0}
    assert {e["predicted_roi"] for e in global_} == {2.0}
    assert {e["predicted_roi"] for e in all_} == {1.0, 2.0}

    # start_ts filter excludes earlier local event
    assert rt.ROITracker.fetch_prediction_events(tracker, start_ts="2021-01-02", scope="local") == []
    assert {e["predicted_roi"] for e in rt.ROITracker.fetch_prediction_events(tracker, start_ts="2021-01-02", scope="global")} == {2.0}

    # workflow filter with no matching entries
    assert rt.ROITracker.fetch_prediction_events(tracker, workflow_id="missing", scope="all") == []

