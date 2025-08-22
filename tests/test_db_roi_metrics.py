import sqlite3
from menace_sandbox.roi_tracker import ROITracker


def test_update_db_metrics_and_report(tmp_path):
    tracker = ROITracker()
    metrics = {
        "A": {"roi": 3.0, "win_rate": 0.9, "regret_rate": 0.1},
        "B": {"roi": 1.0, "win_rate": 0.8, "regret_rate": 0.05},
        "C": {"roi": 2.0, "win_rate": 0.9, "regret_rate": 0.2},
    }
    db_path = tmp_path / "db_metrics.sqlite"
    tracker.update_db_metrics(metrics, sqlite_path=str(db_path))

    # ensure origin_db_deltas and stored metrics
    assert tracker.origin_db_delta_history["A"] == [3.0]
    assert tracker.db_roi_metrics["B"]["win_rate"] == 0.8

    # report should sort by win-rate desc then regret-rate asc
    report = tracker.db_roi_report()
    assert [r["origin_db"] for r in report] == ["A", "C", "B"]

    best = tracker.best_db_performance()
    assert best["highest_win_rate"]["origin_db"] == "A"
    assert best["lowest_regret"]["origin_db"] == "B"

    # sqlite entries
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT origin_db, win_rate, regret_rate, roi FROM db_roi_metrics"
        ).fetchall()
    assert ("A", 0.9, 0.1, 3.0) in rows
    assert len(rows) == 3


def test_update_db_metrics_zero_roi():
    tracker = ROITracker()
    tracker.update_db_metrics({"Z": {"roi": 0.0, "win_rate": 0.5, "regret_rate": 0.5}})
    assert tracker.origin_db_delta_history["Z"] == [0.0]
