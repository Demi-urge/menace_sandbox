import pytest

from menace_sandbox.db_router import init_db_router
from menace_sandbox.roi_results_db import ROIResult, ROIResultsDB


def test_roi_results_db_trend(tmp_path):
    init_db_router(
        "test_roi_results_db",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    db = ROIResultsDB(tmp_path / "roi.db")
    for i, gain in enumerate([1.0, 2.0, 3.0]):
        db.add(
            ROIResult(
                workflow_id="wf",
                run_id=f"run{i}",
                module=None,
                runtime=1.0,
                success_rate=1.0,
                roi_gain=gain,
            )
        )
    rows = db.fetch_runs("wf")
    assert len(rows) == 3
    trend = db.trend("wf", "roi_gain")
    assert trend == pytest.approx(1.0)
