import json
import pytest

from menace_sandbox.db_router import init_db_router
from menace_sandbox.roi_results_db import ROIResultsDB, module_impact_report


def test_roi_results_db_add_and_report(tmp_path):
    init_db_router(
        "test_roi_results_db",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    db_path = tmp_path / "roi.db"
    db = ROIResultsDB(db_path)

    db.add_result(
        workflow_id="wf",
        run_id="r1",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=1.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
        module_deltas={"alpha": {"roi_delta": 0.1}, "beta": {"roi_delta": -0.1}},
    )
    db.add_result(
        workflow_id="wf",
        run_id="r2",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=1.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
        module_deltas={"alpha": {"roi_delta": 0.2}, "beta": {"roi_delta": -0.2}},
    )

    cur = db.conn.cursor()
    cur.execute("SELECT workflow_id, run_id, module_deltas FROM roi_results WHERE run_id='r2'")
    wf, run, deltas_json = cur.fetchone()
    assert wf == "wf"
    assert run == "r2"
    deltas = json.loads(deltas_json)
    assert deltas["alpha"]["roi_delta"] == pytest.approx(0.2)

    report = module_impact_report("wf", "r2", db_path)
    assert report["improved"] == {"alpha": pytest.approx(0.1)}
    assert report["regressed"] == {"beta": pytest.approx(-0.1)}

