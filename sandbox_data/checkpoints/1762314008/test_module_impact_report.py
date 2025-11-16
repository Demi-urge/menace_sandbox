from menace_sandbox.db_router import init_db_router
from menace_sandbox.roi_results_db import ROIResultsDB, module_impact_report


def test_module_impact_report(tmp_path):
    init_db_router(
        "test_module_impact_report",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    db_path = tmp_path / "roi.db"
    db = ROIResultsDB(db_path)

    db.log_result(
        workflow_id="wf",
        run_id="r1",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=1.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
        module_deltas={
            "alpha": {"roi_delta": 0.1, "success_rate": 0.5},
            "beta": {"roi_delta": -0.1, "success_rate": 1.0},
        },
    )
    db.log_result(
        workflow_id="wf",
        run_id="r2",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=1.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
        module_deltas={
            "alpha": {"roi_delta": 0.2, "success_rate": 0.25},
            "beta": {"roi_delta": -0.2, "success_rate": 0.5},
        },
    )

    report = module_impact_report("wf", "r2", db_path)
    assert report["improved"]["alpha"] == {
        "roi_delta": 0.1,
        "success_rate": 0.25,
    }
    assert report["regressed"]["beta"] == {
        "roi_delta": -0.1,
        "success_rate": 0.5,
    }

