import json
import pytest
import sqlite3
import sys
import types


class _StubDBRouter:
    def __init__(self, _name, local_path, _shared_path):
        self.path = local_path

    def get_connection(self, _name, operation: str | None = None):
        return sqlite3.connect(self.path)  # noqa: SQL001


sys.modules.setdefault(
    "menace_sandbox.db_router",
    types.SimpleNamespace(
        init_db_router=lambda *a, **k: None,
        DBRouter=_StubDBRouter,
        LOCAL_TABLES=set(),
    ),
)
sys.modules.setdefault("db_router", sys.modules["menace_sandbox.db_router"])

from menace_sandbox.db_router import init_db_router  # noqa: E402
from menace_sandbox.roi_results_db import ROIResultsDB, module_impact_report  # noqa: E402


def test_roi_results_db_add_and_report(tmp_path):
    init_db_router(
        "test_roi_results_db",
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
        failure_reason=None,
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
        failure_reason="alpha: boom",
    )

    cur = db.conn.cursor()
    cur.execute(
        "SELECT workflow_id, run_id, module_deltas, failure_reason "
        "FROM workflow_results WHERE run_id='r2'",
    )
    wf, run, deltas_json, failure_reason = cur.fetchone()
    assert wf == "wf"
    assert run == "r2"
    deltas = json.loads(deltas_json)
    assert deltas["alpha"]["roi_delta"] == pytest.approx(0.2)
    assert deltas["alpha"]["success_rate"] == pytest.approx(0.25)
    assert failure_reason == "alpha: boom"

    report = module_impact_report("wf", "r2", db_path)
    assert report["improved"]["alpha"]["roi_delta"] == pytest.approx(0.1)
    assert report["improved"]["alpha"]["success_rate"] == pytest.approx(0.25)
    assert report["regressed"]["beta"]["roi_delta"] == pytest.approx(-0.1)
    assert report["regressed"]["beta"]["success_rate"] == pytest.approx(0.5)

    assert db.projected_revenue() == pytest.approx(2.0)
