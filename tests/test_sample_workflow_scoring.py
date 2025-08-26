import sys
import types
import json

import pytest

from menace_sandbox.db_router import init_db_router


def test_sample_workflow_scoring(tmp_path):
    init_db_router(
        "sample_workflow",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    class StubTracker:
        def __init__(self) -> None:
            self.metrics_history = {}
            self.roi_history = []
            self.module_deltas = {}

        def update(self, roi_before, roi_after, *, modules=None, **_: object) -> None:
            delta = roi_after - roi_before
            self.roi_history.append(delta)
            if modules:
                for m in modules:
                    self.module_deltas.setdefault(m, []).append(delta)

        def cache_correlations(self, pairs):
            pass

    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=StubTracker)
    )

    from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer
    from menace_sandbox.roi_results_db import ROIResultsDB

    tracker = StubTracker()
    results_db = ROIResultsDB(tmp_path / "roi.db")
    scorer = CompositeWorkflowScorer(tracker=tracker, results_db=results_db)

    workflow_id = "demo"

    def step1() -> bool:
        return True

    def step2() -> bool:
        return True

    run_id, result = scorer.score_workflow(
        workflow_id, {"step1": step1, "step2": step2}
    )

    assert result["success_rate"] == pytest.approx(1.0)

    cur = results_db.conn.cursor()
    cur.execute(
        "SELECT module_deltas FROM workflow_results WHERE workflow_id=? AND run_id=?",
        (workflow_id, run_id),
    )
    deltas = json.loads(cur.fetchone()[0])
    assert deltas["step1"]["success_rate"] == pytest.approx(1.0)
    assert deltas["step2"]["success_rate"] == pytest.approx(1.0)
