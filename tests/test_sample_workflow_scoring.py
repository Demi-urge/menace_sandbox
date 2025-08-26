import sys
import types

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

    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=StubTracker)
    )

    from menace_sandbox.data_bot import MetricsDB
    from menace_sandbox.neuroplasticity import PathwayDB
    from menace_sandbox.roi_scorer import CompositeWorkflowScorer

    metrics_db = MetricsDB(tmp_path / "metrics.db")
    pathway_db = PathwayDB(tmp_path / "pathways.db")
    tracker = StubTracker()
    scorer = CompositeWorkflowScorer(
        metrics_db, pathway_db, db_path=tmp_path / "roi.db", tracker=tracker
    )

    workflow_id = "demo"

    def step1() -> bool:
        metrics_db.log_eval(workflow_id, "step1_runtime", 0.01)
        return True

    def step2() -> bool:
        metrics_db.log_eval(workflow_id, "step2_runtime", 0.02)
        return True

    run_id, result = scorer.score_workflow(
        workflow_id, {"step1": step1, "step2": step2}
    )

    assert result["success"] is True

    cur = scorer.conn.cursor()
    cur.execute(
        "SELECT workflow_id, run_id FROM workflow_results WHERE workflow_id=? AND run_id=?",
        (workflow_id, run_id),
    )
    assert cur.fetchone() == (workflow_id, run_id)

    cur.execute(
        "SELECT module FROM workflow_module_deltas WHERE workflow_id=? AND run_id=?",
        (workflow_id, run_id),
    )
    modules = {row[0] for row in cur.fetchall()}
    assert {"step1", "step2"}.issubset(modules)

