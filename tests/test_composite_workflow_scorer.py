import pytest

from menace_sandbox.db_router import init_db_router


def test_composite_workflow_scorer_records_metrics(tmp_path):
    init_db_router(
        "test_scorer",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    import sys
    import types

    class StubTracker:
        def __init__(self) -> None:
            self.metrics_history = {}
            self.roi_history = []
            self.module_deltas = {}

        def update(self, roi_before, roi_after, *, modules=None):
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
    from menace_sandbox.roi_calculator import ROICalculator
    from menace_sandbox.roi_scorer import CompositeWorkflowScorer

    metrics_db = MetricsDB(tmp_path / "metrics.db")
    pathway_db = PathwayDB(tmp_path / "pathways.db")
    tracker = StubTracker()
    calc = ROICalculator()
    calc.profiles[next(iter(calc.profiles))]["veto"] = {}
    scorer = CompositeWorkflowScorer(
        metrics_db,
        pathway_db,
        db_path=tmp_path / "roi.db",
        tracker=tracker,
        calculator=calc,
    )

    workflow_id = "demo_workflow"

    def workflow():
        metrics_db.log_eval(workflow_id, "fast_runtime", 0.01)
        metrics_db.log_eval(workflow_id, "fast_failures", 0.0)
        tracker.update(0.0, 1.0, modules=["fast"])

        metrics_db.log_eval(workflow_id, "slow_runtime", 0.05)
        metrics_db.log_eval(workflow_id, "slow_failures", 0.0)
        tracker.update(0.0, 2.0, modules=["slow"])
        return True

    run_id, scorecard = scorer.score(workflow_id, workflow)

    assert scorecard.workflow_id == workflow_id
    assert scorecard.success is True
    assert scorecard.runtime > 0
    assert scorecard.roi_gain > 0
    assert "fast_runtime" in scorecard.metrics
    assert "slow_runtime" in scorecard.metrics
    assert "workflow_synergy" in scorecard.metrics
    assert "bottleneck_index" in scorecard.metrics
    assert "patchability" in scorecard.metrics

    cur = scorer.conn.cursor()
    cur.execute(
        "SELECT workflow_id, run_id FROM workflow_results WHERE workflow_id=? AND run_id=?",
        (workflow_id, run_id),
    )
    assert cur.fetchone() == (workflow_id, run_id)

    cur.execute(
        "SELECT module, runtime, roi_delta FROM workflow_module_deltas WHERE workflow_id=? AND run_id=?",
        (workflow_id, run_id),
    )
    rows = cur.fetchall()
    mod_data = {mod: (runtime, delta) for mod, runtime, delta in rows}

    assert mod_data["slow"][0] > mod_data["fast"][0]
    assert mod_data["slow"][1] > mod_data["fast"][1]
    assert scorer.module_deltas()["fast"] == pytest.approx(1.0)
    assert scorer.module_deltas()["slow"] == pytest.approx(2.0)

