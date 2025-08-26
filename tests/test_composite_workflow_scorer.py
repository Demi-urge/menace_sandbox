import json
import pytest
import types
import sys

from menace_sandbox.db_router import init_db_router


def test_composite_workflow_scorer_writes_results(tmp_path, monkeypatch):
    init_db_router(
        "test_composite_scorer",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    class StubTracker:
        def __init__(self):
            self.metrics_history = {
                "synergy_efficiency": [1.0],
                "synergy_reliability": [1.0],
            }
            self.roi_history = [1.0, 2.0]
            self.module_deltas = {"fast": [1.0], "slow": [2.0]}
            self.timings = {"fast": 0.1, "slow": 0.3}

    tracker = StubTracker()

    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=StubTracker)
    )
    from menace_sandbox.roi_results_db import ROIResultsDB
    from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer

    def fake_run_workflow_simulations(**kwargs):
        details = {
            "group": [
                {"module": "fast", "result": {"exit_code": 0}},
                {"module": "slow", "result": {"exit_code": 0}},
            ]
        }
        return tracker, details

    monkeypatch.setattr(
        "menace_sandbox.sandbox_runner.environment.run_workflow_simulations",
        fake_run_workflow_simulations,
    )

    db_path = tmp_path / "roi.db"
    results_db = ROIResultsDB(db_path)
    scorer = CompositeWorkflowScorer(tracker=tracker, results_db=results_db)

    result = scorer.evaluate("wf1")
    assert result.success_rate == 1.0
    assert result.roi_gain == pytest.approx(3.0)

    cur = results_db.conn.cursor()
    cur.execute("SELECT workflow_id, module_deltas FROM roi_results")
    wf, deltas_json = cur.fetchone()
    assert wf == "wf1"
    deltas = json.loads(deltas_json)
    assert deltas["fast"]["roi_delta"] == pytest.approx(1.0)
    assert deltas["slow"]["roi_delta"] == pytest.approx(2.0)

