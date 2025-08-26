import os
import sys
import types
import pytest
import json
from statistics import fmean, pvariance
from menace_sandbox.roi_results_db import ROIResultsDB

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.pop("menace_sandbox.roi_tracker", None)

# Stub out optional heavy dependencies
sys.modules.setdefault("menace_sandbox.self_test_service", types.ModuleType("self_test_service"))
sys.modules.setdefault("menace_sandbox.error_bot", types.ModuleType("error_bot"))
sys.modules.setdefault(
    "menace_sandbox.menace_memory_manager", types.ModuleType("menace_memory_manager")
)
sys.modules.setdefault(
    "menace_sandbox.chatgpt_enhancement_bot", types.ModuleType("chatgpt_enhancement_bot")
)
sys.modules.setdefault(
    "menace_sandbox.chatgpt_idea_bot", types.ModuleType("chatgpt_idea_bot")
)
run_auto = types.ModuleType("run_autonomous")
run_auto._verify_required_dependencies = lambda: None
sys.modules.setdefault("menace_sandbox.run_autonomous", run_auto)

import menace_sandbox.composite_workflow_scorer as rs  # noqa: E402


class DummyTracker:
    def __init__(self) -> None:
        self.metrics_history: dict[str, list[float]] = {}
        self.roi_history: list[float] = []
        self.module_deltas: dict[str, list[float]] = {}
        self.timings: dict[str, float] = {}
        self.scheduling_overhead: dict[str, float] = {}

    def update(self, roi_before, roi_after, modules=None, **_):
        if modules:
            delta = roi_after - roi_before
            for m in modules:
                self.module_deltas.setdefault(m, []).append(delta)
        return None, [], False, False

    def cache_correlations(self, pairs):
        pass


def test_module_deltas_persistence(tmp_path, monkeypatch):
    tracker = DummyTracker()
    db_path = tmp_path / "roi_results.db"
    results_db = ROIResultsDB(db_path)
    scorer = rs.CompositeWorkflowScorer(tracker=tracker, results_db=results_db)

    vals = iter([0.3, -0.1])
    monkeypatch.setattr(
        scorer.calculator,
        "calculate",
        lambda metrics, profile_type: (next(vals), None, None),
    )

    def alpha() -> bool:
        return True

    def beta() -> bool:
        return True

    run_id, _ = scorer.score_workflow("wf", {"alpha": alpha, "beta": beta})
    assert scorer.module_attribution["alpha"]["roi_delta"] == pytest.approx(0.3)
    assert scorer.module_attribution["beta"]["roi_delta"] == pytest.approx(-0.1)

    cur = results_db.conn.cursor()
    cur.execute(
        "SELECT module_deltas FROM workflow_results WHERE workflow_id=? AND run_id=?",
        ("wf", run_id),
    )
    deltas = json.loads(cur.fetchone()[0])
    assert deltas["alpha"]["roi_delta"] == pytest.approx(0.3)
    assert deltas["beta"]["roi_delta"] == pytest.approx(-0.1)

    tracker2 = DummyTracker()
    for mod, info in deltas.items():
        tracker2.module_deltas.setdefault(mod, []).append(info["roi_delta"])
    assert tracker2.module_deltas == {"alpha": [0.3], "beta": [-0.1]}


def test_fetch_module_trajectories_after_multiple_runs(tmp_path, monkeypatch):
    tracker = DummyTracker()
    db_path = tmp_path / "roi_results.db"
    results_db = ROIResultsDB(db_path)
    scorer = rs.CompositeWorkflowScorer(tracker=tracker, results_db=results_db)

    vals = iter([0.2, 0.4, -0.1])
    monkeypatch.setattr(
        scorer.calculator,
        "calculate",
        lambda metrics, profile_type: (next(vals), None, None),
    )

    def alpha() -> bool:
        return True

    for i in range(3):
        scorer.score_workflow("wf", {"alpha": alpha}, run_id=f"r{i}")

    traj = results_db.fetch_module_trajectories("wf", "alpha")["alpha"]
    deltas = [0.2, 0.4, -0.1]
    expected_ma = [fmean(deltas[: i + 1]) for i in range(len(deltas))]
    expected_var = [pvariance(deltas[: i + 1]) if i > 0 else 0.0 for i in range(len(deltas))]

    assert [t["roi_delta"] for t in traj] == pytest.approx(deltas)
    assert [t["moving_avg"] for t in traj] == pytest.approx(expected_ma)
    assert [t["variance"] for t in traj] == pytest.approx(expected_var)
