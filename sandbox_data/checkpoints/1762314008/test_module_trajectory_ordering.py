import os
import sys
import types
from statistics import fmean, pvariance

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
# ensure lightweight imports by stubbing optional modules
sys.modules.pop("menace_sandbox.roi_tracker", None)

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
from menace_sandbox.roi_results_db import ROIResultsDB  # noqa: E402


class DummyTracker:
    """Minimal tracker recording module deltas for testing."""

    def __init__(self) -> None:
        self.metrics_history: dict[str, list[float]] = {}
        self.roi_history: list[float] = []
        self.module_deltas: dict[str, list[float]] = {}
        self.timings: dict[str, float] = {}
        self.scheduling_overhead: dict[str, float] = {}

    def update(self, roi_before, roi_after, modules=None, **_):  # noqa: D401
        if modules:
            delta = roi_after - roi_before
            for m in modules:
                self.module_deltas.setdefault(m, []).append(delta)
        return None, [], False, False

    def cache_correlations(self, pairs):
        pass


def test_module_trajectories_preserve_order(tmp_path, monkeypatch):
    tracker = DummyTracker()
    db_path = tmp_path / "roi.db"
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

    run_ids = ["r2", "r1", "r3"]
    for rid in run_ids:
        scorer.score_workflow("wf", {"alpha": alpha}, run_id=rid)

    traj = results_db.fetch_module_trajectories("wf", "alpha")["alpha"]

    deltas = [0.2, 0.4, -0.1]
    expected_ma = [fmean(deltas[: i + 1]) for i in range(len(deltas))]
    expected_var = [pvariance(deltas[: i + 1]) if i > 0 else 0.0 for i in range(len(deltas))]

    assert [t["run_id"] for t in traj] == run_ids
    assert [t["roi_delta"] for t in traj] == pytest.approx(deltas)
    assert [t["moving_avg"] for t in traj] == pytest.approx(expected_ma)
    assert [t["variance"] for t in traj] == pytest.approx(expected_var)
