import sys
import types
import time

import pytest

# Stub heavy dependencies before importing the scorer module
sys.modules.setdefault(
    "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=object)
)
sys.modules.setdefault(
    "menace_sandbox.roi_calculator", types.SimpleNamespace(ROICalculator=object)
)
sys.modules.setdefault(
    "menace_sandbox.sandbox_runner", types.SimpleNamespace()
)

from menace_sandbox.db_router import init_db_router
from menace_sandbox.composite_workflow_scorer import (
    CompositeWorkflowScorer,
    compute_bottleneck_index,
    compute_patchability,
    compute_workflow_synergy,
)
from menace_sandbox.roi_results_db import ROIResultsDB


def test_compute_metric_helpers():
    roi_hist = [1.0, 2.0, 3.0]
    module_hist = {"a": [0.5, 1.0, 1.5], "b": [0.5, 1.0, 1.5]}
    assert compute_workflow_synergy(roi_hist, module_hist, window=3) == pytest.approx(1.0)

    tracker = types.SimpleNamespace(timings={"a": 2.0, "b": 1.0})
    assert compute_bottleneck_index(tracker) == pytest.approx(1.0 / 6.0)

    history = [1.0, 2.0, 4.0]
    expected_patch = compute_patchability(history, window=3)
    assert expected_patch == pytest.approx(1.2026755886059102)


def test_composite_scorer_end_to_end(tmp_path, monkeypatch):
    init_db_router(
        "test_roi_scorer",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    class StubTracker:
        def __init__(self) -> None:
            self.roi_history = [1.0, 2.0, 3.0]
            self.module_deltas: dict[str, list[float]] = {}
            self.timings: dict[str, float] = {}

        def update(self, roi_before, roi_after, modules=None, **_kwargs):
            self.roi_history.append(roi_after)
            for m in modules or []:
                self.module_deltas.setdefault(m, []).append(roi_after - roi_before)

    class StubCalc:
        profiles = {"default": {}}

        def calculate(self, metrics, _profile):
            return float(sum(metrics.values())), False, []

    tracker = StubTracker()
    calc = StubCalc()
    db_path = tmp_path / "roi_results.db"
    results_db = ROIResultsDB(db_path)
    scorer = CompositeWorkflowScorer(tracker=tracker, calculator=calc, results_db=results_db)

    modules = {"mod_a": lambda: True, "mod_b": lambda: True}

    counter = iter([0, 1, 3, 4, 5, 6])
    monkeypatch.setattr(time, "perf_counter", lambda: next(counter))

    run_id = "run1"
    rid, metrics = scorer.score_workflow("wf1", modules, run_id=run_id)
    assert rid == run_id

    assert metrics["roi_gain"] == pytest.approx(3.5)
    assert metrics["workflow_synergy_score"] == pytest.approx(0.0)
    assert metrics["bottleneck_index"] == pytest.approx(1.0 / 6.0)
    expected_patch = compute_patchability(tracker.roi_history, scorer.history_window)
    assert metrics["patchability_score"] == pytest.approx(expected_patch)

    results = results_db.fetch_results("wf1", run_id)
    assert len(results) == 1
    row = results[0]
    assert row.workflow_id == "wf1" and row.run_id == run_id
    assert row.module_deltas["mod_a"]["roi_delta"] == pytest.approx(1.5)
    assert row.module_deltas["mod_b"]["roi_delta"] == pytest.approx(2.0)

    attrib = results_db.fetch_module_attribution()
    assert attrib["mod_a"]["roi_delta"] == pytest.approx(1.5)
    assert attrib["mod_a"]["bottleneck"] == pytest.approx(2.0 / 3.0)
    assert scorer.module_attribution["mod_b"]["bottleneck_contribution"] == pytest.approx(1.0 / 3.0)
