# Tests for CompositeWorkflowScorer utilities.

import sys
import types
from collections import defaultdict
from unittest import mock

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Lightweight stubs for heavy dependencies
# ---------------------------------------------------------------------------


class _StubTracker:
    """Minimal ROITracker replacement used during import."""

    def __init__(self):
        self.roi_history = []
        self.module_deltas = defaultdict(list)
        self.timings = {}
        self.scheduling_overhead = {}

    def update(self, _before, roi_after, *, modules=None, **_kw):
        self.roi_history.append(roi_after)
        if modules:
            for m in modules:
                self.module_deltas[m].append(roi_after)


class _StubResultsDB:
    """Simple ROIResultsDB stand‑in collecting log calls."""

    def __init__(self, *a, **k):
        self.log_result = mock.Mock()
        self.log_module_attribution = mock.Mock()


# Register stubs before importing the scorer to avoid heavy imports.
sys.modules.setdefault(
    "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=_StubTracker)
)
sys.modules.setdefault("roi_tracker", sys.modules["menace_sandbox.roi_tracker"])
sys.modules.setdefault(
    "menace_sandbox.roi_results_db", types.SimpleNamespace(ROIResultsDB=_StubResultsDB)
)
sys.modules.setdefault("roi_results_db", sys.modules["menace_sandbox.roi_results_db"])
sys.modules.setdefault(
    "menace_sandbox.sandbox_runner",
    types.SimpleNamespace(environment=types.SimpleNamespace()),
)
sys.modules.setdefault("sandbox_runner", sys.modules["menace_sandbox.sandbox_runner"])
class _StubPatchDB:
    def success_rate(self, limit: int = 50) -> float:
        return 1.0

sys.modules.setdefault(
    "menace_sandbox.code_database", types.SimpleNamespace(PatchHistoryDB=_StubPatchDB)
)
sys.modules.setdefault("code_database", sys.modules["menace_sandbox.code_database"])

from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer


# ---------------------------------------------------------------------------
# Tests exercising run, score_workflow and evaluate
# ---------------------------------------------------------------------------


def test_run_records_metrics_and_ids(monkeypatch):
    """``run`` persists metrics with workflow/run identifiers."""

    class Tracker:
        def __init__(self):
            self.roi_history = [1.0, 2.0]
            self.module_deltas = {"mod1": [0.5], "mod2": [1.5]}
            self.timings = {"mod1": 0.1, "mod2": 0.2}
            self.scheduling_overhead = {"mod1": 0.01, "mod2": 0.02}

    tracker = Tracker()
    results_db = _StubResultsDB()
    scorer = CompositeWorkflowScorer(tracker=tracker, results_db=results_db)
    scorer._roi_start = 0
    scorer._module_start = {"mod1": 0, "mod2": 0}
    scorer._module_successes = {"mod1": True, "mod2": False}

    import menace_sandbox.code_database as code_db_mod

    monkeypatch.setattr(
        code_db_mod.PatchHistoryDB,
        "success_rate",
        lambda self, limit=50: 0.25,
    )

    run_id = "run1"
    wf_id = "wf1"
    result = scorer.run(lambda: True, wf_id, run_id)

    kwargs = results_db.log_result.call_args.kwargs
    assert kwargs["workflow_id"] == wf_id
    assert kwargs["run_id"] == run_id
    assert set(kwargs["module_deltas"].keys()) == {"mod1", "mod2"}
    assert result.roi_gain == pytest.approx(3.0)
    assert results_db.log_module_attribution.call_count == 2
    expected_patch = (1.0 / np.std([1.0, 2.0])) * 0.25
    assert kwargs["patchability_score"] == pytest.approx(expected_patch)
    assert result.patchability_score == pytest.approx(expected_patch)


def test_score_workflow_persists_results_and_ids():
    """``score_workflow`` records metrics via ``log_result``."""

    tracker = _StubTracker()
    results_db = _StubResultsDB()
    calculator = types.SimpleNamespace(
        calculate=lambda metrics, _p: (sum(metrics.values()), False, []),
        profiles={"default": {}},
    )
    scorer = CompositeWorkflowScorer(
        tracker=tracker, calculator=calculator, results_db=results_db
    )

    def mod_a():
        return True

    def mod_b():
        return False

    run_id, data = scorer.score_workflow(
        "wf2", {"mod_a": mod_a, "mod_b": mod_b}, run_id="rid123"
    )

    assert run_id == "rid123"
    kwargs = results_db.log_result.call_args.kwargs
    assert kwargs["workflow_id"] == "wf2"
    assert kwargs["run_id"] == "rid123"
    assert set(kwargs["module_deltas"].keys()) == {"mod_a", "mod_b"}
    assert tracker.roi_history  # tracker.update was invoked


def test_evaluate_logs_run_and_workflow(monkeypatch):
    """``evaluate`` obtains metrics from sandbox simulations."""

    class Tracker:
        def __init__(self):
            self.roi_history = [0.5, 1.0]
            self.module_deltas = {"m1": [0.5], "m2": [1.0]}
            self.timings = {"m1": 0.1, "m2": 0.2}
            self.scheduling_overhead = {"m1": 0.01, "m2": 0.02}

    tracker = Tracker()
    results_db = _StubResultsDB()

    def fake_run_workflow_simulations(**_kw):
        details = {
            "group": [
                {"module": "m1", "result": {"exit_code": 0}},
                {"module": "m2", "result": {"exit_code": 1}},
            ]
        }
        return tracker, details

    import menace_sandbox.sandbox_runner as sandbox_runner

    monkeypatch.setattr(
        sandbox_runner,
        "environment",
        types.SimpleNamespace(run_workflow_simulations=fake_run_workflow_simulations),
    )

    scorer = CompositeWorkflowScorer(tracker=tracker, results_db=results_db)
    result = scorer.evaluate("wf3")

    kwargs = results_db.log_result.call_args.kwargs
    assert kwargs["workflow_id"] == "wf3"
    assert kwargs["run_id"]
    assert set(kwargs["module_deltas"].keys()) == {"m1", "m2"}
    assert result.success_rate == pytest.approx(0.5)
