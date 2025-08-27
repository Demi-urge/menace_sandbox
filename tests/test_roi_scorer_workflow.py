"""Tests for ROI scoring utilities using the composite scorer."""

import sys
import time
import types
import yaml
import pytest


# Stub heavy dependencies before importing the scorer module
sys.modules.setdefault(
    "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=object)
)
sys.modules.setdefault(
    "menace_sandbox.sandbox_runner",
    types.SimpleNamespace(
        WorkflowSandboxRunner=lambda: types.SimpleNamespace(
            run=lambda fn, **kw: fn()
        )
    ),
)

from menace_sandbox.db_router import init_db_router  # noqa: E402
from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer  # noqa: E402
from menace_sandbox.workflow_scorer_core import (  # noqa: E402
    compute_bottleneck_index,
    compute_patchability,
    compute_workflow_synergy,
)
from menace_sandbox.roi_calculator import ROICalculator  # noqa: E402
from menace_sandbox.roi_results_db import ROIResultsDB  # noqa: E402


def test_compute_metric_helpers():
    module_hist = {
        "a": [1.0, 2.0, 3.0],
        "b": [1.0, 2.0, 3.0],
        "c": [3.0, 2.0, 1.0],
    }

    class Tracker:
        def __init__(self, history=None):
            self.module_deltas = module_hist
            self.correlation_history = history or {}

        def cache_correlations(self, pairs):
            for k, v in pairs.items():
                self.correlation_history.setdefault(k, []).append(v)

    tracker = Tracker()
    baseline = compute_workflow_synergy(tracker, window=3)
    assert baseline == pytest.approx((-1.0) / 3.0)

    tracker.correlation_history = {
        ("a", "b"): [0.8, 0.9, 0.95],
        ("a", "c"): [0.1, -0.2, 0.2],
        ("b", "c"): [0.0, -0.3, 0.3],
    }
    weighted = compute_workflow_synergy(tracker, window=3)
    assert weighted > baseline

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
            self.correlation_history: dict[tuple[str, str], list[float]] = {}

        def cache_correlations(self, pairs):
            for k, v in pairs.items():
                self.correlation_history.setdefault(k, []).append(v)

        def update(self, roi_before, roi_after, modules=None, **_kwargs):
            self.roi_history.append(roi_after)
            for m in modules or []:
                self.module_deltas.setdefault(m, []).append(roi_after - roi_before)

    tracker = StubTracker()
    db_path = tmp_path / "roi_results.db"
    results_db = ROIResultsDB(db_path)
    profile = {
        "default": {
            "weights": {
                "profitability": 0.125,
                "efficiency": 0.125,
                "reliability": 0.125,
                "resilience": 0.125,
                "maintainability": 0.125,
                "security": 0.125,
                "latency": 0.125,
                "energy": 0.125,
            }
        }
    }
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    scorer = CompositeWorkflowScorer(
        tracker=tracker,
        calculator_factory=lambda: ROICalculator(profiles_path=profile_path),
        results_db=results_db,
    )

    modules = {"mod_a": lambda: True, "mod_b": lambda: True}

    counter = iter([0, 1, 3, 4, 5, 6])
    monkeypatch.setattr(time, "perf_counter", lambda: next(counter))

    run_id = "run1"
    rid, metrics = scorer.score_workflow("wf1", modules, run_id=run_id)
    assert rid == run_id

    assert metrics["roi_gain"] == pytest.approx(0.4375)
    assert metrics["workflow_synergy_score"] == pytest.approx(0.0)
    assert metrics["bottleneck_index"] == pytest.approx(1.0 / 6.0)
    expected_patch = compute_patchability(tracker.roi_history, scorer.history_window)
    assert metrics["patchability_score"] == pytest.approx(expected_patch)

    results = results_db.fetch_results("wf1", run_id)
    assert len(results) == 1
    row = results[0]
    assert row.workflow_id == "wf1" and row.run_id == run_id
    assert row.module_deltas["mod_a"]["roi_delta"] == pytest.approx(0.1875)
    assert row.module_deltas["mod_a"]["success_rate"] == pytest.approx(1.0)
    assert row.module_deltas["mod_b"]["roi_delta"] == pytest.approx(0.25)
    assert row.module_deltas["mod_b"]["success_rate"] == pytest.approx(1.0)

    attrib = results_db.fetch_module_attribution()
    assert attrib["mod_a"]["roi_delta"] == pytest.approx(0.1875)
    assert attrib["mod_a"]["bottleneck"] == pytest.approx(2.0 / 3.0)
    assert scorer.module_attribution["mod_b"]["bottleneck_contribution"] == pytest.approx(1.0 / 3.0)
