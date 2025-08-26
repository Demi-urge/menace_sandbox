"""Tests for :class:`CompositeWorkflowScorer`.

This suite exercises the lightweight evaluation pipeline using stubbed
workflow runs.  Modules from ``tests/fixtures/workflow_modules`` are copied
into a temporary directory to mimic a minimal workflow.  The scorer is run
against these modules and writes aggregated metrics to a temporary SQLite
database.  Stored per-module deltas provide regression coverage for module
attribution logic.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from menace_sandbox.db_router import init_db_router


FIXTURES = Path(__file__).parent / "fixtures" / "workflow_modules"


def _copy_fixture_modules(tmp_path: Path) -> None:
    """Copy minimal workflow modules into ``tmp_path``."""

    for name in ("mod_a.py", "mod_b.py", "mod_c.py"):
        shutil.copy(FIXTURES / name, tmp_path / name)


def test_composite_workflow_scorer_records_metrics(tmp_path, monkeypatch):
    """Composite scorer emits metrics and persists results."""

    _copy_fixture_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    # Isolate database interactions.
    init_db_router(
        "test_composite_scorer",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    class StubTracker:
        def __init__(self) -> None:
            self.roi_history = [1.0, 2.0, 3.0]
            self.module_deltas = {
                "mod_a": [1.0],
                "mod_b": [2.0],
                "mod_c": [-0.5],
            }
            self.timings = {"mod_a": 0.1, "mod_b": 0.2, "mod_c": 0.3}

        def workflow_variance(self, workflow_id: str) -> float:  # pragma: no cover - trivial
            return 0.0

    tracker = StubTracker()

    # Avoid importing the heavy ``roi_tracker`` module.
    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=StubTracker)
    )

    # Provide a tiny synergy history to exercise synergy scoring.
    sys.modules.setdefault(
        "menace_sandbox.synergy_history_db",
        types.SimpleNamespace(load_history=lambda: [{"mod_a,mod_b": 0.5}]),
    )

    from menace_sandbox.roi_results_db import ROIResultsDB
    from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer
    import menace_sandbox.sandbox_runner as sandbox_runner

    def fake_run_workflow_simulations(**_kwargs):
        details = {
            "group": [
                {"module": "mod_a", "result": {"exit_code": 0}},
                {"module": "mod_b", "result": {"exit_code": 1}},
                {"module": "mod_c", "result": {"exit_code": 0}},
            ]
        }
        return tracker, details

    # Patch the environment runner to avoid loading heavy dependencies.
    sandbox_runner.environment = types.SimpleNamespace(
        run_workflow_simulations=fake_run_workflow_simulations
    )

    db_path = tmp_path / "roi_results.db"
    scorer = CompositeWorkflowScorer(
        tracker=tracker, results_db=ROIResultsDB(db_path)
    )
    workflow_id = "wf_example"
    result = scorer.evaluate(workflow_id)

    # Basic sanity checks on aggregate metrics.
    assert result.runtime > 0
    assert 0.0 <= result.success_rate <= 1.0
    assert result.workflow_synergy_score == 0.0
    assert result.bottleneck_index == pytest.approx(0.13608276348795434)
    expected_patch = 1.0 / np.std([1.0, 2.0, 3.0])
    assert result.patchability_score == pytest.approx(expected_patch)

    # Ensure results persisted with workflow/run identifiers and per-module deltas.
    cur = scorer.results_db.conn.cursor()
    cur.execute("SELECT workflow_id, run_id, module_deltas FROM workflow_results")
    wf, run_id, deltas_json = cur.fetchone()
    assert wf == workflow_id
    assert run_id  # run identifier was generated
    deltas = json.loads(deltas_json)
    assert deltas["mod_a"]["roi_delta"] == pytest.approx(1.0)
    assert deltas["mod_b"]["roi_delta"] == pytest.approx(2.0)
    assert deltas["mod_c"]["roi_delta"] == pytest.approx(-0.5)

    # Regression coverage for module attribution utilities.
    report = scorer.results_db.module_impact_report(workflow_id, run_id)
    assert report["improved"]["mod_a"] == pytest.approx(1.0)
    assert report["regressed"]["mod_c"] == pytest.approx(-0.5)

    # Module attribution records were stored and exposed
    attrib = scorer.results_db.fetch_module_attribution()
    total = 0.1 + 0.2 + 0.3
    assert attrib["mod_a"]["roi_delta"] == pytest.approx(1.0)
    assert attrib["mod_a"]["bottleneck"] == pytest.approx(0.1 / total)
    assert scorer.module_attribution["mod_c"]["bottleneck_contribution"] == pytest.approx(0.3 / total)


def test_score_workflow_parallel_execution(tmp_path):
    """Modules run concurrently when concurrency hints are provided."""

    class StubTracker:
        def __init__(self) -> None:
            from collections import defaultdict

            self.roi_history: list[float] = []
            self.module_deltas: Dict[str, list[float]] = defaultdict(list)
            self.timings: Dict[str, float] = {}
            self.scheduling_overhead: Dict[str, float] = {}

        def update(self, _before, roi_after, *, modules=None, **_kw):
            self.roi_history.append(roi_after)
            if modules:
                for m in modules:
                    self.module_deltas[m].append(roi_after)

    calculator = types.SimpleNamespace(
        calculate=lambda metrics, _p: (sum(metrics.values()), False, []),
        profiles={"default": {}},
    )

    tracker = StubTracker()
    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=StubTracker)
    )
    from menace_sandbox.roi_results_db import ROIResultsDB
    from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer

    scorer = CompositeWorkflowScorer(
        tracker=tracker, calculator=calculator, results_db=ROIResultsDB(tmp_path / "db.sqlite")
    )

    def mod_a():
        time.sleep(0.2)
        return True

    def mod_b():
        time.sleep(0.2)
        return True

    run_id, result = scorer.score_workflow(
        "wf_parallel",
        {"mod_a": mod_a, "mod_b": mod_b},
        concurrency_hints={"max_workers": 2},
    )

    assert result["runtime"] < 0.35
    assert tracker.scheduling_overhead["mod_a"] >= 0.0
    assert result["per_module"]["mod_a"]["scheduling_overhead"] >= 0.0


def test_compute_workflow_synergy_history_weighting():
    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=object)
    )
    sys.modules.setdefault(
        "menace_sandbox.roi_calculator", types.SimpleNamespace(ROICalculator=object)
    )
    sys.modules.setdefault(
        "menace_sandbox.sandbox_runner", types.SimpleNamespace()
    )
    from menace_sandbox.composite_workflow_scorer import compute_workflow_synergy

    roi_hist = [1.0, 2.0, 3.0]
    module_hist = {
        "mod_a": [1.0, 2.0, 3.0],
        "mod_b": [1.0, 2.0, 3.0],
        "mod_c": [3.0, 2.0, 1.0],
    }
    baseline = compute_workflow_synergy(roi_hist, module_hist, window=3, history_loader=lambda: [])
    assert baseline == pytest.approx((1.0 + 1.0 - 1.0) / 3.0)

    loader = lambda: [{"mod_a|mod_b": 1.0}]
    weighted = compute_workflow_synergy(roi_hist, module_hist, window=3, history_loader=loader)
    assert weighted == pytest.approx(1.0)

