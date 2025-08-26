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
from pathlib import Path

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
    assert result.workflow_synergy_score == pytest.approx((1.0 + 2.0 - 0.5) / 6.0)
    assert result.bottleneck_index == pytest.approx(0.3 / (0.1 + 0.2 + 0.3))
    expected_patch = 1.0 / (1.0 + np.std([1.0, 2.0, 3.0]))
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

