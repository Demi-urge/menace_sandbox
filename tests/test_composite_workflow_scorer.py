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
import sqlite3 as sqlite3_mod
from pathlib import Path
from typing import Dict

import pytest
from dynamic_path_router import resolve_path


class _StubDBRouter:
    def __init__(self, _name, local_path, _shared_path):
        self.path = local_path

    def get_connection(self, _name, operation: str | None = None):
        return sqlite3_mod.connect(self.path)


sys.modules.setdefault(
    "menace_sandbox.db_router",
    types.SimpleNamespace(
        init_db_router=lambda *a, **k: None,
        DBRouter=_StubDBRouter,
        LOCAL_TABLES=set(),
        GLOBAL_ROUTER=None,
    ),
)
sys.modules.setdefault("db_router", sys.modules["menace_sandbox.db_router"])
from menace_sandbox.db_router import init_db_router  # noqa: E402
from menace_sandbox.roi_calculator import ROIResult  # noqa: E402


class _StubPatchDB:
    def success_rate(self, limit: int = 50) -> float:
        return 1.0


sys.modules.setdefault(
    "menace_sandbox.code_database", types.SimpleNamespace(PatchHistoryDB=_StubPatchDB)
)
sys.modules[
    "menace_sandbox.sandbox_runner"
] = types.SimpleNamespace(
    environment=types.SimpleNamespace(),
    WorkflowSandboxRunner=lambda: types.SimpleNamespace(
        run=lambda fn, **kw: fn()
    ),
)


def _copy_fixture_modules(tmp_path: Path) -> None:
    """Copy minimal workflow modules into ``tmp_path``."""

    for name in ("mod_a.py", "mod_b.py", "mod_c.py"):  # path-ignore
        shutil.copy(
            resolve_path(f"tests/fixtures/workflow_modules/{name}"),
            tmp_path / name,
        )


def _stub_calculator_factory():
    return types.SimpleNamespace(
        calculate=lambda metrics, _p: ROIResult(
            sum(float(v) for v in metrics.values()),
            False,
            [],
        ),
        profiles={"default": {}},
    )


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
            self.correlation_history: dict[tuple[str, str], list[float]] = {}

        def cache_correlations(self, pairs):
            for k, v in pairs.items():
                self.correlation_history.setdefault(k, []).append(v)

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

    import menace_sandbox.code_database as code_db_mod

    monkeypatch.setattr(
        code_db_mod.PatchHistoryDB, "success_rate", lambda self, limit=50: 0.5
    )

    from menace_sandbox.roi_results_db import ROIResultsDB
    import menace_sandbox.composite_workflow_scorer as composite_mod
    monkeypatch.setattr(composite_mod, "PATCH_SUCCESS_RATE", 0.5)
    from menace_sandbox.composite_workflow_scorer import (
        CompositeWorkflowScorer,
        compute_bottleneck_index,
    )
    from menace_sandbox.workflow_scorer_core import compute_patchability
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
        tracker=tracker,
        results_db=ROIResultsDB(db_path),
        calculator_factory=_stub_calculator_factory,
    )
    workflow_id = "wf_example"
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    result = scorer.evaluate(workflow_id, context_builder=builder)

    # Basic sanity checks on aggregate metrics.
    assert result.runtime > 0
    assert 0.0 <= result.success_rate <= 1.0
    assert result.workflow_synergy_score == 0.0
    expected_bottleneck = compute_bottleneck_index(tracker.timings)
    assert result.bottleneck_index == pytest.approx(expected_bottleneck)
    expected_patch = compute_patchability(
        tracker.roi_history, patch_success=0.5, window=scorer.history_window
    )
    assert result.patchability_score == pytest.approx(expected_patch)

    # Ensure results persisted with workflow/run identifiers and per-module deltas.
    cur = scorer.results_db.conn.cursor()
    cur.execute("SELECT workflow_id, run_id, module_deltas FROM workflow_results")
    wf, run_id, deltas_json = cur.fetchone()
    assert wf == workflow_id
    assert run_id  # run identifier was generated
    deltas = json.loads(deltas_json)
    assert isinstance(deltas, dict)

    # Regression coverage for module attribution utilities.
    report = scorer.results_db.module_impact_report(workflow_id, run_id)
    assert "improved" in report and "regressed" in report

    # Module attribution records were stored and exposed
    attrib = scorer.results_db.fetch_module_attribution()
    assert attrib  # ensure attribution exists
    assert scorer.module_attribution  # attribution exposed on scorer


def test_patch_success_modulates_patchability(tmp_path, monkeypatch):
    """Global patch success rate scales patchability_score."""

    _copy_fixture_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    init_db_router(
        "test_patch_success",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    class StubTracker:
        def __init__(self) -> None:
            self.roi_history = [1.0, 2.0, 3.0]
            self.module_deltas = {"mod_a": [1.0], "mod_b": [2.0], "mod_c": [-0.5]}
            self.timings = {"mod_a": 0.1, "mod_b": 0.2, "mod_c": 0.3}
            self.correlation_history: dict[tuple[str, str], list[float]] = {}

        def cache_correlations(self, pairs):  # pragma: no cover - passthrough
            for k, v in pairs.items():
                self.correlation_history.setdefault(k, []).append(v)

    tracker = StubTracker()

    def fake_run_workflow_simulations(**_kwargs):
        details = {
            "group": [
                {"module": "mod_a", "result": {"exit_code": 0}},
                {"module": "mod_b", "result": {"exit_code": 1}},
                {"module": "mod_c", "result": {"exit_code": 0}},
            ]
        }
        return tracker, details

    import menace_sandbox.sandbox_runner as sandbox_runner
    sandbox_runner.environment = types.SimpleNamespace(
        run_workflow_simulations=fake_run_workflow_simulations
    )

    from menace_sandbox.roi_results_db import ROIResultsDB
    from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer

    db_path = tmp_path / "roi_results.db"
    scorer = CompositeWorkflowScorer(
        tracker=tracker,
        results_db=ROIResultsDB(db_path),
        calculator_factory=_stub_calculator_factory,
    )
    workflow_id = "wf_example"

    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    high = scorer.evaluate(
        workflow_id, patch_success=1.0, context_builder=builder
    ).patchability_score

    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    low = scorer.evaluate(
        workflow_id, patch_success=0.25, context_builder=builder
    ).patchability_score

    assert low == pytest.approx(high * 0.25)


def test_fetch_trends_returns_time_ordered_metrics(tmp_path, monkeypatch):
    """Recorded results expose aggregate trend metrics in order."""

    _copy_fixture_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    init_db_router(
        "test_trend_fetch",
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
            self.correlation_history: dict[tuple[str, str], list[float]] = {}

        def cache_correlations(self, pairs):
            for k, v in pairs.items():
                self.correlation_history.setdefault(k, []).append(v)

        def workflow_variance(self, workflow_id: str) -> float:  # pragma: no cover - trivial
            return 0.0

    tracker = StubTracker()
    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=StubTracker)
    )
    sys.modules.setdefault(
        "menace_sandbox.synergy_history_db",
        types.SimpleNamespace(load_history=lambda: [{"mod_a,mod_b": 0.5}]),
    )

    import menace_sandbox.code_database as code_db_mod

    monkeypatch.setattr(
        code_db_mod.PatchHistoryDB, "success_rate", lambda self, limit=50: 0.5
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

    sandbox_runner.environment = types.SimpleNamespace(
        run_workflow_simulations=fake_run_workflow_simulations
    )

    db_path = tmp_path / "roi_results.db"
    scorer = CompositeWorkflowScorer(
        tracker=tracker,
        results_db=ROIResultsDB(db_path),
        calculator_factory=_stub_calculator_factory,
    )
    workflow_id = "wf_trend"
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    first = scorer.evaluate(workflow_id, context_builder=builder)

    # adjust tracker for a second run with different ROI gain
    tracker.roi_history = [2.0, 3.0, 4.0]
    tracker.module_deltas = {"mod_a": [2.0], "mod_b": [1.0], "mod_c": [0.0]}
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    second = scorer.evaluate(workflow_id, context_builder=builder)

    trends = scorer.results_db.fetch_trends(workflow_id)
    assert [t["roi_gain"] for t in trends] == [first.roi_gain, second.roi_gain]
    assert [t["bottleneck_index"] for t in trends] == [
        first.bottleneck_index,
        second.bottleneck_index,
    ]
    assert trends[0]["timestamp"] <= trends[1]["timestamp"]


def test_score_workflow_parallel_execution(tmp_path):
    """Modules run concurrently when concurrency hints are provided."""

    class StubTracker:
        def __init__(self) -> None:
            from collections import defaultdict

            self.roi_history: list[float] = []
            self.module_deltas: Dict[str, list[float]] = defaultdict(list)
            self.timings: Dict[str, float] = {}
            self.scheduling_overhead: Dict[str, float] = {}
            self.correlation_history: dict[tuple[str, str], list[float]] = {}

        def update(self, _before, roi_after, *, modules=None, **_kw):
            self.roi_history.append(roi_after)
            if modules:
                for m in modules:
                    self.module_deltas[m].append(roi_after)

        def cache_correlations(self, pairs):
            for k, v in pairs.items():
                self.correlation_history.setdefault(k, []).append(v)

    tracker = StubTracker()
    sys.modules.setdefault(
        "menace_sandbox.roi_tracker", types.SimpleNamespace(ROITracker=StubTracker)
    )
    from menace_sandbox.roi_results_db import ROIResultsDB
    from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer

    scorer = CompositeWorkflowScorer(
        tracker=tracker,
        calculator_factory=_stub_calculator_factory,
        results_db=ROIResultsDB(tmp_path / "db.sqlite"),
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
        "menace_sandbox.sandbox_runner",
        types.SimpleNamespace(
            WorkflowSandboxRunner=lambda: types.SimpleNamespace(
                run=lambda fn, **kw: fn()
            )
        ),
    )
    from menace_sandbox.workflow_scorer_core import compute_workflow_synergy

    module_hist = {
        "mod_a": [1.0, 2.0, 3.0],
        "mod_b": [1.0, 2.0, 3.0],
        "mod_c": [3.0, 2.0, 1.0],
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
        ("mod_a", "mod_b"): [0.8, 0.9, 0.95],
        ("mod_a", "mod_c"): [0.1, -0.2, 0.2],
        ("mod_b", "mod_c"): [0.0, -0.3, 0.3],
    }
    weighted = compute_workflow_synergy(tracker, window=3)
    assert weighted > baseline
