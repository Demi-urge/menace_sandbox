import sys
import types
from threading import Lock

import pytest
import meta_workflow_planner as mwp
from meta_workflow_planner import MetaWorkflowPlanner
from roi_results_db import ROIResultsDB
from workflow_stability_db import WorkflowStabilityDB


class ModuleMetric:
    def __init__(self, name: str, success: bool = True) -> None:
        self.name = name
        self.result = 1.0
        self.success = success
        self.duration = 0.0


class Metrics:
    def __init__(self, modules, crash_count: int = 0) -> None:
        self.modules = modules
        self.crash_count = crash_count


class DummyRunner:
    def __init__(self, failures, rois=None) -> None:
        self.failures = list(failures)
        self.rois = list(rois) if rois is not None else [1.0] * len(self.failures)
        self.calls = 0
        self._lock = Lock()

    def run(self, funcs):
        with self._lock:
            idx = self.calls
            self.calls += 1
        fail = self.failures[idx] if idx < len(self.failures) else 0
        roi = self.rois[idx] if idx < len(self.rois) else 1.0
        modules = []
        for i, fn in enumerate(funcs):
            success = i >= fail
            m = ModuleMetric(fn.__name__, success=success)
            m.result = roi if i == 0 else 0.0
            modules.append(m)
        return Metrics(modules)


@pytest.fixture
def planner(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    class StubComparator:
        @staticmethod
        def _entropy(spec):  # noqa: D401
            """Deterministic entropy for tests."""
            return 0.0

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )
    planner = MetaWorkflowPlanner()
    planner.cluster_map = {}
    return planner


def wf() -> float:
    return 1.0


def test_mutate_pipeline_triggers_on_rising_failures(planner):
    workflows = {"a": wf, "b": wf, "c": wf}
    pipeline = ["a", "b"]
    planner._update_cluster_map(
        pipeline,
        roi_gain=2.0,
        failures=0,
        entropy=0.0,
        step_metrics=[
            {"module": m, "roi": 1.0, "failures": 0, "entropy": 0.0}
            for m in pipeline
        ],
    )
    runner = DummyRunner([1])
    results = planner.mutate_pipeline(
        pipeline, workflows, runner=runner, failure_threshold=10
    )
    assert results and any(r["chain"] != pipeline for r in results)


def test_manage_pipeline_splits_on_rising_failures(planner):
    workflows = {"a": wf, "b": wf}
    pipeline = ["a", "b"]
    planner._update_cluster_map(
        pipeline,
        roi_gain=2.0,
        failures=0,
        entropy=0.0,
        step_metrics=[
            {"module": m, "roi": 1.0, "failures": 0, "entropy": 0.0}
            for m in pipeline
        ],
    )
    runner = DummyRunner([1, 0, 0, 0])
    results = planner.manage_pipeline(
        pipeline, workflows, runner=runner, failure_threshold=10
    )
    assert results and all(len(r["chain"]) == 1 for r in results)


def test_validate_chain_multi_run_aggregation(tmp_path, monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)

    class StubComparator:
        @staticmethod
        def _entropy(spec):  # noqa: D401
            """Deterministic entropy for tests."""
            return 0.0

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )

    roi_db = ROIResultsDB(path=tmp_path / "roi.db")
    stability_db = WorkflowStabilityDB(path=tmp_path / "stable.json")
    planner = MetaWorkflowPlanner(roi_db=roi_db, stability_db=stability_db)
    planner.cluster_map = {}

    workflows = {"a": wf, "b": wf}
    runner = DummyRunner([0, 1, 0], rois=[1.0, 2.0, 3.0])
    record = planner._validate_chain(
        ["a", "b"],
        workflows,
        runner=runner,
        runs=3,
        failure_threshold=10,
        max_workers=3,
    )
    assert record is not None
    assert record["roi_gain"] == pytest.approx(2.0)
    assert record["roi_var"] == pytest.approx(2 / 3)
    assert record["failures"] == pytest.approx(1 / 6)
    assert record["failures_var"] == pytest.approx(1 / 18)

    chain_id = "a->b"
    db_rec = roi_db.fetch_results(chain_id)[0]
    agg = db_rec.module_deltas["__aggregate__"]
    assert agg["roi_gain_var"] == pytest.approx(2 / 3)
    assert agg["failures_mean"] == pytest.approx(1 / 6)
    stable = stability_db.data[chain_id]
    assert stable["roi"] == pytest.approx(2.0)
    assert stable["roi_var"] == pytest.approx(2 / 3)
    assert stable["failures"] == pytest.approx(1 / 6)
    assert stable["failures_var"] == pytest.approx(1 / 18)
