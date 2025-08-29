import sys
import types

import pytest
import meta_workflow_planner as mwp
from meta_workflow_planner import MetaWorkflowPlanner


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
    def __init__(self, failures) -> None:
        self.failures = list(failures)
        self.calls = 0

    def run(self, funcs):
        fail = self.failures[self.calls] if self.calls < len(self.failures) else 0
        self.calls += 1
        modules = []
        for i, fn in enumerate(funcs):
            success = i >= fail
            modules.append(ModuleMetric(fn.__name__, success=success))
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
    assert results and all(r["chain"] != pipeline for r in results)


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
