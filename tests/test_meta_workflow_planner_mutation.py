import sys
import types
import random

import pytest

import meta_workflow_planner as mwp
from meta_workflow_planner import MetaWorkflowPlanner


class ModuleMetric:
    def __init__(self, name: str, result: float, success: bool = True) -> None:
        self.name = name
        self.result = result
        self.success = success
        self.duration = 0.0


class Metrics:
    def __init__(self, modules) -> None:
        self.modules = modules
        self.crash_count = 0


class DummyRunner:
    def run(self, funcs):
        modules = [ModuleMetric(fn.__name__, fn()) for fn in funcs]
        return Metrics(modules)


def build_planner(monkeypatch: pytest.MonkeyPatch, entropy_value: float) -> MetaWorkflowPlanner:
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "_load_embeddings", lambda: {})
    monkeypatch.setattr(mwp, "persist_embedding", lambda *a, **k: None)

    class DummyROI:
        def __init__(self) -> None:
            self.logged: list[dict[str, object]] = []

        def log_result(self, **kwargs):
            self.logged.append(kwargs)

    planner = MetaWorkflowPlanner(roi_db=DummyROI())
    planner.cluster_map = {}

    class StubComparator:
        @staticmethod
        def _entropy(_spec):
            return entropy_value

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )
    return planner


def test_mutate_chains_generates_better_offspring(monkeypatch: pytest.MonkeyPatch):
    planner = build_planner(monkeypatch, entropy_value=0.0)
    runner = DummyRunner()

    def low1():
        return 1.0

    def low2():
        return 2.0

    def high():
        return 100.0

    workflows = {"low1": low1, "low2": low2, "high": high}

    parent1 = planner._validate_chain(["low1"], workflows, runner=runner, runs=1)
    parent2 = planner._validate_chain(["low2"], workflows, runner=runner, runs=1)
    parent_roi = max(parent1["roi_gain"], parent2["roi_gain"])

    monkeypatch.setattr(random, "choices", lambda pop, weights, k: [pop[0], pop[1]])
    monkeypatch.setattr(random, "randint", lambda a, b: 1)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randrange", lambda n: 0)
    monkeypatch.setattr(random, "choice", lambda seq: seq[-1])

    offspring = planner.mutate_chains(
        [["low1"], ["low2"]],
        workflows,
        runner=runner,
        runs=1,
    )
    assert offspring
    assert max(rec["roi_gain"] for rec in offspring) > parent_roi


def test_validate_chain_rejects_high_entropy(monkeypatch: pytest.MonkeyPatch):
    planner = build_planner(monkeypatch, entropy_value=5.0)
    runner = DummyRunner()

    def wf():
        return 1.0

    workflows = {"a": wf}

    record = planner._validate_chain(
        ["a"], workflows, runner=runner, entropy_threshold=1.0, runs=1
    )
    assert record is None
