import os
import random
import sys
import types

import networkx as nx

from meta_workflow_planner import MetaWorkflowPlanner
from roi_results_db import ROIResultsDB


class DummyGraph:
    def __init__(self, g: nx.DiGraph) -> None:
        self.graph = g


def test_small_chain_roi_improvement(tmp_path):
    db = ROIResultsDB(path=tmp_path / "roi.db")
    db.log_result(
        workflow_id="A",
        run_id="1",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=-1.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
    )
    db.log_result(
        workflow_id="A",
        run_id="2",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=3.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
    )
    db.log_result(
        workflow_id="B",
        run_id="1",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=2.0,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
    )

    g = nx.DiGraph()
    g.add_edge("A", "B")
    planner = MetaWorkflowPlanner(graph=DummyGraph(g), roi_db=db, roi_window=5)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    vec_a = planner.encode("A", {"workflow": []})
    vec_b = planner.encode("B", {"workflow": []})
    os.chdir(old_cwd)

    roi_a = vec_a[2:7]
    assert roi_a[0] == -1.0 and roi_a[1] == 3.0
    assert vec_a[1] == 1.0  # branching
    roi_b = vec_b[2:7]
    assert roi_b[0] == 2.0
    assert vec_b[0] == 1.0  # depth
    assert roi_a[1] > roi_a[0]


def test_genetic_evolution_across_domains(monkeypatch):
    class StubComparator:
        @staticmethod
        def _entropy(spec):
            return 0.0

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )

    class ModuleMetric:
        def __init__(self, name, result):
            self.name = name
            self.result = result
            self.success = True
            self.duration = 0.0

    class Metrics:
        def __init__(self, modules):
            self.modules = modules
            self.crash_count = 0

    class DummyRunner:
        def run(self, funcs):
            modules = [ModuleMetric(fn.__name__, fn()) for fn in funcs]
            return Metrics(modules)

    def f1():
        return 1.0

    def f2():
        return 1.0

    def g1():
        return 1.0

    def g2():
        return 1.0

    workflows = {"f1": f1, "f2": f2, "g1": g1, "g2": g2}

    planner = MetaWorkflowPlanner()

    def _entry(score: float) -> dict:
        return {
            "roi_history": [1.0],
            "failure_history": [0],
            "entropy_history": [0.0],
            "step_metrics": [],
            "step_deltas": [],
            "delta_roi": 0.0,
            "delta_failures": 0.0,
            "delta_entropy": 0.0,
            "converged": True,
            "score": score,
        }

    planner.cluster_map = {("f1", "f2"): _entry(5.0), ("g1", "g2"): _entry(1.0)}

    monkeypatch.setattr(random, "choices", lambda pop, weights, k: [pop[0], pop[1]])
    monkeypatch.setattr(random, "randint", lambda a, b: 1)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    offspring = planner.mutate_chains(
        [["f1", "f2"], ["g1", "g2"]],
        workflows,
        runner=DummyRunner(),
    )
    assert offspring
    chains = [rec["chain"] for rec in offspring]
    assert any("f1" in c and "g2" in c for c in chains)

    planner.cluster_map[("f1", "f2")].update(
        delta_roi=-1.0, delta_failures=0.2, delta_entropy=0.1
    )
    planner.cluster_map[("g1", "g2")].update(
        delta_roi=2.0, delta_failures=-0.1, delta_entropy=-0.2
    )

    def fake_mutate(chains, workflows, **_):
        return [
            {
                "chain": list(chains[0]) + ["x"],
                "roi_gain": 1.0,
                "failures": 0,
                "entropy": 0.0,
            }
        ]

    monkeypatch.setattr(planner, "mutate_chains", fake_mutate)
    records = planner.iterate_pipelines(workflows)
    assert any("f1" in r["chain"] for r in records)
    assert all("g1" not in r["chain"] for r in records)
