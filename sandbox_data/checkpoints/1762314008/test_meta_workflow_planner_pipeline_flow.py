import sys
import types

import pytest

import meta_workflow_planner as mwp


def test_pipeline_mutation_and_remerge(monkeypatch):
    """End-to-end exercise of planning, mutating and remerging pipelines."""

    # Disable external persistence / trackers
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "persist_embedding", lambda *a, **k: None)
    monkeypatch.setattr(mwp, "_load_embeddings", lambda *a, **k: {})
    monkeypatch.setattr(mwp, "shd", types.SimpleNamespace(record=lambda *a, **k: None))

    class DummyROIResultsDB:
        def __init__(self, *a, **k):
            pass

        def log_result(self, *a, **k):
            pass

        def fetch_results(self, *a, **k):
            return []

    monkeypatch.setattr(mwp, "ROIResultsDB", DummyROIResultsDB)

    class DummySuggester:
        def suggest_chains(self, target_embedding, top_k=3):
            return [["a", "b"]]

    monkeypatch.setitem(
        sys.modules,
        "workflow_chain_suggester",
        types.SimpleNamespace(WorkflowChainSuggester=DummySuggester),
    )

    class EntropyLow:
        @staticmethod
        def _entropy(spec):
            return 0.1

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=EntropyLow),
    )
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", EntropyLow)

    class ModuleMetric:
        def __init__(self, name, result, success=True):
            self.name = name
            self.result = result
            self.success = success
            self.duration = 0.0

    class Metrics:
        def __init__(self, modules):
            self.modules = modules
            self.crash_count = 0

    class DummyRunner:
        def run(self, funcs):
            modules = [ModuleMetric(fn.__name__, fn()) for fn in funcs]
            return Metrics(modules)

    def wf_a():
        return 1.0

    def wf_b():
        return 0.5

    def wf_c():
        return 2.0

    workflows = {"a": wf_a, "b": wf_b, "c": wf_c}

    planner = mwp.MetaWorkflowPlanner()
    planner.cluster_map = {}
    monkeypatch.setattr(planner, "_save_cluster_map", lambda: None)

    # 1) Suggest and validate baseline chain
    records = planner.plan_and_validate([0.0], workflows, runner=DummyRunner(), top_k=1, runs=1)
    assert records and records[0]["chain"] == ["a", "b"]
    baseline_roi = records[0]["roi_gain"]
    info_ab = planner.cluster_map[("a", "b")]
    assert baseline_roi == pytest.approx(1.5)
    assert info_ab["entropy_history"][-1] == pytest.approx(0.1)

    # 2) Mutate pipeline after entropy drift
    class EntropyHigh:
        @staticmethod
        def _entropy(spec):
            return 0.5

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=EntropyHigh),
    )
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", EntropyHigh)

    def fake_mutate(chains, workflows, **kwargs):
        new_chain = list(chains[0]) + ["c"]
        step_metrics = [
            {"module": "a", "roi": 1.0, "failures": 0, "entropy": 0.1},
            {"module": "b", "roi": 0.5, "failures": 0, "entropy": 0.2},
            {"module": "c", "roi": 2.0, "failures": 0, "entropy": 0.3},
        ]
        planner._update_cluster_map(
            new_chain,
            roi_gain=3.5,
            failures=0.0,
            entropy=0.3,
            step_metrics=step_metrics,
            save=False,
        )
        return [{"chain": new_chain, "roi_gain": 3.5, "failures": 0.0, "entropy": 0.3}]

    monkeypatch.setattr(planner, "mutate_chains", fake_mutate)

    mutated = planner.mutate_pipeline(
        ["a", "b"],
        workflows,
        runner=DummyRunner(),
        entropy_stability_threshold=0.2,
        failure_threshold=10,
        entropy_threshold=5.0,
        runs=1,
    )

    assert mutated and mutated[0]["chain"] == ["a", "b", "c"]
    assert mutated[0]["roi_gain"] > baseline_roi
    assert planner.cluster_map[("a", "b")]["delta_entropy"] > 0.2
    assert planner.cluster_map[("a", "b", "c")]["roi_history"][-1] == pytest.approx(3.5)

    # 3) Remerge pipelines and verify cluster map update
    class EntropyMerged:
        @staticmethod
        def _entropy(spec):
            return 0.3

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=EntropyMerged),
    )
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", EntropyMerged)

    remerged = planner.remerge_pipelines(
        [["a", "b"], ["c"]],
        workflows,
        runner=DummyRunner(),
        failure_threshold=10,
        entropy_threshold=5.0,
        runs=1,
    )

    assert remerged and remerged[0]["chain"] == ["a", "b", "c"]
    info_abc = planner.cluster_map[("a", "b", "c")]
    assert len(info_abc["roi_history"]) == 2
    assert info_abc["entropy_history"][-1] == pytest.approx(0.3)
    assert info_abc["roi_history"][-1] > baseline_roi
