import sys
import types
from pathlib import Path
import importlib.util

import pytest

stub_run = types.ModuleType("run_autonomous")
stub_run.LOCAL_KNOWLEDGE_MODULE = None
sys.modules.setdefault("run_autonomous", stub_run)

spec = importlib.util.spec_from_file_location(
    "meta_workflow_planner", Path(__file__).resolve().parent.parent / "meta_workflow_planner.py"  # path-ignore
)
mwp = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
assert spec.loader is not None
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules[spec.name] = mwp
spec.loader.exec_module(mwp)  # type: ignore[misc]


def test_encode_chain_updates_cluster_map(monkeypatch):
    planner = mwp.MetaWorkflowPlanner()
    planner.cluster_map = {}
    monkeypatch.setattr(planner, "_save_cluster_map", lambda: None)
    monkeypatch.setattr(mwp, "persist_embedding", lambda *a, **k: None)
    monkeypatch.setattr(mwp, "get_cached_chain", lambda cid: None)
    captured = {}

    def fake_set(cid, vec):
        captured["id"] = cid
        captured["vec"] = vec

    monkeypatch.setattr(mwp, "set_cached_chain", fake_set)
    monkeypatch.setattr(planner, "encode", lambda cid, wf: [0.1, 0.2])

    vec = planner.encode_chain(["a", "b"])

    assert vec == [0.1, 0.2]
    assert captured["id"] == "a->b"
    assert planner.cluster_map[("a", "b")]["embedding"] == [0.1, 0.2]


class ModuleMetric:
    def __init__(self, name: str, result: float, success: bool = True) -> None:
        self.name = name
        self.result = result
        self.success = success
        self.duration = 0.0


class DummyRunner:
    def run(self, funcs):
        modules = [
            ModuleMetric(fn.__name__, 1.5 if i == 0 else 0.0, success=True)
            for i, fn in enumerate(funcs)
        ]
        return types.SimpleNamespace(modules=modules, crash_count=0)


def test_mutate_pipeline_triggers_on_entropy(tmp_path, monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    class DummyROIResultsDB:
        def __init__(self, *a, **k):
            pass

        def log_result(self, *a, **k):
            pass

        def fetch_results(self, *a, **k):
            return []

    monkeypatch.setattr(mwp, "ROIResultsDB", DummyROIResultsDB)
    monkeypatch.setattr(mwp, "persist_embedding", lambda *a, **k: None)
    monkeypatch.setattr(mwp, "_load_embeddings", lambda *a, **k: {})

    class StubComparator:
        @staticmethod
        def _entropy(spec):
            return 3.0

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )

    planner = mwp.MetaWorkflowPlanner()
    planner.cluster_map = {}
    monkeypatch.setattr(planner, "_save_cluster_map", lambda: None)

    baseline_steps = [
        {"module": "a", "roi": 1.0, "failures": 0, "entropy": 0.0},
        {"module": "b", "roi": 0.0, "failures": 0, "entropy": 0.0},
    ]
    planner._update_cluster_map(
        ["a", "b"],
        roi_gain=1.0,
        failures=0,
        entropy=0.0,
        step_metrics=baseline_steps,
        save=False,
    )

    workflows = {"a": lambda: 1.0, "b": lambda: 1.0, "c": lambda: 1.0}
    runner = DummyRunner()

    def fake_mutate_chains(chains, workflows, **kwargs):
        new_chain = list(chains[0]) + ["c"]
        planner._update_cluster_map(
            new_chain,
            roi_gain=2.0,
            failures=0,
            entropy=0.0,
            step_metrics=[
                {"module": m, "roi": 1.0, "failures": 0, "entropy": 0.0}
                for m in new_chain
            ],
            save=False,
        )
        return [
            {"chain": new_chain, "roi_gain": 2.0, "failures": 0, "entropy": 0.0}
        ]

    monkeypatch.setattr(planner, "mutate_chains", fake_mutate_chains)
    monkeypatch.setattr(mwp, "shd", types.SimpleNamespace(record=lambda *a, **k: None))

    results = planner.mutate_pipeline(
        ["a", "b"],
        workflows,
        runner=runner,
        entropy_stability_threshold=1.0,
        failure_threshold=10,
        entropy_threshold=5.0,
    )

    assert results[0]["chain"] == ["a", "b", "c"]
    info = planner.cluster_map[("a", "b")]
    assert info["delta_roi"] == pytest.approx(0.5)
    assert info["delta_entropy"] == pytest.approx(3.0)
    assert ("a", "b", "c") in planner.cluster_map


def test_simulate_meta_workflow_aggregates(monkeypatch):
    class Metrics:
        def __init__(self, result: float, success: bool) -> None:
            self.modules = [ModuleMetric("m", result, success)]
            self.crash_count = 0 if success else 1

    class DummyRunner:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)

        def run(self, funcs):
            roi, ok = self.outcomes.pop(0)
            return Metrics(roi, ok)

    outcomes = [(1.0, True), (2.0, False), (3.0, True)]
    runner = DummyRunner(outcomes)

    class StubComparator:
        @staticmethod
        def _entropy(spec):
            return 0.5

    monkeypatch.setitem(
        sys.modules,
        "workflow_synergy_comparator",
        types.SimpleNamespace(WorkflowSynergyComparator=StubComparator),
    )

    spec = {
        "steps": [
            {"workflow_id": "a", "workflow": lambda: 1.0},
            {
                "steps": [
                    {"workflow_id": "b", "workflow": lambda: 2.0},
                    {"workflow_id": "c", "workflow": lambda: 3.0},
                ]
            },
        ]
    }

    result = mwp.simulate_meta_workflow(spec, runner=runner)
    assert result["roi_gain"] == pytest.approx(6.0)
    assert result["failures"] == 1
    assert result["entropy"] == pytest.approx(0.5)
