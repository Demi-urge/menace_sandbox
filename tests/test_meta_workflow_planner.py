import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

import meta_workflow_planner as mwp
from meta_workflow_planner import MetaWorkflowPlanner, find_synergy_candidates
from vector_utils import cosine_similarity


class DummyGraph:
    """Wrapper exposing ``graph`` attribute for the planner."""

    def __init__(self, g: nx.DiGraph) -> None:
        self.graph = g


class DummyROI:
    """Return pre-seeded ROI trends for workflows."""

    def __init__(self, trends):
        self.trends = trends

    def fetch_trends(self, workflow_id: str):
        return self.trends.get(workflow_id, [])


def _load_records(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _cluster(records, threshold: float = 0.72):
    clusters = []
    for rec in records:
        placed = False
        for cluster in clusters:
            if cosine_similarity(rec["vector"], cluster[0]["vector"]) >= threshold:
                cluster.append(rec)
                placed = True
                break
        if not placed:
            clusters.append([rec])
    return clusters


def _retrieve(records, query_vec):
    best_id = None
    best_score = -1.0
    for rec in records:
        score = cosine_similarity(rec["vector"], query_vec)
        if score > best_score:
            best_id = rec["id"]
            best_score = score
    return best_id


@pytest.fixture
def sample_embeddings(tmp_path):
    g = nx.DiGraph()
    trends = {
        "wf1": [{"roi_gain": 1.0}, {"roi_gain": 1.0}],
        "wf2": [{"roi_gain": 1.0}, {"roi_gain": 1.0}],
        "wf3": [{"roi_gain": 5.0}, {"roi_gain": 5.0}],
    }
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(g), roi_db=DummyROI(trends), roi_window=3
    )
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    workflows = {
        "wf1": {"workflow": ["common", "a"]},
        "wf2": {"workflow": ["common", "b"]},
        "wf3": {"workflow": ["x"]},
    }
    vecs = {wid: planner.encode(wid, wf) for wid, wf in workflows.items()}
    records = _load_records(Path("embeddings.jsonl"))
    yield planner, vecs, records
    os.chdir(old_cwd)


def test_embedding_generation(sample_embeddings):
    planner, vecs, records = sample_embeddings
    vec = vecs["wf1"]
    assert (
        len(vec)
        == 2
        + planner.roi_window
        + 2
        + planner.roi_window
        + planner.max_functions
        + planner.max_modules
        + planner.max_tags
    )
    roi_segment = vec[2:5]
    assert roi_segment[:2] == [1.0, 1.0]
    assert records[0]["id"] == "wf1"


def test_embedding_clustering(sample_embeddings):
    _planner, _vecs, records = sample_embeddings
    clusters = _cluster(records)
    cluster_ids = [sorted(rec["id"] for rec in group) for group in clusters]
    assert ["wf1", "wf2"] in cluster_ids
    assert ["wf3"] in cluster_ids


def test_embedding_retrieval(sample_embeddings):
    _planner, vecs, records = sample_embeddings
    query_vec = vecs["wf1"]
    best = _retrieve(records, query_vec)
    assert best == "wf1"


def test_find_synergy_candidates(sample_embeddings):
    planner, _vecs, _records = sample_embeddings
    cands = find_synergy_candidates("wf1", top_k=2, retriever=None, roi_db=planner.roi_db)
    assert cands
    # wf3 has the highest ROI weight and should therefore rank first
    assert cands[0]["workflow_id"] == "wf3"
    assert cands[0]["roi"] == pytest.approx(5.0)
    assert any(c["workflow_id"] == "wf2" for c in cands)


def test_compose_pipeline_chaining(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.9, 0.1],
        "wf3": [0.8, 0.2],
    }
    monkeypatch.setattr(mwp, "_load_embeddings", lambda: embeddings)

    planner = MetaWorkflowPlanner(
        graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({})
    )
    workflows = {wid: {} for wid in embeddings}
    pipeline = planner.compose_pipeline("wf1", workflows, length=3)
    assert pipeline == ["wf1", "wf2", "wf3"]


def test_compose_pipeline_roi_weighting(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.6, 0.8],
        "wf3": [0.55, 0.835],
    }
    monkeypatch.setattr(mwp, "_load_embeddings", lambda: embeddings)

    roi_trends = {"wf2": [{"roi_gain": 0.0}], "wf3": [{"roi_gain": 5.0}]}
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI(roi_trends)
    )
    workflows = {wid: {} for wid in embeddings}
    pipeline = planner.compose_pipeline("wf1", workflows, length=2)
    assert pipeline == ["wf1", "wf3"]
    pipeline = planner.compose_pipeline("wf1", workflows, length=2, roi_weight=0.0)
    assert pipeline == ["wf1", "wf2"]


def test_cluster_workflows_roi_weighting(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.5, 0.5],
        "wf3": [-1.0, 0.0],
    }
    monkeypatch.setattr(mwp, "_load_embeddings", lambda: embeddings)

    roi_trends = {"wf2": [{"roi_gain": 0.5}]}
    planner = MetaWorkflowPlanner(roi_db=DummyROI(roi_trends))
    workflows = {wid: {} for wid in embeddings}
    clusters = planner.cluster_workflows(workflows, threshold=0.75)
    cluster_ids = [sorted(c) for c in clusters]
    assert ["wf1", "wf2"] in cluster_ids
    assert ["wf3"] in cluster_ids


def test_compose_pipeline_io_compatibility(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.9, 0.1],
        "wf3": [0.8, 0.2],
    }
    monkeypatch.setattr(mwp, "_load_embeddings", lambda: embeddings)

    class SigGraph:
        def __init__(self, sigs):
            self.sigs = sigs

        def get_io_signature(self, wid):
            return self.sigs.get(wid)

    sigs = {
        "wf1": {"outputs": ["a"]},
        "wf2": {"inputs": ["a"], "outputs": ["b"]},
        "wf3": {"inputs": ["c"], "outputs": ["d"]},
    }
    planner = MetaWorkflowPlanner(graph=SigGraph(sigs), roi_db=DummyROI({}))
    workflows = {wid: {} for wid in embeddings}
    pipeline = planner.compose_pipeline("wf1", workflows, length=3)
    assert pipeline == ["wf1", "wf2"]


def test_plan_and_validate_sandbox(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    class DummyDB:
        def log_result(self, **kwargs):
            self.logged = kwargs

    planner = MetaWorkflowPlanner(
        graph=DummyGraph(nx.DiGraph()), roi_db=DummyDB()
    )

    class DummySuggester:
        def suggest_chains(self, target_embedding, top_k=3):
            return [["wf1", "wf2"]]

    monkeypatch.setitem(
        sys.modules,
        "workflow_chain_suggester",
        types.SimpleNamespace(WorkflowChainSuggester=DummySuggester),
    )

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

    def wf1():
        return 1.0

    def wf2():
        return 2.0

    workflows = {"wf1": wf1, "wf2": wf2}
    records = planner.plan_and_validate([0.0], workflows, runner=DummyRunner(), top_k=1)
    assert records and records[0]["chain"] == ["wf1", "wf2"]
    assert records[0]["roi_gain"] == pytest.approx(3.0)


def test_iterate_pipelines(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)

    class DummyROI:
        def __init__(self):
            self.logged: list[dict[str, Any]] = []

        def log_result(self, **kwargs):
            self.logged.append(kwargs)

    class DummyStability:
        def __init__(self):
            self.data: dict[str, dict[str, float]] = {}
            self.records: list[tuple[str, float, int, float]] = []

        def record_metrics(self, workflow_id, roi, failures, entropy, *, roi_delta=None):
            self.records.append((workflow_id, roi, failures, entropy))
            self.data[workflow_id] = {"entropy": entropy}

    planner = MetaWorkflowPlanner(roi_db=DummyROI(), stability_db=DummyStability())
    planner.cluster_map = {
        ("a", "b"): {"delta_roi": -0.1, "converged": True},
        ("c", "d"): {"delta_roi": 0.5, "converged": True},
        ("e",): {"delta_roi": 0.4, "converged": True},
        ("f",): {"delta_roi": 0.3, "converged": True},
    }
    planner.stability_db.data.update(
        {
            "a->b": {"entropy": 0.2},
            "c->d": {"entropy": 2.5},
            "e": {"entropy": 0.3},
            "f": {"entropy": 0.2},
        }
    )

    def fake_mutate(chains, workflows, **_):
        assert chains == [["a", "b"]]
        return [{"chain": ["a", "b", "x"], "roi_gain": 1.0, "failures": 0, "entropy": 0.1}]

    def fake_split(pipeline, workflows, **_):
        assert list(pipeline) == ["c", "d"]
        return [
            {"chain": ["c"], "roi_gain": 0.2, "failures": 0, "entropy": 0.5},
            {"chain": ["d"], "roi_gain": 0.4, "failures": 0, "entropy": 0.5},
        ]

    def fake_remerge(pipelines, workflows, **_):
        assert pipelines == [["e"], ["f"]]
        return [{"chain": ["e", "f"], "roi_gain": 0.8, "failures": 0, "entropy": 0.2}]

    monkeypatch.setattr(planner, "mutate_chains", fake_mutate)
    monkeypatch.setattr(planner, "split_pipeline", fake_split)
    monkeypatch.setattr(planner, "remerge_pipelines", fake_remerge)

    records = planner.iterate_pipelines({})
    chains = [r["chain"] for r in records]
    assert ["a", "b", "x"] in chains
    assert ["d"] in chains
    assert ["e", "f"] in chains
    assert len(planner.roi_db.logged) == 3
    assert len(planner.stability_db.records) == 3
