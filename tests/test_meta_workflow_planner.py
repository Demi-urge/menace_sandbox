import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import pytest

import cache_utils
import meta_workflow_planner as mwp
from meta_workflow_planner import MetaWorkflowPlanner, find_synergy_candidates
from vector_utils import cosine_similarity, persist_embedding


class DummyGraph:
    """Wrapper exposing ``graph`` attribute and static I/O signatures."""

    def __init__(self, g: nx.DiGraph) -> None:
        self.graph = g

    def get_io_signature(self, _wid):
        return {"inputs": {"x": "text/plain"}, "outputs": {"x": "text/plain"}}


class DummyROI:
    """Return pre-seeded ROI trends for workflows."""

    def __init__(self, trends):
        self.trends = trends

    def fetch_trends(self, workflow_id: str):
        return self.trends.get(workflow_id, [])


class DummySynergyComparator:
    """Comparator stub returning zero aggregate synergy."""

    @staticmethod
    def compare(_a, _b):
        return types.SimpleNamespace(aggregate=0.0)


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


class DummyRetriever:
    """Simple retriever using cosine similarity over provided embeddings."""

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def _get_retriever(self):
        return self

    def retrieve(self, query_vec, top_k, dbs=None):
        hits = []
        for wid, vec in self.embeddings.items():
            score = cosine_similarity(vec, query_vec)
            hits.append(types.SimpleNamespace(record_id=wid, score=score, metadata={"id": wid}))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k], None, None


def _persist_embeddings(tmp_path: Path, embeddings: Mapping[str, Sequence[float]]):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    for wid, vec in embeddings.items():
        persist_embedding("workflow_meta", wid, vec)
    return old_cwd


@pytest.fixture
def sample_embeddings(tmp_path, monkeypatch):
    g = nx.DiGraph()
    trends = {
        "wf1": [{"roi_gain": 1.0}, {"roi_gain": 1.0}],
        "wf2": [{"roi_gain": 1.0}, {"roi_gain": 1.0}],
        "wf3": [{"roi_gain": 5.0}, {"roi_gain": 5.0}],
    }

    def fake_embed(text, embedder=None):
        return [1.0, 0.0]

    class DummyEmbedder:
        def get_sentence_embedding_dimension(self):
            return 2

    monkeypatch.setattr(mwp, "governed_embed", fake_embed)
    monkeypatch.setattr(mwp, "get_embedder", lambda: DummyEmbedder())

    class DummyBuilder:
        def build(self, *_, **__):
            return {}

        def refresh_db_weights(self):
            pass

    planner = MetaWorkflowPlanner(
        context_builder=DummyBuilder(),
        graph=DummyGraph(g),
        roi_db=DummyROI(trends),
        roi_window=3,
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
    embed_dim = 2
    yield planner, vecs, records, embed_dim
    os.chdir(old_cwd)


def test_embedding_generation(sample_embeddings):
    planner, vecs, records, embed_dim = sample_embeddings
    vec = vecs["wf1"]
    assert (
        len(vec)
        == 2
        + planner.roi_window
        + 2
        + planner.roi_window
        + 3 * embed_dim
        + planner.max_domains
        + planner.max_domains
    )
    roi_segment = vec[2:5]
    assert roi_segment[:2] == [1.0, 1.0]
    assert records[0]["id"] == "wf1"


def test_embedding_clustering(sample_embeddings):
    _planner, _vecs, records, _embed_dim = sample_embeddings
    clusters = _cluster(records, threshold=0.65)
    cluster_ids = [sorted(rec["id"] for rec in group) for group in clusters]
    assert ["wf1", "wf2", "wf3"] in cluster_ids


def test_embedding_retrieval(sample_embeddings):
    _planner, vecs, records, _embed_dim = sample_embeddings
    query_vec = vecs["wf1"]
    best = _retrieve(records, query_vec)
    assert best == "wf1"


def test_domain_transition_vector(monkeypatch):
    monkeypatch.setattr(mwp, "governed_embed", lambda text, embedder=None: [1.0, 0.0])
    monkeypatch.setattr(
        mwp, "get_embedder", lambda: type("E", (), {"get_sentence_embedding_dimension": lambda self: 2})()
    )
    class DummyBuilder:
        def build(self, *_, **__):
            return {}

        def refresh_db_weights(self):
            pass

    planner = MetaWorkflowPlanner(context_builder=DummyBuilder())
    planner.domain_index = {"other": 0, "alpha": 1, "beta": 2}
    planner.cluster_map[("__domain_transitions__",)] = {
        (1, 2): {"count": 1.0, "delta_roi": 1.0}
    }
    workflow = {"domain": "alpha"}
    vec = planner.encode_workflow("wf", workflow)
    embed_dim = 2
    base = (
        2
        + planner.roi_window
        + 2
        + planner.roi_window
        + 3 * embed_dim
    )
    domain_start = base
    trans_start = domain_start + planner.max_domains
    assert vec[domain_start + planner.domain_index["alpha"]] == 1.0
    assert vec[trans_start + planner.domain_index["beta"]] == pytest.approx(1.0)


def test_find_synergy_candidates(sample_embeddings):
    planner, _vecs, records, _embed_dim = sample_embeddings
    emb_map = {rec["id"]: rec["vector"] for rec in records}
    retr = DummyRetriever(emb_map)
    cands = find_synergy_candidates(
        "wf1",
        top_k=2,
        context_builder=planner.context_builder,
        retriever=retr,
        roi_db=planner.roi_db,
    )
    assert cands
    # wf3 has the highest ROI weight and should therefore rank first
    assert cands[0]["workflow_id"] == "wf3"
    assert cands[0]["roi"] == pytest.approx(5.0)
    assert any(c["workflow_id"] == "wf2" for c in cands)


def test_compose_pipeline_chaining(monkeypatch, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.9, 0.1],
        "wf3": [0.8, 0.2],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = DummyRetriever(embeddings)
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({})
    )
    workflows = {wid: {} for wid in embeddings}
    pipeline = planner.compose_pipeline("wf1", workflows, length=3, retriever=retr)
    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf2", "wf3"]


def test_compose_pipeline_roi_weighting(monkeypatch, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.6, 0.8],
        "wf3": [0.55, 0.835],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    roi_trends = {"wf2": [{"roi_gain": 0.0}], "wf3": [{"roi_gain": 5.0}]}
    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = DummyRetriever(embeddings)
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI(roi_trends)
    )
    workflows = {wid: {} for wid in embeddings}
    pipeline = planner.compose_pipeline("wf1", workflows, length=2, retriever=retr)
    assert pipeline == ["wf1", "wf3"]
    pipeline = planner.compose_pipeline(
        "wf1", workflows, length=2, roi_weight=0.0, retriever=retr
    )
    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf2"]


def test_compose_pipeline_synergy_weighting(monkeypatch, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    class WeightedComparator:
        @staticmethod
        def compare(a, b):
            a_id = a.get("id") if isinstance(a, dict) else a
            b_id = b.get("id") if isinstance(b, dict) else b
            if {a_id, b_id} == {"wf1", "wf3"}:
                score = 0.9
            elif {a_id, b_id} == {"wf1", "wf2"}:
                score = 0.1
            else:
                score = 0.0
            return types.SimpleNamespace(aggregate=score)

    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", WeightedComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.6, 0.8],
        "wf3": [0.55, 0.835],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = DummyRetriever(embeddings)
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({})
    )
    workflows = {wid: {"id": wid} for wid in embeddings}

    pipeline = planner.compose_pipeline(
        "wf1", workflows, length=2, synergy_weight=0.0, retriever=retr
    )
    assert pipeline == ["wf1", "wf2"]
    pipeline = planner.compose_pipeline(
        "wf1", workflows, length=2, synergy_weight=1.0, retriever=retr
    )
    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf3"]


def test_compose_pipeline_cluster_map_synergy(monkeypatch):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)

    class ZeroComparator:
        @staticmethod
        def compare(a, b):
            return types.SimpleNamespace(aggregate=0.0)

    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", ZeroComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.6, 0.8],
        "wf3": [0.55, 0.835],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    planner = MetaWorkflowPlanner(graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({}))
    planner.cluster_map = {
        ("wf1", "wf2"): {"score": 0.1},
        ("wf1", "wf3"): {"score": 0.9},
    }
    monkeypatch.setattr(MetaWorkflowPlanner, "encode_workflow", fake_encode)

    workflows = {wid: {} for wid in embeddings}
    retr = DummyRetriever(embeddings)

    pipeline = planner.compose_pipeline(
        "wf1", workflows, length=2, synergy_weight=0.0, retriever=retr
    )
    assert pipeline == ["wf1", "wf2"]
    pipeline = planner.compose_pipeline(
        "wf1", workflows, length=2, synergy_weight=1.0, retriever=retr
    )
    assert pipeline == ["wf1", "wf3"]


def test_compose_pipeline_transition_matrix(monkeypatch, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.9, 0.1],
        "wf3": [0.8, 0.2],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = DummyRetriever(embeddings)
    planner = MetaWorkflowPlanner(graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({}))
    planner.domain_index.update({"a": 1, "b": 2})
    planner.cluster_map = {
        ("__domain_transitions__",): {
            (planner.domain_index["a"], planner.domain_index["a"]): {"count": 10, "roi": -1.0}
        }
    }
    workflows = {"wf1": {"domain": "a"}, "wf2": {"domain": "a"}, "wf3": {"domain": "b"}}
    pipeline = planner.compose_pipeline("wf1", workflows, length=2, retriever=retr)
    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf2"]


def test_compose_pipeline_missing_transition(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.6, 0.8],
        "wf3": [0.9, 0.1],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = DummyRetriever(embeddings)
    planner = MetaWorkflowPlanner(graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({}))
    planner.domain_index.update({"a": 1, "b": 2, "c": 3})

    def fake_trans_probs(self):
        return {
            (planner.domain_index["a"], planner.domain_index["b"]): 0.5,
        }

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "transition_probabilities", fake_trans_probs)

    workflows = {
        "wf1": {"domain": "a"},
        "wf2": {"domain": "b"},
        "wf3": {"domain": "c"},
    }

    with caplog.at_level("DEBUG"):
        pipeline = planner.compose_pipeline("wf1", workflows, length=2, retriever=retr)

    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf3"]
    assert "no transition stats" in caplog.text


def test_compose_pipeline_negative_transition(monkeypatch, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {"wf1": [1.0], "wf2": [2.0], "wf3": [3.0]}

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    def fake_cosine(a, b):
        pair = (a[0], b[0])
        if pair in ((1.0, 2.0), (2.0, 1.0)):
            return 0.95
        if pair in ((1.0, 3.0), (3.0, 1.0)):
            return 0.75
        return 0.0

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)
    monkeypatch.setattr(mwp, "cosine_similarity", fake_cosine)

    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = DummyRetriever(embeddings)
    planner = MetaWorkflowPlanner(graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({}))
    planner.domain_index.update({"a": 1, "b": 2, "c": 3})

    def fake_trans_probs(self):
        return {
            (planner.domain_index["a"], planner.domain_index["b"]): -0.9,
        }

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "transition_probabilities", fake_trans_probs)

    workflows = {
        "wf1": {"domain": "a"},
        "wf2": {"domain": "b"},
        "wf3": {"domain": "c"},
    }

    pipeline = planner.compose_pipeline("wf1", workflows, length=2, retriever=retr)
    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf3"]


def test_update_cluster_map_records_transitions():
    planner = MetaWorkflowPlanner()

    class DummyCodeDB:
        def get_context_tags(self, wid):
            return {"a": ["alpha"], "b": ["beta"]}[wid]

    planner.code_db = DummyCodeDB()
    planner.cluster_map = {}
    planner._update_cluster_map(["a", "b"], roi_gain=2.0, failures=1.0, entropy=0.5)
    planner._update_cluster_map(["a", "b"], roi_gain=3.0, failures=2.0, entropy=1.0)
    matrix = planner.cluster_map.get(("__domain_transitions__",), {})
    entry = matrix.get(
        (planner.domain_index["alpha"], planner.domain_index["beta"])
    )
    assert (
        entry
        and entry["count"] == 2
        and entry.get("delta_roi", 0.0) != 0.0
        and entry.get("delta_failures", 0.0) != 0.0
        and entry.get("delta_entropy", 0.0) != 0.0
    )


def test_transition_probabilities_normalization():
    planner = MetaWorkflowPlanner()
    planner.domain_index = {"other": 0, "a": 1, "b": 2, "c": 3}
    planner.cluster_map = {
        ("__domain_transitions__",): {
            (1, 2): {
                "count": 2,
                "delta_roi": 1.0,
                "delta_failures": 0.1,
                "delta_entropy": 0.1,
            },
            (1, 3): {
                "count": 1,
                "delta_roi": 1.0,
                "delta_failures": 0.3,
                "delta_entropy": 0.4,
            },
            (2, 3): {
                "count": 1,
                "delta_roi": -1.0,
                "delta_failures": 0.1,
                "delta_entropy": 0.1,
            },
        }
    }
    probs = planner.transition_probabilities()
    assert probs[(1, 2)] > probs[(1, 3)] > 0
    assert probs[(2, 3)] == 0
    assert abs(sum(probs.values()) - 1.0) < 1e-8


@pytest.mark.parametrize("use_sklearn", [True, False])
def test_cluster_workflows_roi_weighting(monkeypatch, use_sklearn):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "_load_cluster_map", lambda self: None)
    monkeypatch.setattr(
        cache_utils,
        "_get_cache",
        lambda: types.SimpleNamespace(get=lambda *a, **k: None),
    )

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.5, 0.5],
        "wf3": [-1.0, 0.0],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    roi_trends = {"wf2": [{"roi_gain": 0.5}]}
    planner = MetaWorkflowPlanner(roi_db=DummyROI(roi_trends))
    workflows = {wid: {} for wid in embeddings}
    retr = DummyRetriever(embeddings)

    monkeypatch.setattr(mwp, "_HAS_SKLEARN", use_sklearn)
    if use_sklearn:
        sk = pytest.importorskip("sklearn.cluster")
        monkeypatch.setattr(mwp, "DBSCAN", sk.DBSCAN)
    else:
        monkeypatch.setattr(mwp, "DBSCAN", None)

    clusters = planner.cluster_workflows(
        workflows, retriever=retr, epsilon=0.5, min_samples=2
    )
    cluster_ids = [sorted(c) for c in clusters]
    assert ["wf1", "wf2"] in cluster_ids
    assert ["wf3"] in cluster_ids


def test_compose_pipeline_io_compatibility(monkeypatch, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.9, 0.1],
        "wf3": [0.8, 0.2],
        "wf4": [0.95, 0.05],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    class SigGraph:
        def __init__(self, sigs):
            self.sigs = sigs

        def get_io_signature(self, wid):
            return self.sigs.get(wid)

    sigs = {
        "wf1": {"outputs": {"a": "text/plain"}},
        # wf2 requires an extra input not produced by wf1
        "wf2": {
            "inputs": {"a": "text/plain", "extra": "text/plain"},
            "outputs": {"b": "text/plain"},
        },
        # wf3 matches wf1's outputs exactly
        "wf3": {"inputs": {"a": "text/plain"}, "outputs": {"c": "text/plain"}},
        # wf4 lacks explicit input type and should be skipped
        "wf4": {"inputs": {"a": ""}, "outputs": {"d": "text/plain"}},
    }
    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = DummyRetriever(embeddings)
    planner = MetaWorkflowPlanner(graph=SigGraph(sigs), roi_db=DummyROI({}))
    workflows = {wid: {} for wid in embeddings}
    pipeline = planner.compose_pipeline("wf1", workflows, length=3, retriever=retr)
    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf3"]


def test_compose_pipeline_retrieval_limited(monkeypatch, tmp_path):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {
        "wf1": [1.0, 0.0],
        "wf2": [0.9, 0.1],
        "wf3": [0.8, 0.2],
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    class LimitedRetriever(DummyRetriever):
        def retrieve(self, query_vec, top_k, dbs=None):
            hit = types.SimpleNamespace(record_id="wf3", score=1.0, metadata={"id": "wf3"})
            return [hit], None, None

    old_cwd = _persist_embeddings(tmp_path, embeddings)
    retr = LimitedRetriever(embeddings)
    planner = MetaWorkflowPlanner(graph=DummyGraph(nx.DiGraph()), roi_db=DummyROI({}))
    workflows = {wid: {} for wid in embeddings}
    pipeline = planner.compose_pipeline("wf1", workflows, length=2, retriever=retr)
    os.chdir(old_cwd)
    assert pipeline == ["wf1", "wf3"]


def test_compose_meta_workflow_skip_incompatible(monkeypatch):
    monkeypatch.setattr(
        mwp,
        "find_synergy_chain",
        lambda start_workflow_id, length=5: ["wf1", "wf2", "wf3"],
    )

    class SigGraph:
        def __init__(self, sigs):
            self.sigs = sigs

        def get_io_signature(self, wid):
            return self.sigs.get(wid)

    sigs = {
        "wf1": {"outputs": {"a": "text/plain"}},
        "wf2": {
            "inputs": {"b": "text/plain"},
            "outputs": {"c": "text/plain"},
        },
        "wf3": {"inputs": {"a": "text/plain"}, "outputs": {"d": "text/plain"}},
    }

    result = mwp.compose_meta_workflow("wf1", length=3, graph=SigGraph(sigs))
    assert result["chain"] == "wf1->wf3"
    assert [s["workflow_id"] for s in result["steps"]] == ["wf1", "wf3"]


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
        ("c", "d"): {"delta_roi": 3.0, "delta_entropy": 2.5, "converged": True},
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
