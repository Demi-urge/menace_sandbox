import json
import os
from pathlib import Path

import networkx as nx
import pytest

from meta_workflow_planner import MetaWorkflowPlanner
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
    assert len(vec) == 2 + 3 + planner.max_functions + planner.max_modules + planner.max_tags
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

