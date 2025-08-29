from vector_utils import persist_embedding
import meta_workflow_planner as mwp


def test_find_synergy_chain_prefers_persisted(tmp_path, monkeypatch):
    path = tmp_path / "embeddings.jsonl"
    persist_embedding("workflow_meta", "a", [1.0, 0.0], path=path)
    persist_embedding("workflow_meta", "b", [0.0, 1.0], path=path)
    persist_embedding(
        "workflow_chain",
        "a->b",
        [1.0, 0.0],
        path=path,
        metadata={"roi": 1.0, "entropy": 0.0},
    )
    orig_emb = mwp._load_embeddings
    orig_chain = mwp._load_chain_embeddings
    monkeypatch.setattr(mwp, "_load_embeddings", lambda path=path: orig_emb(path))
    monkeypatch.setattr(mwp, "_load_chain_embeddings", lambda path=path: orig_chain(path))
    chain = mwp.find_synergy_chain("a", length=2)
    assert chain == ["a", "b"]


def test_find_synergy_chain_prefers_reinforced(tmp_path, monkeypatch):
    path = tmp_path / "embeddings.jsonl"
    persist_embedding("workflow_meta", "a", [1.0, 0.0], path=path)
    persist_embedding("workflow_meta", "b", [0.6, 0.8], path=path)
    persist_embedding("workflow_meta", "c", [0.9, 0.1], path=path)

    orig_emb = mwp._load_embeddings
    orig_chain = mwp._load_chain_embeddings
    monkeypatch.setattr(mwp, "_load_embeddings", lambda path=path: orig_emb(path))
    monkeypatch.setattr(mwp, "_load_chain_embeddings", lambda path=path: orig_chain(path))

    class DummyTracker:
        def __init__(self, *_, **__):
            self.final_roi_history = {"b": [1.0], "c": [1.0]}

        def load_history(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(mwp, "ROITracker", DummyTracker)
    monkeypatch.setattr(mwp, "WorkflowGraph", lambda *a, **k: None)
    monkeypatch.setattr(mwp, "ROIResultsDB", lambda *a, **k: None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "CodeDB", None)
    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "_load_cluster_map", lambda self: None)
    monkeypatch.setattr(
        mwp.MetaWorkflowPlanner, "_workflow_domain", lambda self, wid: ([0], ["d"])
    )
    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "transition_probabilities", lambda self: {})

    chain = mwp.find_synergy_chain("a", length=2, cluster_map={})
    assert chain == ["a", "c"]

    chain = mwp.find_synergy_chain(
        "a", length=2, cluster_map={("a", "b"): {"score": 1.0}}
    )
    assert chain == ["a", "b"]


def test_find_synergy_chain_penalises_failure_entropy(tmp_path, monkeypatch):
    path = tmp_path / "embeddings.jsonl"
    persist_embedding("workflow_meta", "a", [1.0, 0.0], path=path)
    # "b" is more similar to "a" than "c" but has worse metrics
    persist_embedding("workflow_meta", "b", [0.9, 0.1], path=path)
    persist_embedding("workflow_meta", "c", [0.6, 0.8], path=path)

    orig_emb = mwp._load_embeddings
    orig_chain = mwp._load_chain_embeddings
    monkeypatch.setattr(mwp, "_load_embeddings", lambda path=path: orig_emb(path))
    monkeypatch.setattr(mwp, "_load_chain_embeddings", lambda path=path: orig_chain(path))

    class DummyTracker:
        def __init__(self, *_, **__):
            self.final_roi_history = {"b": [1.0], "c": [1.0]}

        def load_history(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(mwp, "ROITracker", DummyTracker)
    monkeypatch.setattr(mwp, "WorkflowGraph", lambda *a, **k: None)
    monkeypatch.setattr(mwp, "ROIResultsDB", lambda *a, **k: None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "CodeDB", None)
    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "_load_cluster_map", lambda self: None)
    monkeypatch.setattr(
        mwp.MetaWorkflowPlanner, "_workflow_domain", lambda self, wid: ([0], ["d"])
    )
    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "transition_probabilities", lambda self: {})

    cluster_map = {
        ("b",): {"failure_history": [0.9], "entropy_history": [0.9]},
        ("c",): {"failure_history": [0.1], "entropy_history": [0.1]},
    }

    chain = mwp.find_synergy_chain("a", length=2, cluster_map=cluster_map)
    assert chain == ["a", "c"]
