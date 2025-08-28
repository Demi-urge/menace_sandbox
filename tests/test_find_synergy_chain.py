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
