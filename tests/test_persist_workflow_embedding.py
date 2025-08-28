import types

import workflow_vectorizer as wv


def test_persist_workflow_embedding_includes_metadata(monkeypatch):
    captured = {}

    def fake_persist(kind, record_id, vec, *, origin_db=None, metadata=None, path="embeddings.jsonl"):
        captured["kind"] = kind
        captured["id"] = record_id
        captured["vec"] = list(vec)
        captured["origin_db"] = origin_db
        captured["metadata"] = metadata or {}

    monkeypatch.setattr(wv, "persist_embedding", fake_persist)

    workflow = {
        "category": "ops",
        "status": "new",
        "workflow": [],
        "roi": 5.0,
        "failure_rate": 0.25,
        "failure_reason": "boom",
    }

    vec = wv.persist_workflow_embedding("w1", workflow)

    assert captured["kind"] == "workflow"
    assert captured["id"] == "w1"
    assert captured["origin_db"] == "workflow"
    assert "roi_curve" in captured["metadata"]
    assert captured["metadata"]["roi"] == 5.0
    assert captured["metadata"]["failure_rate"] == 0.25
    assert captured["metadata"]["failure_reason"] == "boom"
    assert isinstance(vec, list)
