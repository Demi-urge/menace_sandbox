from vector_service.vectorizer import SharedVectorService


def test_failure_and_research_embeddings_persist(monkeypatch):
    calls = []

    def fake_persist(kind, record_id, vec, *, path="embeddings.jsonl"):
        calls.append((kind, record_id, list(vec)))

    monkeypatch.setattr(
        "vector_service.vectorizer.persist_embedding", fake_persist
    )

    svc = SharedVectorService()

    failure = {
        "cause": "timeout",
        "demographics": "general",
        "profitability": 1000.0,
        "retention": 50.0,
        "cac": 200.0,
        "roi": 1.5,
    }
    research = {
        "category": "ai",
        "type": "text",
        "data_depth": 0.5,
        "energy": 10,
        "corroboration_count": 2,
    }

    f_vec = svc.vectorise_and_store("failure", "f1", failure)
    r_vec = svc.vectorise_and_store("research", "r1", research)

    assert calls[0][0] == "failure" and len(f_vec) > 0
    assert calls[1][0] == "research" and len(r_vec) > 0
