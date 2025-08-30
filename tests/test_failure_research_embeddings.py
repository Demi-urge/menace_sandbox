from vector_service.vectorizer import SharedVectorService


class DummyStore:
    def __init__(self):
        self.calls = []

    def add(self, kind, record_id, vector, *, origin_db=None, metadata=None):
        self.calls.append((kind, record_id, list(vector)))

    def query(self, vector, top_k=5):  # pragma: no cover - unused
        return []

    def load(self):  # pragma: no cover - unused
        pass


def test_failure_and_research_embeddings_persist():
    store = DummyStore()
    svc = SharedVectorService(vector_store=store)

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

    assert store.calls[0][0] == "failure" and len(f_vec) > 0
    assert store.calls[1][0] == "research" and len(r_vec) > 0
