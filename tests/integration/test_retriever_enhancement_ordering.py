import types

from vector_service.retriever import PatchRetriever


class DummyStore:
    def __init__(self, vectors, meta):
        self.vectors = vectors
        self.ids = [str(i) for i in range(len(vectors))]
        self.meta = meta

    def query(self, vector, top_k=5):
        return [(vid, 0.0) for vid in self.ids[:top_k]]


def test_enhancement_score_affects_ordering():
    vectors = [[1.0, 0.0], [1.0, 0.0]]
    meta = [
        {"origin_db": "patch", "metadata": {"text": "a", "enhancement_score": 0.0}},
        {"origin_db": "patch", "metadata": {"text": "b", "enhancement_score": 1.0}},
    ]
    store = DummyStore(vectors, meta)
    vec_service = types.SimpleNamespace(vectorise=lambda kind, record: [1.0, 0.0])
    pr = PatchRetriever(store=store, vector_service=vec_service, enhancement_weight=1.0)
    results = pr.search("query", top_k=2)
    ids = [r["record_id"] for r in results]
    assert ids == ["1", "0"]
