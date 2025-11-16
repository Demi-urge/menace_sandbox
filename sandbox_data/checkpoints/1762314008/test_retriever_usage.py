from vector_service import Retriever


class _Hit:
    def __init__(self, origin: str, rid: int):
        self.origin_db = origin
        self.score = 0.9
        self.metadata = {
            "id": rid,
            "contextual_metrics": {"model_score": 1.0},
            "redacted": True,
        }
        self.reason = "match"


class _DummyUR:
    def retrieve_with_confidence(self, query: str, top_k: int = 5):
        return [_Hit("bot", 1), _Hit("workflow", 2)], 0.9, []


def test_retriever_search():
    retriever = Retriever(retriever=_DummyUR())
    hits = retriever.search("q", top_k=5)
    assert hits and {h["origin_db"] for h in hits} == {"bot", "workflow"}

