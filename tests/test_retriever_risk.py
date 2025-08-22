import pytest

from vector_service import Retriever


class DummyHit:
    def __init__(self, score=1.0):
        self.origin_db = "db"
        self.record_id = "1"
        self.score = score
        self.text = "text"
        self.metadata = {"redacted": True}

    def to_dict(self):
        return {
            "origin_db": self.origin_db,
            "record_id": self.record_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


class DummyRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve_with_confidence(self, query, top_k):
        return self.hits, 0.9, None


def test_risk_score_propagation_and_penalty(monkeypatch):
    hit = DummyHit()
    r = Retriever(retriever=DummyRetriever([hit]), cache=None, risk_penalty=2.0)
    monkeypatch.setattr(
        r.patch_safety,
        "evaluate",
        lambda meta, err=None, origin="": (True, 0.2, {}),
    )
    res = r.search("q")
    assert res[0]["risk_score"] == pytest.approx(0.2)
    assert res[0]["metadata"]["risk_score"] == pytest.approx(0.2)
    assert res[0]["score"] == pytest.approx(0.6)
