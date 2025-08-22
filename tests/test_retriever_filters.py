import pytest

from vector_service import Retriever
from patch_safety import _VIOLATIONS


class DummyHit:
    def __init__(self, text, rid):
        self.origin_db = "db"
        self.record_id = rid
        self.score = 0.9
        self.text = text
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


def test_search_filters_license_hit():
    hits = [
        DummyHit("text", "1"),
        DummyHit("regular text", "2"),
    ]
    hits[0].metadata["license"] = "GPL-3.0"
    r = Retriever(retriever=DummyRetriever(hits), cache=None)
    start = _VIOLATIONS.labels("license")._value.get()
    res = r.search("q")
    assert len(res) == 1
    assert res[0]["record_id"] == "2"
    assert _VIOLATIONS.labels("license")._value.get() == start + 1


def test_search_attaches_semantic_alerts():
    hits = [DummyHit("eval('data')", "1")]
    r = Retriever(retriever=DummyRetriever(hits), cache=None)
    res = r.search("q")
    alerts = res[0]["metadata"].get("semantic_alerts")
    assert alerts and any("eval" in alert[1] for alert in alerts)
