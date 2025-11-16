import sys
import pytest
from vector_service import Retriever


class _Hit:
    origin_db = "code"
    score = 1.0
    reason = "match"
    metadata = {
        "code": "print('hello')",
        "contextual_metrics": {"complexity": 2.0, "model_score": 1.0},
        "redacted": True,
    }


class _DummyUR:
    def retrieve_with_confidence(self, query: str, top_k: int = 1):
        return [_Hit()], 1.0, []


def test_code_snippet_retrieval():
    retriever = Retriever(retriever=_DummyUR())
    hits = retriever.search("greeting", top_k=1)
    assert hits and hits[0]["origin_db"] == "code"
    assert "print('hello')" in hits[0]["metadata"]["code"]
    metrics = hits[0]["metadata"]["contextual_metrics"]
    assert metrics["model_score"] == pytest.approx(1.0)
