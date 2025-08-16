import logging
from types import SimpleNamespace

from vector_service import Retriever, MalformedPromptError, RateLimitError


class DummyRetriever:
    def __init__(self, score=0.5, raise_error: Exception | None = None):
        self.score = score
        self.raise_error = raise_error
        self.calls = []

    def retrieve(self, query, top_k=5):
        if self.raise_error:
            raise self.raise_error
        self.calls.append((query, top_k))
        hit = SimpleNamespace(
            origin_db="bot",
            record_id="1",
            score=self.score,
            metadata={},
            reason="",
        )
        return [hit], "sid", [("bot", "1")]


def test_retriever_fallback_low_score():
    primary = DummyRetriever(score=0.1)
    fallback = DummyRetriever(score=0.9)
    r = Retriever(retriever=primary, fallback_retriever=fallback, score_threshold=0.5)
    res = r.search("alpha")
    assert res[0]["score"] == 0.9
    assert primary.calls and fallback.calls


def test_logging_includes_session_id(caplog):
    dummy = DummyRetriever()
    r = Retriever(retriever=dummy)
    caplog.set_level(logging.INFO)
    r.search("beta", session_id="sess")
    assert any(rec.session_id == "sess" for rec in caplog.records)


def test_malformed_query_raises():
    r = Retriever(retriever=DummyRetriever())
    try:
        r.search(" ")
    except MalformedPromptError:
        pass
    else:  # pragma: no cover - ensure exception was raised
        assert False


def test_rate_limit_error():
    err = Exception("Rate limit exceeded")
    r = Retriever(retriever=DummyRetriever(raise_error=err))
    try:
        r.search("gamma")
    except RateLimitError:
        pass
    else:  # pragma: no cover
        assert False
