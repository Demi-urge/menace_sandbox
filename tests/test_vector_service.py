import logging
from types import SimpleNamespace
import pytest

from vector_service import (
    Retriever,
    ContextBuilder as VSContextBuilder,
    PatchLogger,
    EmbeddingBackfill,
    MalformedPromptError,
    RateLimitError,
    VectorServiceError,
)


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


def test_retriever_primary_success():
    dummy = DummyRetriever(score=0.8)
    r = Retriever(retriever=dummy)
    res = r.search("alpha")
    assert res and res[0]["score"] == 0.8
    assert dummy.calls


def test_retriever_fallback_no_results():
    class EmptyRetriever(DummyRetriever):
        def retrieve(self, query, top_k=5):
            self.calls.append((query, top_k))
            return [], "sid", []

    class FallbackRetriever(DummyRetriever):
        def retrieve(self, query, top_k=5):
            self.calls.append((query, top_k))
            hit = SimpleNamespace(
                origin_db="bot", record_id="1", score=self.score, metadata={}
            )
            return [hit], "sid", [("bot", "1")]

    primary = EmptyRetriever()
    fallback = FallbackRetriever(score=0.9)
    r = Retriever(retriever=primary, fallback_retriever=fallback)
    res = r.search("alpha")
    assert res and res[0]["score"] == 0.9
    assert primary.calls and fallback.calls


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


class DummyBuilder:
    def __init__(self, response: str = "ok", error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def build_context(self, desc: str, **kwargs):
        self.calls.append((desc, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_context_builder_build(monkeypatch):
    dummy = DummyBuilder("CTX")
    monkeypatch.setattr("vector_service._LegacyContextBuilder", lambda: dummy)
    cb = VSContextBuilder()
    res = cb.build("task", session_id="sid")
    assert res == "CTX"
    assert dummy.calls[0][0] == "task"


def test_context_builder_errors(monkeypatch):
    cb = VSContextBuilder()
    with pytest.raises(MalformedPromptError):
        cb.build(" ")

    err_builder = DummyBuilder(error=Exception("rate limit exceeded"))
    monkeypatch.setattr("vector_service._LegacyContextBuilder", lambda: err_builder)
    with pytest.raises(RateLimitError):
        cb.build("task")

    other_builder = DummyBuilder(error=Exception("boom"))
    monkeypatch.setattr("vector_service._LegacyContextBuilder", lambda: other_builder)
    with pytest.raises(VectorServiceError):
        cb.build("task")


def test_patch_logger_metrics_db():
    calls = []

    class DummyMetrics:
        def log_patch_outcome(self, patch_id, result, pairs, session_id=""):
            calls.append((patch_id, result, pairs, session_id))

    logger = PatchLogger(metrics_db=DummyMetrics())
    logger.track_contributors(["db1:1", "2"], True, patch_id="p", session_id="s")
    assert calls[0][2] == [("db1", "1"), ("", "2")]
    assert calls[0][3] == "s"


def test_patch_logger_vector_metrics():
    calls = []

    class DummyVector:
        def update_outcome(self, session_id, pairs, contribution, patch_id, win, regret):
            calls.append((session_id, pairs, contribution, patch_id, win, regret))

    logger = PatchLogger(vector_metrics=DummyVector())
    logger.track_contributors(["db1:1"], False, patch_id="p", session_id="s")
    assert calls[0][1] == [("db1", "1")]
    assert calls[0][5] is True


def test_embedding_backfill_run(monkeypatch):
    instances = []

    class DummyDB:
        def __init__(self, vector_backend=None):
            self.backend = vector_backend

    def fake_process(db, batch_size, session_id=""):
        instances.append((db, batch_size, session_id))

    eb = EmbeddingBackfill(batch_size=10, backend="ann")
    monkeypatch.setattr(eb, "_load_known_dbs", lambda: [DummyDB])
    monkeypatch.setattr(eb, "_process_db", fake_process)
    eb.run(session_id="sid", batch_size=5, backend="fb")
    assert len(instances) == 1
    db, bs, sid = instances[0]
    assert isinstance(db, DummyDB) and db.backend == "fb"
    assert bs == 5 and sid == "sid"
