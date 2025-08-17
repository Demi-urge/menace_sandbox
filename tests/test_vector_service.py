import pytest
from vector_service import (
    Retriever,
    ContextBuilder,
    PatchLogger,
    EmbeddingBackfill,
    ErrorResult,
    VectorServiceError,
    RateLimitError,
    MalformedPromptError,
    FallbackResult,
)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class DummyHit:
    def __init__(self, score=0.9, metadata=None):
        self.origin_db = "db"
        self.record_id = "1"
        self.score = score
        self.metadata = metadata or {}


class DummyRetrieverSuccess:
    def retrieve_with_confidence(self, query, top_k):
        return [DummyHit()], 0.9, None


class DummyRetrieverRateLimit:
    def retrieve(self, query, top_k):
        raise Exception("429 rate limit")


class DummyRetrieverNoHits:
    def retrieve_with_confidence(self, query, top_k):
        return [], 0.0, None


def test_retriever_search_success_returns_results():
    r = Retriever(retriever=DummyRetrieverSuccess())
    results, conf = r.search("query", include_confidence=True)
    assert results[0]["record_id"] == "1"
    assert conf == pytest.approx(0.9)


def test_retriever_search_rate_limit_raises_error():
    r = Retriever(retriever=DummyRetrieverRateLimit())
    with pytest.raises(RateLimitError):
        r.search("q")


def test_retriever_search_fallback_when_no_hits():
    r = Retriever(retriever=DummyRetrieverNoHits())
    res = r.search("q")
    assert isinstance(res, FallbackResult)
    assert res.reason == "no results"


def test_retriever_search_malformed_prompt():
    r = Retriever(retriever=DummyRetrieverSuccess())
    with pytest.raises(MalformedPromptError):
        r.search("  ")


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class DummyBuilder:
    def __init__(self, exc=None):
        self.exc = exc

    def build_context(self, task_description, **kwargs):
        if self.exc:
            raise self.exc
        return "ok"


def test_context_builder_success():
    cb = ContextBuilder(builder=DummyBuilder())
    assert cb.build("task") == "ok"


def test_context_builder_returns_value_error_object():
    cb = ContextBuilder(builder=DummyBuilder(exc=ValueError("bad")))
    res = cb.build("task")
    assert isinstance(res, ErrorResult)
    assert res.error == "value_error"


def test_context_builder_returns_rate_limit_object():
    cb = ContextBuilder(builder=DummyBuilder(exc=Exception("429 rate limit")))
    res = cb.build("task")
    assert isinstance(res, ErrorResult)
    assert res.error == "rate_limited"


def test_context_builder_wraps_other_errors():
    cb = ContextBuilder(builder=DummyBuilder(exc=RuntimeError("boom")))
    with pytest.raises(VectorServiceError):
        cb.build("task")


# ---------------------------------------------------------------------------
# PatchLogger
# ---------------------------------------------------------------------------


class DummyMetricsDB:
    def __init__(self, fail=False):
        self.fail = fail
        self.called = False
        self.args = None

    def log_patch_outcome(self, patch_id, result, pairs, session_id=""):
        if self.fail:
            raise Exception("db failure")
        self.called = True
        self.args = (patch_id, result, pairs, session_id)


class DummyVectorMetricsDB:
    def __init__(self):
        self.called = False

    def update_outcome(self, session_id, pairs, contribution, patch_id, win, regret):
        self.called = True


def test_patch_logger_metrics_db_success():
    db = DummyMetricsDB()
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["a:b"], True, patch_id="p", session_id="s")
    assert db.called
    assert db.args[2] == [("a", "b")]


def test_patch_logger_vector_metrics_db_success():
    vm = DummyVectorMetricsDB()
    pl = PatchLogger(vector_metrics=vm)
    pl.track_contributors(["x"], False, session_id="s")
    assert vm.called


def test_patch_logger_metrics_db_error_raises():
    pl = PatchLogger(metrics_db=DummyMetricsDB(fail=True))
    with pytest.raises(VectorServiceError):
        pl.track_contributors(["a"], True)


def test_patch_logger_malformed_vector_ids():
    pl = PatchLogger(metrics_db=DummyMetricsDB())
    with pytest.raises(MalformedPromptError):
        pl.track_contributors([123], True)


# ---------------------------------------------------------------------------
# EmbeddingBackfill
# ---------------------------------------------------------------------------


class DummyDB:
    def __init__(self, vector_backend="annoy"):
        self.batch = None

    def backfill_embeddings(self, batch_size=0):
        self.batch = batch_size


class ErrorDB(DummyDB):
    def backfill_embeddings(self, batch_size=0):
        raise RuntimeError("boom")


def test_embedding_backfill_run_processes_databases(monkeypatch):
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self: [DummyDB]
    )
    called = []

    def fake_process(self, db, *, batch_size, session_id=""):
        called.append((db.__class__, batch_size, session_id))

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)
    eb = EmbeddingBackfill(batch_size=7)
    eb.run(session_id="sid")
    assert called == [(DummyDB, 7, "sid")]


def test_embedding_backfill_run_continues_on_error(monkeypatch):
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self: [ErrorDB, DummyDB]
    )
    order = []

    def fake_process(self, db, *, batch_size, session_id=""):
        order.append(db.__class__)
        if isinstance(db, ErrorDB):
            raise RuntimeError("fail")

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)
    eb = EmbeddingBackfill()
    eb.run()
    assert order == [ErrorDB, DummyDB]
