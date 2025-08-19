import pytest
from vector_service import (
    Retriever,
    PatchLogger,
    EmbeddingBackfill,
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
        base_meta = {"redacted": True}
        if metadata:
            base_meta.update(metadata)
        self.metadata = base_meta


class DummyRetrieverSuccess:
    def retrieve_with_confidence(self, query, top_k):
        return [DummyHit()], 0.9, None


class DummyRetrieverRateLimit:
    def retrieve(self, query, top_k):
        raise Exception("429 rate limit")


class DummyRetrieverNoHits:
    def retrieve_with_confidence(self, query, top_k):
        return [], 0.0, None


class DummyRetrieverLowConfidence:
    def retrieve_with_confidence(self, query, top_k):
        return [DummyHit(score=0.05)], 0.05, None


def test_retriever_search_success_returns_results():
    r = Retriever(retriever=DummyRetrieverSuccess())
    results = r.search("query", session_id="s")
    assert results[0]["record_id"] == "1"


def test_retriever_search_rate_limit_raises_error():
    r = Retriever(retriever=DummyRetrieverRateLimit())
    with pytest.raises(RateLimitError):
        r.search("q", session_id="s")


def test_retriever_search_fallback_when_no_hits():
    r = Retriever(retriever=DummyRetrieverNoHits())
    res = r.search("q", session_id="s")
    assert isinstance(res, FallbackResult)
    assert res.reason == "no results"


def test_retriever_search_low_confidence_fallback():
    r = Retriever(retriever=DummyRetrieverLowConfidence())
    res = r.search("q", session_id="s")
    assert isinstance(res, FallbackResult)
    assert res.reason == "low confidence"


def test_retriever_search_malformed_prompt():
    r = Retriever(retriever=DummyRetrieverSuccess())
    with pytest.raises(MalformedPromptError):
        r.search("  ", session_id="s")


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


class DummyPatchDB:
    def __init__(self):
        self.called = False
        self.args = None

    def record_vector_metrics(self, session_id, pairs, patch_id, contribution, win, regret):
        self.called = True
        self.args = (session_id, pairs, patch_id, contribution, win, regret)


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


def test_patch_logger_patch_db_success():
    pdb = DummyPatchDB()
    pl = PatchLogger(patch_db=pdb)
    pl.track_contributors(["a:b"], True, patch_id="7", session_id="sid")
    assert pdb.called
    assert pdb.args[2] == 7


def test_patch_logger_metrics_db_failure_ignored():
    db = DummyMetricsDB(fail=True)
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["a:b"], True, patch_id="p", session_id="s")


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


class InitFallbackDB:
    instances = []

    def __init__(self, vector_backend=""):
        if vector_backend:
            raise TypeError("no backend")
        self.__class__.instances.append(self)
        self.processed = False

    def backfill_embeddings(self, batch_size=0):
        self.processed = True
        self.batch = batch_size


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
    eb.run(session_id="s")
    assert order == [ErrorDB, DummyDB]


def test_embedding_backfill_instantiation_fallback(monkeypatch):
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self: [InitFallbackDB]
    )
    eb = EmbeddingBackfill(batch_size=3)
    eb.run()
    assert InitFallbackDB.instances and InitFallbackDB.instances[0].processed
    assert InitFallbackDB.instances[0].batch == 3


def test_embedding_backfill_filters_by_db(monkeypatch):
    class WorkflowDB(DummyDB):
        pass

    class OtherDB(DummyDB):
        pass

    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self: [WorkflowDB, OtherDB]
    )
    called = []

    def fake_process(self, db, *, batch_size, session_id=""):
        called.append(db.__class__)

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)
    eb = EmbeddingBackfill()
    eb.run(db="workflows")
    assert called == [WorkflowDB]
