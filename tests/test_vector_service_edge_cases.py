import pytest
from vector_service import (
    Retriever,
    FallbackResult,
    PatchLogger,
    EmbeddingBackfill,
    EmbeddableDBMixin,
    MalformedPromptError,
)


class DummyHit:
    def __init__(self, score: float = 0.9):
        self.origin_db = "db"
        self.record_id = "1"
        self.score = score
        self.metadata = {"redacted": True}

    def to_dict(self):
        return {"origin_db": self.origin_db, "score": self.score}


class DummyRetriever:
    """Simple retriever stub returning predefined confidence."""

    def __init__(self, confidence: float = 0.9):
        self.confidence = confidence
        self.calls = 0

    def retrieve_with_confidence(self, query, top_k):
        self.calls += 1
        return [DummyHit(score=self.confidence)], self.confidence, None


def test_retriever_search_success():
    r = Retriever(retriever=DummyRetriever(confidence=0.9))
    results = r.search("query", session_id="s")
    assert results[0]["record_id"] == "1"


def test_retriever_low_confidence_fallback():
    dr = DummyRetriever(confidence=0.01)
    r = Retriever(retriever=dr, similarity_threshold=0.1)
    res = r.search("question", session_id="s")
    assert isinstance(res, FallbackResult)
    assert res.reason == "low confidence"
    assert res.confidence == 0.01
    assert dr.calls == 2


def test_retriever_malformed_query():
    r = Retriever(retriever=DummyRetriever())
    with pytest.raises(MalformedPromptError):
        r.search("", session_id="s")


class MockVectorMetricsDB:
    def __init__(self):
        self.args = None

    def update_outcome(self, session_id, pairs, contribution, patch_id, win, regret):
        self.args = (session_id, pairs, contribution, patch_id, win, regret)


def test_patch_logger_updates_vector_metrics_db():
    vm = MockVectorMetricsDB()
    pl = PatchLogger(vector_metrics=vm)
    pl.track_contributors(["db1:123", "id2"], True, patch_id="p1", session_id="s1")
    assert vm.args == (
        "s1",
        [("db1", "123"), ("", "id2")],
        0.0,
        "p1",
        True,
        False,
    )


def test_embedding_backfill_run_on_dummy_db(monkeypatch):
    calls = []

    class DummyDB(EmbeddableDBMixin):
        def __init__(self, vector_backend=None):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            calls.append((self.vector_backend, batch_size))

    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [DummyDB]
    )
    eb = EmbeddingBackfill()
    eb.run(batch_size=5, backend="vec", session_id="s")
    assert calls == [("vec", 5)]
