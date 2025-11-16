import sys
import types

from vector_service import Retriever, FallbackResult


class _DummyUR:
    def retrieve_with_confidence(self, query: str, top_k: int = 5):
        return [], 0.0, []


def _stub_code_db(monkeypatch):
    class _CodeDB:
        def search_fts(self, q, limit):
            return []

    monkeypatch.setitem(sys.modules, "code_database", types.SimpleNamespace(CodeDB=_CodeDB))


def _stub_router(monkeypatch, result):
    class _Router:
        def search_fts(self, query, dbs=None, limit=None):
            return [result]

    monkeypatch.setitem(
        sys.modules,
        "db_router",
        types.SimpleNamespace(DBRouter=_Router),
    )


def test_fts_fallback(monkeypatch):
    _stub_code_db(monkeypatch)
    result = {
        "origin_db": "memory",
        "record_id": 42,
        "score": 0.5,
        "snippet": "hello",
    }
    _stub_router(monkeypatch, result)
    retriever = Retriever(retriever=_DummyUR())
    res = retriever.search("query")
    assert isinstance(res, FallbackResult)
    assert len(res) == 1
    item = res[0]
    assert item["origin_db"] == "memory"
    assert item["score"] == 0.5
    assert item["snippet"] == "hello"


def test_fts_fallback_disabled(monkeypatch):
    _stub_code_db(monkeypatch)
    result = {
        "origin_db": "memory",
        "record_id": 42,
        "score": 0.5,
        "snippet": "hello",
    }
    _stub_router(monkeypatch, result)
    retriever = Retriever(retriever=_DummyUR(), use_fts_fallback=False)
    res = retriever.search("query")
    assert isinstance(res, FallbackResult)
    assert len(res) == 0

