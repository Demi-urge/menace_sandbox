import io
import json
import sys
import types
import importlib

class _FallbackResult(list):
    def __init__(self, reason: str, hits: list):
        super().__init__(hits)
        self.reason = reason

class _VectorServiceError(Exception):
    pass

def _load_cli(monkeypatch, retriever_impl, fts_impl=None):
    sys.modules.pop("menace_cli", None)
    mod = types.ModuleType("vector_service.retriever")
    mod.Retriever = retriever_impl
    mod.FallbackResult = _FallbackResult
    mod.VectorServiceError = _VectorServiceError
    mod.fts_search = fts_impl or (lambda q, dbs=None, limit=None: [])
    monkeypatch.setitem(sys.modules, "vector_service.retriever", mod)
    monkeypatch.setitem(sys.modules, "code_database", types.SimpleNamespace(PatchHistoryDB=object))
    return importlib.import_module("menace_cli")


def test_vector_success(monkeypatch):
    class DummyRetriever:
        def __init__(self, *a, **kw):
            pass
        def search(self, query, session_id="", top_k=5, dbs=None):
            return [{"origin_db": "code", "record_id": 1, "score": 0.5, "text": "vec"}]
        def save_cache(self):
            pass
    menace_cli = _load_cli(monkeypatch, DummyRetriever)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    res = menace_cli.main(["retrieve", "q", "--json"])
    assert res == 0
    out = json.loads(buf.getvalue())
    assert out == [{"origin_db": "code", "record_id": 1, "score": 0.5, "snippet": "vec"}]


def test_vector_fallback(monkeypatch):
    class DummyRetriever:
        def __init__(self, *a, **kw):
            pass
        def search(self, query, session_id="", top_k=5, dbs=None):
            return _FallbackResult("no results", [
                {"origin_db": "heuristic", "record_id": None, "score": 0.0, "text": "h"}
            ])
        def save_cache(self):
            pass
    fts_hits = [{"origin_db": "code", "record_id": 2, "score": 0.0, "snippet": "fts"}]
    menace_cli = _load_cli(monkeypatch, DummyRetriever, lambda q, dbs=None, limit=None: fts_hits)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    res = menace_cli.main(["retrieve", "q", "--json"])
    assert res == 0
    out = json.loads(buf.getvalue())
    assert out == [
        {"origin_db": "heuristic", "record_id": None, "score": 0.0, "snippet": "h"},
        {"origin_db": "code", "record_id": 2, "score": 0.0, "snippet": "fts"},
    ]


def test_retriever_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from vector_service import retriever as retr_mod

    calls = {"n": 0}

    class _Backend:
        def retrieve(self, query, top_k=5, dbs=None):
            calls["n"] += 1
            hit = types.SimpleNamespace(
                metadata={"redacted": True},
                score=1.0,
                text="t",
                record_id=1,
                origin_db="code",
            )
            return [hit], None, None

    monkeypatch.setattr(retr_mod.Retriever, "_get_retriever", lambda self: _Backend())
    monkeypatch.setattr(
        retr_mod.Retriever,
        "_parse_hits",
        lambda self, hits: [
            {"origin_db": "code", "record_id": 1, "score": 1.0, "snippet": "t"}
        ],
    )

    monkeypatch.setitem(sys.modules, "code_database", types.SimpleNamespace(PatchHistoryDB=object))
    sys.modules.pop("menace_cli", None)
    menace_cli = importlib.import_module("menace_cli")

    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["n"] == 1

    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["n"] == 1

    menace_cli.main(["retrieve", "q", "--no-cache", "--json"])
    assert calls["n"] == 2
