import io
import json
import sys
import types
import importlib


def _load_cli(monkeypatch, tmp_path, retriever_impl, fts_impl=None):
    sys.modules.pop("menace_cli", None)
    ur = types.ModuleType("universal_retriever")
    ur.UniversalRetriever = retriever_impl
    monkeypatch.setitem(sys.modules, "universal_retriever", ur)

    vs = types.ModuleType("vector_service.retriever")
    vs.fts_search = fts_impl or (lambda q, dbs=None, limit=None: [])
    monkeypatch.setitem(sys.modules, "vector_service.retriever", vs)
    monkeypatch.setitem(
        sys.modules, "code_database", types.SimpleNamespace(PatchHistoryDB=object)
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    return importlib.import_module("menace_cli")


def test_vector_success(monkeypatch, tmp_path):
    class DummyRetriever:
        def retrieve(self, query, top_k=5, dbs=None):
            hit = types.SimpleNamespace(
                origin_db="code", record_id=1, score=0.5, text="vec"
            )
            return [hit], "s", []

    menace_cli = _load_cli(monkeypatch, tmp_path, DummyRetriever)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    res = menace_cli.main(["retrieve", "q", "--json", "--no-cache"])
    assert res == 0
    out = json.loads(buf.getvalue())
    assert out == [
        {"origin_db": "code", "record_id": 1, "score": 0.5, "snippet": "vec"}
    ]


def test_vector_fallback(monkeypatch, tmp_path):
    class DummyRetriever:
        def retrieve(self, query, top_k=5, dbs=None):
            return [], "s", []

    fts_hits = [{"origin_db": "code", "record_id": 2, "score": 0.0, "text": "fts"}]
    menace_cli = _load_cli(
        monkeypatch, tmp_path, DummyRetriever, lambda q, dbs=None, limit=None: fts_hits
    )
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    res = menace_cli.main(["retrieve", "q", "--json", "--no-cache"])
    assert res == 0
    out = json.loads(buf.getvalue())
    assert out == [
        {"origin_db": "code", "record_id": 2, "score": 0.0, "snippet": "fts"}
    ]


def test_retriever_cache(monkeypatch, tmp_path):
    calls = {"n": 0}

    class DummyRetriever:
        def retrieve(self, query, top_k=5, dbs=None):
            calls["n"] += 1
            hit = types.SimpleNamespace(origin_db="code", record_id=1, score=1.0, text="t")
            return [hit], "s", []

    menace_cli = _load_cli(monkeypatch, tmp_path, DummyRetriever)

    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["n"] == 1

    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["n"] == 1

    menace_cli.main(["retrieve", "q", "--json", "--no-cache"])
    assert calls["n"] == 2

    menace_cli.main(["retrieve", "q", "--json", "--rebuild-cache"])
    assert calls["n"] == 3

    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["n"] == 3

