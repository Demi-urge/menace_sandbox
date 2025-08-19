import io
import json
import sys
import types
import importlib


def _load_cli(monkeypatch, tmp_path, retriever_impl, fts_impl=None, cache_store=None):
    sys.modules.pop("menace_cli", None)

    # stub universal retriever used by menace_cli when vector retriever import fails
    ur = types.ModuleType("universal_retriever")
    ur.UniversalRetriever = retriever_impl
    monkeypatch.setitem(sys.modules, "universal_retriever", ur)

    # stub vector_service.retriever with only fts_search so the import of
    # Retriever fails and menace_cli falls back to universal_retriever
    vs = types.ModuleType("vector_service.retriever")
    vs.fts_search = fts_impl or (lambda q, dbs=None, limit=None: [])
    monkeypatch.setitem(sys.modules, "vector_service.retriever", vs)

    # minimal vector_service package for other imports performed by menace_cli
    vspkg = types.ModuleType("vector_service")
    vspkg.PatchLogger = object
    monkeypatch.setitem(sys.modules, "vector_service", vspkg)

    # stub code_database required by menace_cli imports
    monkeypatch.setitem(
        sys.modules, "code_database", types.SimpleNamespace(PatchHistoryDB=object)
    )

    # cache_utils stub with call tracking so tests can assert interactions
    cache_store = {} if cache_store is None else cache_store
    get_calls: list[tuple] = []
    set_calls: list[tuple] = []

    def get_cached_chain(query, dbs):
        get_calls.append((query, dbs))
        return cache_store.get((query, tuple(dbs) if dbs else None))

    def set_cached_chain(query, dbs, results):
        set_calls.append((query, dbs, results))
        cache_store[(query, tuple(dbs) if dbs else None)] = results

    cu = types.ModuleType("cache_utils")
    cu.get_cached_chain = get_cached_chain
    cu.set_cached_chain = set_cached_chain
    cu._get_cache = lambda: object()
    monkeypatch.setitem(sys.modules, "cache_utils", cu)

    monkeypatch.setenv("HOME", str(tmp_path))
    mod = importlib.import_module("menace_cli")
    mod._cache_calls = {"get": get_calls, "set": set_calls, "store": cache_store}
    return mod


def test_fallback_on_exception(monkeypatch, tmp_path):
    class BrokenRetriever:
        def retrieve(self, query, top_k=5, dbs=None):  # pragma: no cover - intentionally raises
            raise RuntimeError("boom")

    fts_hits = [{"origin_db": "code", "record_id": 9, "score": 0.1, "text": "fts"}]
    menace_cli = _load_cli(
        monkeypatch,
        tmp_path,
        BrokenRetriever,
        lambda q, dbs=None, limit=None: fts_hits,
    )
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    res = menace_cli.main(["retrieve", "q", "--json", "--no-cache"])
    assert res == 0
    out = json.loads(buf.getvalue())
    assert out == [
        {"origin_db": "code", "record_id": 9, "score": 0.1, "snippet": "fts"}
    ]


def test_cache_utils_called(monkeypatch, tmp_path):
    calls = {"retrieve": 0}

    class DummyRetriever:
        def retrieve(self, query, top_k=5, dbs=None):
            calls["retrieve"] += 1
            hit = types.SimpleNamespace(origin_db="code", record_id=1, score=1.0, text="vec")
            return [hit], "s", []

    menace_cli = _load_cli(monkeypatch, tmp_path, DummyRetriever)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["retrieve"] == 1
    assert len(menace_cli._cache_calls["get"]) == 1
    assert len(menace_cli._cache_calls["set"]) == 1

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json"])
    # second call should use cache and not call retriever again
    assert calls["retrieve"] == 1
    assert len(menace_cli._cache_calls["get"]) == 2
    assert len(menace_cli._cache_calls["set"]) == 1


def test_rebuild_cache(monkeypatch, tmp_path):
    calls = {"retrieve": 0}

    class DummyRetriever:
        def retrieve(self, query, top_k=5, dbs=None):
            calls["retrieve"] += 1
            hit = types.SimpleNamespace(
                origin_db="code",
                record_id=calls["retrieve"],
                score=1.0,
                text=f"t{calls['retrieve']}",
            )
            return [hit], "s", []

    menace_cli = _load_cli(monkeypatch, tmp_path, DummyRetriever)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["retrieve"] == 1
    assert len(menace_cli._cache_calls["get"]) == 1
    assert len(menace_cli._cache_calls["set"]) == 1

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json", "--rebuild-cache"])
    # retrieval should run again and get_cached_chain should not be consulted
    assert calls["retrieve"] == 2
    assert len(menace_cli._cache_calls["get"]) == 1
    assert len(menace_cli._cache_calls["set"]) == 2
    out = json.loads(buf.getvalue())
    assert out[0]["record_id"] == 2
