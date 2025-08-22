import io
import json
import sys
import types
import importlib
from collections.abc import Sequence


class FallbackResult(Sequence):
    def __init__(self, reason, results, confidence=0.0):
        self.reason = reason
        self.results = list(results)
        self.confidence = confidence

    def __iter__(self):
        return iter(self.results)

    def __len__(self):  # pragma: no cover - simple delegation
        return len(self.results)

    def __getitem__(self, item):  # pragma: no cover - simple delegation
        return self.results[item]


def _load_cli(monkeypatch, tmp_path, retriever_impl, fts_impl=None, cache_store=None):
    sys.modules.pop("menace_cli", None)

    vsr = types.ModuleType("vector_service.retriever")
    vsr.Retriever = retriever_impl
    vsr.FallbackResult = FallbackResult
    vsr.fts_search = fts_impl or (lambda q, dbs=None, limit=None: [])
    vspkg = types.ModuleType("vector_service")
    vspkg.retriever = vsr
    vspkg.PatchLogger = object
    monkeypatch.setitem(sys.modules, "vector_service", vspkg)
    monkeypatch.setitem(sys.modules, "vector_service.retriever", vsr)

    monkeypatch.setitem(
        sys.modules, "code_database", types.SimpleNamespace(PatchHistoryDB=object)
    )

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
    cu.clear_cache = lambda: None
    cu.show_cache = lambda: {}
    cu.cache_stats = lambda: {}
    monkeypatch.setitem(sys.modules, "cache_utils", cu)

    monkeypatch.setenv("HOME", str(tmp_path))
    mod = importlib.import_module("menace_cli")
    mod._cache_calls = {"get": get_calls, "set": set_calls, "store": cache_store}
    return mod


def test_fallback_result(monkeypatch, tmp_path):
    class DummyRetriever:
        def __init__(self, cache=None):
            pass

        def search(self, query, top_k=5, dbs=None):
            return FallbackResult("oops", [])

    fts_hits = [{"origin_db": "code", "record_id": 9, "score": 0.1, "text": "fts"}]
    menace_cli = _load_cli(
        monkeypatch,
        tmp_path,
        DummyRetriever,
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
    calls = {"search": 0}

    class DummyRetriever:
        def __init__(self, cache=None):
            pass

        def search(self, query, top_k=5, dbs=None):
            calls["search"] += 1
            hit = types.SimpleNamespace(origin_db="code", record_id=1, score=1.0, text="vec")
            return [hit]

    menace_cli = _load_cli(monkeypatch, tmp_path, DummyRetriever)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["search"] == 1
    assert len(menace_cli._cache_calls["get"]) == 1
    assert len(menace_cli._cache_calls["set"]) == 1

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["search"] == 1
    assert len(menace_cli._cache_calls["get"]) == 2
    assert len(menace_cli._cache_calls["set"]) == 1


def test_rebuild_cache(monkeypatch, tmp_path):
    calls = {"search": 0}

    class DummyRetriever:
        def __init__(self, cache=None):
            pass

        def search(self, query, top_k=5, dbs=None):
            calls["search"] += 1
            hit = types.SimpleNamespace(
                origin_db="code",
                record_id=calls["search"],
                score=1.0,
                text=f"t{calls['search']}",
            )
            return [hit]

    menace_cli = _load_cli(monkeypatch, tmp_path, DummyRetriever)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json"])
    assert calls["search"] == 1
    assert len(menace_cli._cache_calls["get"]) == 1
    assert len(menace_cli._cache_calls["set"]) == 1

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    menace_cli.main(["retrieve", "q", "--json", "--rebuild-cache"])
    assert calls["search"] == 2
    assert len(menace_cli._cache_calls["get"]) == 1
    assert len(menace_cli._cache_calls["set"]) == 2
    out = json.loads(buf.getvalue())
    assert out[0]["record_id"] == 2

