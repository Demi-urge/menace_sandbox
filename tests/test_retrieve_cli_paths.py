import json
import sys
import types
import io


class _FallbackResult(list):
    def __init__(self, reason: str, hits: list):
        super().__init__(hits)
        self.reason = reason


class _VectorServiceError(Exception):
    pass


def _load_cli(monkeypatch):
    sys.modules.pop("menace_cli", None)
    code_mod = types.SimpleNamespace(
        PatchHistoryDB=object,
        CodeDB=type("C", (), {"search_fts": lambda self, q: []}),
    )
    vs_mod = types.SimpleNamespace(
        Retriever=object,
        FallbackResult=_FallbackResult,
        VectorServiceError=_VectorServiceError,
    )
    monkeypatch.setitem(sys.modules, "code_database", code_mod)
    monkeypatch.setitem(sys.modules, "vector_service", vs_mod)
    import importlib

    return importlib.import_module("menace_cli")


class DummyCache:
    def get(self, *a, **kw):
        return None

    def set(self, *a, **kw):
        pass


def test_retrieve_vector_path(monkeypatch):
    menace_cli = _load_cli(monkeypatch)

    calls = {}

    class DummyRetriever:
        def search(self, query, session_id="", dbs=None):
            calls["args"] = (query, session_id, dbs)
            return [{"origin_db": "code", "record_id": 1, "score": 0.5, "text": "vec"}]

    monkeypatch.setattr(menace_cli, "Retriever", lambda: DummyRetriever())
    monkeypatch.setattr(menace_cli, "RetrievalCache", lambda: DummyCache())
    monkeypatch.setattr(menace_cli, "get_db_mtimes", lambda dbs: {})
    monkeypatch.setattr(menace_cli, "uuid4", lambda: types.SimpleNamespace(hex="sess"))

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    res = menace_cli.main(["retrieve", "q"])
    assert res == 0
    assert calls["args"][1] == "sess"
    out = json.loads(buf.getvalue())
    assert out == [
        {"origin_db": "code", "record_id": 1, "score": 0.5, "snippet": "vec"}
    ]


def test_retrieve_fallback_path(monkeypatch):
    menace_cli = _load_cli(monkeypatch)

    class DummyRetriever:
        def search(self, query, session_id="", dbs=None):
            return menace_cli.FallbackResult(
                "err", [{"origin_db": "heuristic", "record_id": None, "score": 0.0, "text": "h"}]
            )

    monkeypatch.setattr(menace_cli, "Retriever", lambda: DummyRetriever())
    monkeypatch.setattr(menace_cli, "RetrievalCache", lambda: DummyCache())
    monkeypatch.setattr(menace_cli, "get_db_mtimes", lambda dbs: {})
    monkeypatch.setitem(
        menace_cli.FTS_HELPERS,
        "code",
        lambda q: [{"id": 2, "code": "fts"}],
    )

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    res = menace_cli.main(["retrieve", "q", "--db", "code"])
    assert res == 0
    out = json.loads(buf.getvalue())
    assert out == [
        {"origin_db": "heuristic", "record_id": None, "score": 0.0, "snippet": "h"},
        {"origin_db": "code", "record_id": 2, "score": 0.0, "snippet": "fts"},
    ]
