import os
import time
from pathlib import Path
import types
import sys

# Provide lightweight stubs for heavy modules required by ``menace_cli``


class _StubCodeDB:
    def search_fallback(self, q):
        return []


sys.modules.setdefault(
    "code_database",
    types.SimpleNamespace(PatchHistoryDB=object, CodeDB=_StubCodeDB),
)


class _FallbackResult(list):
    def __init__(self, reason: str, hits: list):
        super().__init__(hits)
        self.reason = reason


class _VectorServiceError(Exception):
    pass


sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(
        Retriever=object, FallbackResult=_FallbackResult, VectorServiceError=_VectorServiceError
    ),
)

import menace_cli
from retrieval_cache import RetrievalCache


def test_retrieve_cache_and_invalidation(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache"  # shelve base path
    monkeypatch.setattr(menace_cli, "RetrievalCache", lambda: RetrievalCache(cache_path))

    fts_calls = {"n": 0}

    def fts_stub(q):
        fts_calls["n"] += 1
        return ["fts"]

    def fts_fail(q):
        raise AssertionError("FTS should not run")

    monkeypatch.setitem(menace_cli.FTS_HELPERS, "code", fts_stub)

    class DummyRetriever:
        def search(self, query, session_id="cli", dbs=None):
            return menace_cli.FallbackResult("err", [])

    constructs = {"n": 0}

    def retr_factory():
        constructs["n"] += 1
        return DummyRetriever()

    def retr_fail():
        raise AssertionError("Retriever should not run")

    monkeypatch.setattr(menace_cli, "Retriever", retr_factory)

    # First run: populate cache and invoke retriever + FTS
    menace_cli.main(["retrieve", "q", "--db", "code"])
    assert constructs["n"] == 1
    assert fts_calls["n"] == 1

    # Second run: should use cache only
    monkeypatch.setattr(menace_cli, "Retriever", retr_fail)
    monkeypatch.setitem(menace_cli.FTS_HELPERS, "code", fts_fail)
    menace_cli.main(["retrieve", "q", "--db", "code"])
    assert constructs["n"] == 1
    assert fts_calls["n"] == 1

    # Modify DB to invalidate cache and ensure retriever and FTS run again
    code_db = Path("code.db")
    if code_db.exists():
        now = code_db.stat().st_mtime
        os.utime(code_db, (now + 5, now + 5))
    monkeypatch.setattr(menace_cli, "Retriever", retr_factory)
    monkeypatch.setitem(menace_cli.FTS_HELPERS, "code", fts_stub)
    menace_cli.main(["retrieve", "q", "--db", "code"])
    assert constructs["n"] == 2
    assert fts_calls["n"] == 2
