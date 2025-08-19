import sys
import types

class _StubCodeDB:
    def search_fts(self, q):
        return []

sys.modules.setdefault(
    "code_database",
    types.SimpleNamespace(PatchHistoryDB=object, CodeDB=_StubCodeDB),
)

import menace_cli
from vector_service import retriever as retr_mod


def test_cli_retrieve_persistent_cache(monkeypatch, tmp_path):
    cache_file = tmp_path / "cache.json"

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
            {"origin_db": "code", "record_id": 1, "score": 1.0, "text": "t"}
        ],
    )

    def retr_factory(*a, **kw):
        kw.setdefault("cache_path", str(cache_file))
        return retr_mod.Retriever(*a, **kw)

    monkeypatch.setattr(menace_cli, "Retriever", retr_factory)

    menace_cli.main(["retrieve", "q"])  # populate cache
    assert calls["n"] == 1

    menace_cli.main(["retrieve", "q"])  # reuse cache
    assert calls["n"] == 1

    menace_cli.main(["retrieve", "q", "--no-cache"])  # bypass cache
    assert calls["n"] == 2
