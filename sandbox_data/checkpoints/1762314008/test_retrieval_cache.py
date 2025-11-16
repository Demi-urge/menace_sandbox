import importlib
import types

import db_router
import retrieval_cache


def test_cache_hit_and_ttl_expiry(monkeypatch, tmp_path):
    db_router.init_db_router(
        "test", local_db_path=str(tmp_path / "l.db"), shared_db_path=str(tmp_path / "s.db")
    )
    importlib.reload(retrieval_cache)
    cache = retrieval_cache.RetrievalCache(ttl=1)

    # initial miss
    assert cache.get("q", ["db"]) is None

    # store and hit
    monkeypatch.setattr(
        retrieval_cache, "time", types.SimpleNamespace(time=lambda: 0.0)
    )
    cache.set("q", ["db"], [{"a": 1}])
    assert cache.get("q", ["db"]) == [{"a": 1}]
    conn = db_router.GLOBAL_ROUTER.get_connection("retrieval_cache")
    assert (
        conn.execute("SELECT COUNT(*) FROM retrieval_cache").fetchone()[0] == 1
    )

    # expire entry
    monkeypatch.setattr(
        retrieval_cache, "time", types.SimpleNamespace(time=lambda: 2.0)
    )
    assert cache.get("q", ["db"]) is None

