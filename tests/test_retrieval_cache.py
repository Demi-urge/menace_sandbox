import types

import retrieval_cache


def test_cache_hit_and_ttl_expiry(monkeypatch, tmp_path):
    path = tmp_path / "metrics.db"
    cache = retrieval_cache.RetrievalCache(path=path, ttl=1)

    # initial miss
    assert cache.get("q", ["db"]) is None

    # store and hit
    monkeypatch.setattr(
        retrieval_cache, "time", types.SimpleNamespace(time=lambda: 0.0)
    )
    cache.set("q", ["db"], [{"a": 1}])
    assert cache.get("q", ["db"]) == [{"a": 1}]

    # expire entry
    monkeypatch.setattr(
        retrieval_cache, "time", types.SimpleNamespace(time=lambda: 2.0)
    )
    assert cache.get("q", ["db"]) is None

