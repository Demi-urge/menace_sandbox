from pathlib import Path
import json

from sandbox_runner import generative_stub_provider as gsp


def _make_cfg(tmp_path: Path) -> gsp.StubProviderConfig:
    return gsp.StubProviderConfig(
        timeout=1.0,
        retries=1,
        retry_base=0.1,
        retry_max=0.2,
        cache_max=10,
        cache_path=tmp_path / "cache.json",
        fallback_model="none",
    )


def test_flush_and_cleanup_cache(tmp_path):
    cfg = _make_cfg(tmp_path)
    key = gsp._cache_key("foo", {"a": 1})
    with gsp._CACHE_LOCK:
        gsp._CACHE[key] = {"a": 1}
    gsp.flush_caches(cfg)
    assert not gsp._CACHE
    assert not gsp._TARGET_STATS
    assert cfg.cache_path.exists()
    with open(cfg.cache_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data == [[f"{key[0]}::{key[1]}", {"a": 1}]]
    gsp.cleanup_cache_files(cfg)
    assert not cfg.cache_path.exists()
