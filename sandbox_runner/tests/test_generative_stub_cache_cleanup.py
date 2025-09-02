from pathlib import Path
import json
import asyncio
import os
import types
import sys

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault(
    "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None)
)
class _DummySettings:
    stub_timeout = 1.0
    stub_retries = 1
    stub_retry_base = 0.1
    stub_retry_max = 0.2
    stub_cache_max = 10
    stub_fallback_model = "none"
    sandbox_stub_model = ""
sys.modules.setdefault(
    "sandbox_settings", types.SimpleNamespace(SandboxSettings=_DummySettings)
)

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


def test_flush_caches_waits_for_save(tmp_path):
    cfg = _make_cfg(tmp_path)
    gsp._CONFIG = cfg
    key = gsp._cache_key("foo", {"a": 1})
    with gsp._CACHE_LOCK:
        gsp._CACHE[key] = {"a": 1}

    async def schedule():
        gsp._schedule_cache_persist(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(schedule())
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    gsp.flush_caches(cfg)
    assert cfg.cache_path.exists()
    with open(cfg.cache_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data == [[f"{key[0]}::{key[1]}", {"a": 1}]]
    assert not gsp._SAVE_TASKS._tasks
