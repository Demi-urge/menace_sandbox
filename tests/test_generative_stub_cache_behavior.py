import os
import sys
import types
import asyncio
import json
from pathlib import Path
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault(
    "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None)
)

class _DummySettings:
    menace_light_imports = True
    stub_timeout = 1.0
    stub_save_timeout = 1.0
    stub_retries = 1
    stub_retry_base = 0.1
    stub_retry_max = 0.2
    stub_cache_max = 10
    stub_fallback_model = "none"
    sandbox_stub_model = ""

sys.modules.setdefault(
    "sandbox_settings", types.SimpleNamespace(SandboxSettings=_DummySettings)
)
sys.modules.setdefault(
    "model_registry", types.SimpleNamespace(get_client=lambda *a, **k: object())
)
sys.modules.setdefault(
    "llm_interface", types.SimpleNamespace(Prompt=str)
)

import sandbox_runner.generative_stub_provider as gsp


def _make_cfg(tmp_path: Path) -> gsp.StubProviderConfig:
    return gsp.StubProviderConfig(
        timeout=1.0,
        retries=1,
        retry_base=0.1,
        retry_max=0.2,
        cache_max=10,
        cache_path=tmp_path / "cache.json",
        fallback_model="none",
        save_timeout=1.0,
    )


def _reset(cfg: gsp.StubProviderConfig) -> None:
    gsp._CONFIG = cfg
    with gsp._CACHE_LOCK:
        gsp._CACHE.clear()
        gsp._TARGET_STATS.clear()
    gsp._SAVE_TASKS = gsp._SaveTaskManager()
    gsp.cleanup_cache_files(cfg)


def _target(x: int) -> None:
    return None


def test_stub_cache_persistence(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _reset(cfg)

    class Gen:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            return types.SimpleNamespace(text=json.dumps({"x": 1}))

    gen = Gen()

    async def load():
        return gen

    monkeypatch.setattr(gsp, "_aload_generator", load)

    builder = types.SimpleNamespace(build_prompt=lambda q, *, intent_metadata=None, **k: q)

    async def first():
        res = await gsp.async_generate_stubs(
            [{}], {"target": _target}, cfg, context_builder=builder
        )
        assert res == [{"x": 1}]

    asyncio.run(first())
    assert gen.calls == 1
    gsp.flush_caches(cfg)
    with gsp._CACHE_LOCK:
        gsp._CACHE.clear()

    async def second():
        res = await gsp.async_generate_stubs(
            [{}], {"target": _target}, cfg, context_builder=builder
        )
        assert res == [{"x": 1}]

    asyncio.run(second())
    assert gen.calls == 1
    gsp.flush_caches(cfg)
    gsp.cleanup_cache_files(cfg)


def test_concurrent_cache_access(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _reset(cfg)

    class Gen:
        def __init__(self):
            self.calls = 0

        async def generate(self, prompt):
            self.calls += 1
            await asyncio.sleep(0.05)
            return types.SimpleNamespace(text=json.dumps({"x": 1}))

    gen = Gen()

    async def load():
        return gen

    monkeypatch.setattr(gsp, "_aload_generator", load)

    async def invoke():
        return await gsp.async_generate_stubs(
            [{}], {"target": _target}, cfg, context_builder=builder
        )

    async def run():
        return await asyncio.gather(invoke(), invoke())

    res1, res2 = asyncio.run(run())
    assert res1 == res2 == [{"x": 1}]
    with gsp._CACHE_LOCK:
        assert len(gsp._CACHE) == 1
    gsp.flush_caches(cfg)
    gsp.cleanup_cache_files(cfg)


def test_signature_validation(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _reset(cfg)

    class Gen:
        def generate(self, prompt):
            return types.SimpleNamespace(text=json.dumps({"x": "bad"}))

    async def load():
        return Gen()

    monkeypatch.setattr(gsp, "_aload_generator", load)

    with pytest.raises(RuntimeError):
        asyncio.run(
            gsp.async_generate_stubs(
                [{}], {"target": _target}, cfg, context_builder=builder
            )
        )
    gsp.cleanup_cache_files(cfg)
