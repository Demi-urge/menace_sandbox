import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import asyncio
import sandbox_runner.generative_stub_provider as gsp
import importlib
import time
import pytest
import logging

class DummyGen:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, max_length=64, num_return_sequences=1):
        self.calls += 1
        return [{"generated_text": "{\"x\": 1}"}]

def test_generate_stubs_cache(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    dummy = DummyGen()
    async def loader():
        return dummy
    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(x: int) -> None:
        pass

    first = gsp_mod.generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})
    second = gsp_mod.generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})

    assert first == [{"x": 1}]
    assert second == [{"x": 1}]
    assert dummy.calls == 1


@pytest.mark.asyncio
async def test_async_generate_stubs_cache(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    dummy = DummyGen()

    async def loader():
        return dummy

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(x: int) -> None:
        pass

    first = await gsp_mod.async_generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})
    second = await gsp_mod.async_generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})

    assert first == [{"x": 1}]
    assert second == [{"x": 1}]
    assert dummy.calls == 1


def test_async_generate_returns_json(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class AsyncGen:
        async def __call__(self, prompt, max_length=64, num_return_sequences=1):
            return [{"generated_text": "{\"y\": 2}"}]

    dummy = AsyncGen()
    async def loader():
        return dummy
    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(y: int) -> None:
        pass

    result = asyncio.run(gsp_mod.async_generate_stubs([{"y": 0}], {"strategy": "synthetic", "target": target}))
    assert result == [{"y": 2}]


@pytest.mark.asyncio
async def test_async_cache_file(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    dummy = DummyGen()
    async def loader():
        return dummy
    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)

    def target(x: int) -> None:
        pass

    orig_save = gsp_mod._save_cache
    def slow_save():
        time.sleep(0.1)
        orig_save()

    monkeypatch.setattr(gsp_mod, "_save_cache", slow_save)
    gsp_mod._CACHE = {}

    marker_time = None
    async def marker():
        nonlocal marker_time
        await asyncio.sleep(0.01)
        marker_time = time.perf_counter()

    start = time.perf_counter()
    await asyncio.gather(
        gsp_mod.async_generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target}),
        marker(),
    )
    assert marker_time - start < 0.1
    assert path.exists()
    assert dummy.calls == 1

    orig_load = gsp_mod._load_cache
    def slow_load():
        time.sleep(0.1)
        return orig_load()

    monkeypatch.setattr(gsp_mod, "_load_cache", slow_load)
    gsp_mod._CACHE = {}
    marker_time = None
    start = time.perf_counter()
    res, _ = await asyncio.gather(
        gsp_mod.async_generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target}),
        marker(),
    )
    assert marker_time - start < 0.1
    assert res == [{"x": 1}]
    assert dummy.calls == 1


def test_atexit_cache_failure_logged(monkeypatch, caplog):
    import atexit

    # prevent side effects from the module's atexit handler
    try:
        atexit.unregister(gsp._atexit_save_cache)
    except Exception:
        pass
    monkeypatch.setattr(atexit, "register", lambda func: None)

    gsp_mod = importlib.reload(gsp)

    def bad_save() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(gsp_mod, "_save_cache", bad_save)

    caplog.set_level(logging.ERROR)
    gsp_mod._atexit_save_cache()

    assert "cache save failed" in caplog.text
