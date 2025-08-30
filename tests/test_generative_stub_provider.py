import os
import asyncio
import subprocess
import sys
import importlib
import time
import pytest
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.pop("sandbox_runner", None)
import sandbox_runner.generative_stub_provider as gsp  # noqa: E402


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

    ctx = {"strategy": "synthetic", "target": target}
    first = await gsp_mod.async_generate_stubs([{"x": 0}], ctx)
    second = await gsp_mod.async_generate_stubs([{"x": 0}], ctx)

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

    ctx = {"strategy": "synthetic", "target": target}
    result = asyncio.run(gsp_mod.async_generate_stubs([{"y": 0}], ctx))
    assert result == [{"y": 2}]


def test_rule_based_fallback(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    async def loader():
        return None

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)

    def target(a: int, b: float, c: bool, d: str, e: int | None = None) -> None:
        pass

    ctx = {"strategy": "synthetic", "target": target}
    res = gsp_mod.generate_stubs([{}], ctx)
    assert res == [{"a": 1, "b": 1.0, "c": True, "d": "value", "e": None}]


def test_generation_failure_propagates(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class BrokenGen:
        def __call__(self, prompt, max_length=64, num_return_sequences=1):
            raise RuntimeError("boom")

    async def loader():
        return BrokenGen()

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(x: int) -> None:
        pass

    ctx = {"strategy": "synthetic", "target": target}
    with pytest.raises(RuntimeError):
        gsp_mod.generate_stubs([{"x": 0}], ctx)


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

    start = time.perf_counter()
    ctx = {"strategy": "synthetic", "target": target}
    await gsp_mod.async_generate_stubs([{"x": 0}], ctx)
    duration = time.perf_counter() - start
    assert duration < 0.1
    await asyncio.sleep(0.15)
    assert path.exists()
    assert dummy.calls == 1

    orig_load = gsp_mod._load_cache

    def slow_load():
        time.sleep(0.1)
        return orig_load()

    monkeypatch.setattr(gsp_mod, "_load_cache", slow_load)
    gsp_mod._CACHE = {}
    marker_time = None

    async def marker():
        nonlocal marker_time
        await asyncio.sleep(0.01)
        marker_time = time.perf_counter()

    start = time.perf_counter()
    ctx = {"strategy": "synthetic", "target": target}
    res, _ = await asyncio.gather(
        gsp_mod.async_generate_stubs([{"x": 0}], ctx),
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


def test_generated_stub_missing_fields(monkeypatch, tmp_path, caplog):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class MissingGen:
        def __call__(self, prompt, max_length=64, num_return_sequences=1):
            return [{"generated_text": "{\"x\": 1}"}]

    dummy = MissingGen()

    async def loader():
        return dummy

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(x: int, y: int) -> None:
        pass

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        gsp_mod.generate_stubs([
            {"x": 0, "y": 0}
        ], {"strategy": "synthetic", "target": target})


def test_generated_stub_bad_type(monkeypatch, tmp_path, caplog):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class TypeGen:
        def __call__(self, prompt, max_length=64, num_return_sequences=1):
            return [{"generated_text": "{\"x\": \"bad\", \"y\": 2}"}]

    dummy = TypeGen()

    async def loader():
        return dummy

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(x: int, y: int) -> None:
        pass

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        gsp_mod.generate_stubs([
            {"x": 0, "y": 0}
        ], {"strategy": "synthetic", "target": target})


def test_import_exit_no_errors(tmp_path):
    cache = tmp_path / "cache.json"
    test_file = tmp_path / "t.py"
    test_file.write_text(
        "import sandbox_runner.generative_stub_provider\n"
        "def test_dummy():\n    pass\n"
    )

    root = Path(__file__).resolve().parents[1]
    parent = root.parent
    env = os.environ.copy()
    env.update(
        {
            "MENACE_LIGHT_IMPORTS": "1",
            "SANDBOX_STUB_CACHE": str(cache),
            "PYTHONPATH": os.pathsep.join([str(parent), str(root)]),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root),
    )
    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_call_with_retry_backoff(monkeypatch):
    monkeypatch.setenv("SANDBOX_STUB_RETRIES", "3")
    monkeypatch.setenv("SANDBOX_STUB_RETRY_BASE", "1")
    monkeypatch.setenv("SANDBOX_STUB_RETRY_MAX", "8")
    gsp_mod = importlib.reload(gsp)

    sleeps: list[float] = []

    async def fake_sleep(d: float):
        sleeps.append(d)

    monkeypatch.setattr(gsp_mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(gsp_mod.random, "uniform", lambda a, b: b)

    calls = {"n": 0}

    async def func():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    result = await gsp_mod._call_with_retry(func)
    assert result == "ok"
    assert sleeps == [1, 2]


def test_cache_persist_after_failure(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    dummy = DummyGen()

    async def loader_ok():
        return dummy

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader_ok)

    calls: list[str] = []
    orig_asave = gsp_mod._asave_cache

    async def slow_asave():
        calls.append("start")
        await asyncio.sleep(0.1)
        await orig_asave()
        calls.append("end")

    monkeypatch.setattr(gsp_mod, "_asave_cache", slow_asave)

    def target(x: int) -> None:
        pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            gsp_mod.async_generate_stubs(
                [{"x": 0}], {"strategy": "synthetic", "target": target}
            )
        )

        class BrokenGen:
            def __call__(self, prompt, max_length=64, num_return_sequences=1):
                raise RuntimeError("boom")

        async def loader_fail():
            return BrokenGen()

        gsp_mod._GENERATOR = None
        gsp_mod._CACHE = {}
        monkeypatch.setattr(gsp_mod, "_aload_generator", loader_fail)

        with pytest.raises(RuntimeError):
            loop.run_until_complete(
                gsp_mod.async_generate_stubs(
                    [{"x": 1}], {"strategy": "synthetic", "target": target}
                )
            )
    finally:
        asyncio.set_event_loop(None)

    gsp_mod._atexit_save_cache()
    loop.close()

    assert path.exists()
    assert calls == ["start", "end"]
    assert not gsp_mod._SAVE_TASKS
