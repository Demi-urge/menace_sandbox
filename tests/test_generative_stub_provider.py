import os
import asyncio
import subprocess
import sys
import importlib
import time
import types
import pytest
import logging
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.pop("sandbox_runner", None)
# stub out heavy test harness dependency to avoid package import issues
th_stub = types.ModuleType("sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: None
th_stub.TestHarnessResult = object
sys.modules.setdefault("sandbox_runner.test_harness", th_stub)
import sandbox_runner.generative_stub_provider as gsp  # noqa: E402


class DummyGen:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None):
        self.calls += 1
        return [{"generated_text": "{\"x\": 1}"}]


class DummyBuilder:
    def build_prompt(self, query, *, intent_metadata=None, **kwargs):
        return query

    def build_context(self, query, *, intent_metadata=None, **kwargs):
        return {}


gsp.build_prompt = lambda query, *, intent_metadata=None, context_builder, **kwargs: context_builder.build_prompt(  # type: ignore[assignment]
    query, intent_metadata=intent_metadata, **kwargs
)

    def build_context(self, query, *, intent_metadata=None, **kwargs):
        return {}


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

    builder = DummyBuilder()
    first = gsp_mod.generate_stubs(
        [{"x": 0}], {"strategy": "synthetic", "target": target}, context_builder=builder
    )
    second = gsp_mod.generate_stubs(
        [{"x": 0}], {"strategy": "synthetic", "target": target}, context_builder=builder
    )

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
    builder = DummyBuilder()
    first = await gsp_mod.async_generate_stubs(
        [{"x": 0}], ctx, context_builder=builder
    )
    second = await gsp_mod.async_generate_stubs(
        [{"x": 0}], ctx, context_builder=builder
    )

    assert first == [{"x": 1}]
    assert second == [{"x": 1}]
    assert dummy.calls == 1


def test_async_generate_returns_json(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class AsyncGen:
        async def __call__(
            self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None
        ):
            return [{"generated_text": "{\"y\": 2}"}]

    dummy = AsyncGen()

    async def loader():
        return dummy
    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(y: int) -> None:
        pass

    ctx = {"strategy": "synthetic", "target": target}
    builder = DummyBuilder()
    result = asyncio.run(
        gsp_mod.async_generate_stubs([{"y": 0}], ctx, context_builder=builder)
    )
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
    res = gsp_mod.generate_stubs(
        [{}], ctx, context_builder=DummyBuilder()
    )[0]
    assert gsp_mod._type_matches(res["a"], int)
    assert gsp_mod._type_matches(res["b"], float)
    assert gsp_mod._type_matches(res["c"], bool)
    assert gsp_mod._type_matches(res["d"], str)
    assert res["e"] is None


def test_generation_failure_propagates(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class BrokenGen:
        def __call__(
            self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None
        ):
            raise RuntimeError("boom")

    async def loader():
        return BrokenGen()

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = {}

    def target(x: int) -> None:
        pass

    ctx = {"strategy": "synthetic", "target": target}
    with pytest.raises(RuntimeError):
        gsp_mod.generate_stubs([{"x": 0}], ctx, context_builder=DummyBuilder())


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
    builder = DummyBuilder()
    await gsp_mod.async_generate_stubs([{"x": 0}], ctx, context_builder=builder)
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
        gsp_mod.async_generate_stubs(
            [{"x": 0}], ctx, context_builder=builder
        ),
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
        def __call__(
            self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None
        ):
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
        gsp_mod.generate_stubs(
            [{"x": 0, "y": 0}],
            {"strategy": "synthetic", "target": target},
            context_builder=DummyBuilder(),
        )


def test_generated_stub_bad_type(monkeypatch, tmp_path, caplog):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class TypeGen:
        def __call__(
            self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None
        ):
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
        gsp_mod.generate_stubs(
            [{"x": 0, "y": 0}],
            {"strategy": "synthetic", "target": target},
            context_builder=DummyBuilder(),
        )


def test_import_exit_no_errors(tmp_path):
    cache = tmp_path / "cache.json"
    test_file = tmp_path / "t.py"  # path-ignore
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
    orig_asave = gsp_mod._async_save_cache

    async def slow_asave():
        calls.append("start")
        await asyncio.sleep(0.1)
        await orig_asave()
        calls.append("end")

    monkeypatch.setattr(gsp_mod, "_async_save_cache", slow_asave)

    def target(x: int) -> None:
        pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    builder = DummyBuilder()
    try:
        loop.run_until_complete(
            gsp_mod.async_generate_stubs(
                [{"x": 0}], {"strategy": "synthetic", "target": target}, context_builder=builder
            )
        )

        class BrokenGen:
            def __call__(
                self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None
            ):
                raise RuntimeError("boom")

        async def loader_fail():
            return BrokenGen()

        gsp_mod._GENERATOR = None
        gsp_mod._CACHE = {}
        monkeypatch.setattr(gsp_mod, "_aload_generator", loader_fail)

        with pytest.raises(RuntimeError):
            loop.run_until_complete(
                gsp_mod.async_generate_stubs(
                    [{"x": 1}], {"strategy": "synthetic", "target": target}, context_builder=builder
                )
            )
    finally:
        gsp_mod._atexit_save_cache()
        loop.close()
        asyncio.set_event_loop(None)

    assert path.exists()
    assert calls == ["start", "end"]
    assert not gsp_mod._SAVE_TASKS._tasks


@pytest.mark.asyncio
async def test_concurrent_stub_generation(monkeypatch, tmp_path):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    class SlowGen:
        def __init__(self):
            self.calls = 0

        async def __call__(
            self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None
        ):
            self.calls += 1
            await asyncio.sleep(0.01)
            return [{"generated_text": "{\"x\": 1}"}]

    dummy = SlowGen()

    async def loader():
        return dummy

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    gsp_mod._CACHE = OrderedDict()

    def target(x: int) -> None:
        pass

    ctx = {"strategy": "synthetic", "target": target}
    builder = DummyBuilder()
    results = await asyncio.gather(
        *(gsp_mod.async_generate_stubs([{"x": 0}], ctx, context_builder=builder) for _ in range(5))
    )

    assert all(res == [{"x": 1}] for res in results)
    with gsp_mod._CACHE_LOCK:
        assert len(gsp_mod._CACHE) == 1
    assert dummy.calls >= 1


def test_stub_generation_aborts_on_model_error(monkeypatch):
    gsp_mod = importlib.reload(gsp)

    async def loader():
        raise gsp_mod.ModelLoadError("no model")

    monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
    ctx = {"strategy": "synthetic", "target": None}
    assert gsp_mod.generate_stubs(
        [{"x": 1}], ctx, context_builder=DummyBuilder()
    ) == [{"x": 1}]


def test_env_reload_updates_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setenv("SANDBOX_STUB_RETRY_BASE", "0.001")
    monkeypatch.setenv("SANDBOX_STUB_RETRY_MAX", "0.001")

    gsp_mod = importlib.reload(gsp)

    class FailGen:
        def __init__(self):
            self.calls = 0

        def generate(self, *a, **k):
            self.calls += 1
            raise RuntimeError("boom")

    def run_expected_calls():
        gen = FailGen()

        async def loader():
            return gen

        monkeypatch.setattr(gsp_mod, "_aload_generator", loader)
        gsp_mod._CACHE = {}
        ctx = {"strategy": "synthetic", "target": None}
        with pytest.raises(RuntimeError):
            gsp_mod.generate_stubs([{"x": 1}], ctx, context_builder=DummyBuilder())
        return gen.calls

    monkeypatch.setenv("SANDBOX_STUB_RETRIES", "1")
    gsp_mod.get_settings(refresh=True)
    assert run_expected_calls() == 1

    monkeypatch.setenv("SANDBOX_STUB_RETRIES", "3")
    gsp_mod.get_settings(refresh=True)
    assert run_expected_calls() == 3


@pytest.mark.asyncio
async def test_cache_load_save_race(monkeypatch, tmp_path, recwarn):
    path = tmp_path / "cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(path))
    gsp_mod = importlib.reload(gsp)

    config = gsp_mod.get_config()

    def always_timeout(self, timeout=None):  # type: ignore[override]
        raise gsp_mod.Timeout()

    monkeypatch.setattr(gsp_mod.FileLock, "acquire", always_timeout)

    await asyncio.gather(
        *(gsp_mod._async_load_cache(config) for _ in range(3)),
        *(gsp_mod._async_save_cache(config) for _ in range(3)),
    )

    messages = {str(w.message) for w in recwarn.list}
    assert any("in-memory cache" in m for m in messages)
