import asyncio
import importlib.util
import logging
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import OrderedDict
from dynamic_path_router import resolve_path

root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
pkg_path = root_path / "sandbox_runner"
sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.__path__ = [str(pkg_path)]
sys.modules["sandbox_runner"] = sandbox_runner_pkg
spec = importlib.util.spec_from_file_location(
    "sandbox_runner.generative_stub_provider",
    resolve_path("sandbox_runner/generative_stub_provider.py"),
)
gsp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gsp)


def _dummy_func(a: int = 0) -> int:
    return a


def test_cache_evict_lru(monkeypatch):
    key1 = ("f1", "1")
    key2 = ("f2", "2")
    key3 = ("f3", "3")
    cfg = gsp.StubProviderConfig(
        timeout=1.0,
        retries=1,
        retry_base=0.1,
        retry_max=0.1,
        cache_max=2,
        cache_path=Path("/dev/null"),
        fallback_model="none",
        save_timeout=1.0,
    )
    with gsp._CACHE_LOCK:
        gsp._CACHE.clear()
        gsp._CACHE[key1] = {"a": 1}
        gsp._CACHE[key2] = {"b": 2}
        gsp._CACHE[key3] = {"c": 3}
        gsp._cache_evict(cfg)
        assert list(gsp._CACHE.keys()) == [key2, key3]
        gsp._CACHE.clear()


def test_move_to_end_logging(monkeypatch, caplog):
    key = gsp._cache_key(_dummy_func.__name__, {"a": 1})

    class FaultyDict(OrderedDict):
        def move_to_end(self, *_):  # type: ignore[override]
            raise RuntimeError("boom")

    monkeypatch.setattr(gsp, "_CACHE", FaultyDict())
    with gsp._CACHE_LOCK:
        gsp._CACHE[key] = {"a": 1}

    async def _fake_aload_generator():
        def _fake_gen(prompt, **_):
            return [{"generated_text": "{}"}]

        return _fake_gen

    monkeypatch.setattr(gsp, "_aload_generator", _fake_aload_generator)

    ctx = {"target": _dummy_func}
    builder = types.SimpleNamespace(
        build_prompt=lambda q, *, intent_metadata=None, **k: q
    )
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(
            gsp.async_generate_stubs([{"a": 1}], ctx, context_builder=builder)
        )
    assert result == [{"a": 1}]
    assert any("failed to update cache LRU" in rec.message for rec in caplog.records)
    with gsp._CACHE_LOCK:
        gsp._CACHE.clear()


def test_thread_safe_cache_eviction(monkeypatch):
    cfg = gsp.StubProviderConfig(
        timeout=1.0,
        retries=1,
        retry_base=0.1,
        retry_max=0.1,
        cache_max=1,
        cache_path=Path("/dev/null"),
        fallback_model="none",
        save_timeout=1.0,
    )
    with gsp._CACHE_LOCK:
        gsp._CACHE.clear()

    def _worker(val: int) -> None:
        key = (f"k{val}", str(val))
        with gsp._CACHE_LOCK:
            gsp._CACHE[key] = {"a": val}
            gsp._cache_evict(cfg)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_worker, range(4)))

    with gsp._CACHE_LOCK:
        assert len(gsp._CACHE) == 1
        remaining = next(iter(gsp._CACHE.values()))
        assert remaining in [{"a": v} for v in range(4)]
        gsp._CACHE.clear()
