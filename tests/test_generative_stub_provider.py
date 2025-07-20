import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import asyncio
import sandbox_runner.generative_stub_provider as gsp

class DummyGen:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, max_length=64, num_return_sequences=1):
        self.calls += 1
        return [{"generated_text": "{\"x\": 1}"}]

def test_generate_stubs_cache(monkeypatch):
    dummy = DummyGen()
    async def loader():
        return dummy
    monkeypatch.setattr(gsp, "_aload_generator", loader)
    gsp._CACHE = {}

    def target(x: int) -> None:
        pass

    first = gsp.generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})
    second = gsp.generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})

    assert first == [{"x": 1}]
    assert second == [{"x": 1}]
    assert dummy.calls == 1


def test_async_generate_returns_json(monkeypatch):
    class AsyncGen:
        async def __call__(self, prompt, max_length=64, num_return_sequences=1):
            return [{"generated_text": "{\"y\": 2}"}]

    dummy = AsyncGen()
    async def loader():
        return dummy
    monkeypatch.setattr(gsp, "_aload_generator", loader)
    gsp._CACHE = {}

    def target(y: int) -> None:
        pass

    result = asyncio.run(gsp.async_generate_stubs([{"y": 0}], {"strategy": "synthetic", "target": target}))
    assert result == [{"y": 2}]
