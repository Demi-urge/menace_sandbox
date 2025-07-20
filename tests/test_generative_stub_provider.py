import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import sandbox_runner.generative_stub_provider as gsp

class DummyGen:
    def __init__(self):
        self.calls = 0
    def __call__(self, prompt, max_length=64, num_return_sequences=1):
        self.calls += 1
        return [{"generated_text": "{\"x\": 1}"}]

def test_generate_stubs_cache(monkeypatch):
    dummy = DummyGen()
    monkeypatch.setattr(gsp, "_load_generator", lambda: dummy)
    gsp._CACHE = {}

    def target(x: int) -> None:
        pass

    first = gsp.generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})
    second = gsp.generate_stubs([{"x": 0}], {"strategy": "synthetic", "target": target})

    assert first == [{"x": 1}]
    assert second == [{"x": 1}]
    assert dummy.calls == 1
