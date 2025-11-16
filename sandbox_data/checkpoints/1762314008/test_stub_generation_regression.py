from __future__ import annotations

import json
import sys
import types

th_stub = types.ModuleType("sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: None
th_stub.TestHarnessResult = object
sys.modules.setdefault("sandbox_runner.test_harness", th_stub)
import sandbox_runner.generative_stub_provider as gsp
from dynamic_path_router import resolve_path


def sample_func(name: str, count: int, active: bool) -> None:
    """Sample function used for stub generation."""
    return None


def test_stub_generation_matches_fixture(monkeypatch):
    """Generated stubs remain deterministic across runs."""

    async def fake_aload_generator():
        return None

    # Avoid loading optional model backends
    monkeypatch.setattr(gsp, "_aload_generator", fake_aload_generator)
    gsp._CACHE.clear()

    stubs = gsp.generate_stubs(
        [{}], {"target": sample_func}, context_builder=types.SimpleNamespace(build_prompt=lambda q, *, intent_metadata=None, **k: q)
    )[0]

    expected = json.loads(
        resolve_path("tests/fixtures/regression/stub_generation.json").read_text()
    )
    assert stubs == expected
