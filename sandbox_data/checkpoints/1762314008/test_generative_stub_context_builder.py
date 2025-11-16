import os
import sys
import types

import pytest
from dynamic_path_router import resolve_path

sys.modules.setdefault(
    "db_router",
    types.SimpleNamespace(GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None),
)

sys.path.append(str(resolve_path("")))

from sandbox_runner import generative_stub_provider as gsp  # noqa: E402


class _StubSettings:
    def __init__(self):
        self.stub_timeout = 1.0
        self.stub_retries = 1
        self.stub_retry_base = 0.0
        self.stub_retry_max = 0.0
        self.stub_cache_max = 1
        self.stub_fallback_model = os.getenv("SANDBOX_STUB_FALLBACK_MODEL", "distilgpt2")
        self.sandbox_stub_model = os.getenv("SANDBOX_STUB_MODEL")
        self.llm_backend = os.getenv("LLM_BACKEND", "openai")
        self.stub_save_timeout = 1.0


def _reset(monkeypatch):
    monkeypatch.setattr(gsp, "_GENERATOR", None)
    monkeypatch.setattr(gsp, "_SETTINGS", None)
    monkeypatch.setattr(gsp, "_CONFIG", None)
    monkeypatch.setattr(gsp, "SandboxSettings", _StubSettings)
    monkeypatch.setattr(
        gsp,
        "build_prompt",
        lambda query, *, intent_metadata=None, context_builder, **kwargs: context_builder.build_prompt(
            query, intent_metadata=intent_metadata, **kwargs
        ),
    )
    with gsp._CACHE_LOCK:
        gsp._CACHE.clear()
        gsp._TARGET_STATS.clear()


class DummyBuilder:
    def __init__(self):
        self.calls = 0
        self.last_query = None

    def build_prompt(self, query, *, intent_metadata=None, **kwargs):
        self.calls += 1
        self.last_query = query
        from prompt_types import Prompt

        return Prompt(user=query, metadata=intent_metadata or {})

    def build_context(self, query, *, intent_metadata=None, **kwargs):
        return {}


class DummyGenerator:
    def __init__(self):
        self.calls = 0
        self.received_builders = []

    def generate(self, prompt, *, context_builder=None):
        self.calls += 1
        self.received_builders.append(context_builder)
        if self.calls == 1:
            raise RuntimeError("fail")
        from llm_interface import LLMResult

        return LLMResult(text="{\"a\": 1}")


def test_context_builder_used_for_retries(monkeypatch, tmp_path):
    _reset(monkeypatch)
    monkeypatch.setenv("SANDBOX_STUB_MODEL", "dummy")

    async def fake_aload_generator(config=None):
        return DummyGenerator()

    monkeypatch.setattr(gsp, "_aload_generator", fake_aload_generator)

    builder = DummyBuilder()
    cfg = gsp.get_config(refresh=True)
    cfg.retries = 2
    cfg.retry_base = 0.0
    cfg.retry_max = 0.0
    cfg.cache_path = tmp_path / "cache.json"

    result = gsp.generate_stubs(
        [{"x": 1}], {"target": None}, context_builder=builder, config=cfg
    )

    assert result == [{"a": 1}]
    assert builder.calls == 2
    assert builder.last_query and "x=1" in builder.last_query


def test_generate_receives_context_builder(monkeypatch, tmp_path):
    _reset(monkeypatch)
    monkeypatch.setenv("SANDBOX_STUB_MODEL", "dummy")

    class RecordingGenerator:
        def __init__(self):
            self.received = []

        def generate(self, prompt, *, context_builder=None):
            self.received.append(context_builder)
            from llm_interface import LLMResult

            return LLMResult(text="{\"a\": 1}")

    generator = RecordingGenerator()

    async def fake_aload_generator(config=None):
        return generator

    monkeypatch.setattr(gsp, "_aload_generator", fake_aload_generator)

    builder = DummyBuilder()
    cfg = gsp.get_config(refresh=True)
    cfg.cache_path = tmp_path / "cache.json"

    result = gsp.generate_stubs(
        [{"x": 1}], {"target": None}, context_builder=builder, config=cfg
    )

    assert result == [{"a": 1}]
    assert generator.received == [builder]


def test_missing_context_builder_raises(monkeypatch):
    _reset(monkeypatch)
    with pytest.raises(ValueError):
        gsp.generate_stubs([{"x": 1}], {"target": None}, context_builder=None)
