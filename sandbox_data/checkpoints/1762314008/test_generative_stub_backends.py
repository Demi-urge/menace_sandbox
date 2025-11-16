import os
import sys
import types
from pathlib import Path
from dynamic_path_router import resolve_path

sys.modules.setdefault(
    "db_router",
    types.SimpleNamespace(GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None),
)

sys.path.append(str(resolve_path("")))

from sandbox_runner import generative_stub_provider as gsp  # noqa: E402
import pytest


class _StubSettings:
    def __init__(self):
        self.stub_timeout = 1.0
        self.stub_retries = 1
        self.stub_retry_base = 1.0
        self.stub_retry_max = 1.0
        self.stub_cache_max = 1
        self.stub_fallback_model = os.getenv("SANDBOX_STUB_FALLBACK_MODEL", "distilgpt2")
        self.sandbox_stub_model = os.getenv("SANDBOX_STUB_MODEL")
        self.llm_backend = os.getenv("LLM_BACKEND", "openai")
        self.stub_save_timeout = 1.0


def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
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


class DummyBuilder:
    def build_prompt(self, query, *, intent_metadata=None, **kwargs):
        return query

    def build_context(self, query, *, intent_metadata=None, **kwargs):
        return {}


def test_llm_client_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setenv("SANDBOX_STUB_MODEL", "foo")

    class DummyClient:
        model = "foo"

        def generate(self, prompt, *, context_builder=None):
            from llm_interface import LLMResult

            return LLMResult(text="{\"a\": 1}")

    monkeypatch.setattr(gsp, "get_client", lambda name, **kw: DummyClient())
    result = gsp.generate_stubs([{}], {"target": None}, context_builder=DummyBuilder())
    assert result == [{"a": 1}]


def test_history_fallback_when_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    monkeypatch.delenv("SANDBOX_STUB_MODEL", raising=False)

    class HistDB:
        def recent(self, n):
            return [{"x": 1}]

    monkeypatch.setattr(gsp, "_get_history_db", lambda: HistDB())
    result = gsp.generate_stubs([{}], {"target": None}, context_builder=DummyBuilder())
    assert result == [{"x": 1}]
