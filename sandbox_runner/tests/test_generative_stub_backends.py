import os
import sys
import types
from pathlib import Path

import pytest


sys.modules.setdefault(
    "db_router",
    types.SimpleNamespace(GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None),
)

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sandbox_runner import generative_stub_provider as gsp  # noqa: E402


class _StubSettings:
    def __init__(self):
        self.stub_timeout = 1.0
        self.stub_retries = 1
        self.stub_retry_base = 1.0
        self.stub_retry_max = 1.0
        self.stub_cache_max = 1
        self.stub_fallback_model = os.getenv("SANDBOX_STUB_FALLBACK_MODEL", "distilgpt2")
        self.sandbox_stub_model = os.getenv("SANDBOX_STUB_MODEL")
        self.huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.stub_models = []


def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gsp, "_GENERATOR", None)
    monkeypatch.setattr(gsp, "_SETTINGS", None)
    monkeypatch.setattr(gsp, "_CONFIG", None)
    monkeypatch.setattr(gsp, "pipeline", None)
    monkeypatch.setattr(gsp, "openai", None)
    monkeypatch.setattr(gsp, "SandboxSettings", _StubSettings)


def test_transformers_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setenv("SANDBOX_ENABLE_TRANSFORMERS", "1")
    monkeypatch.setenv("SANDBOX_STUB_MODEL", "foo")
    monkeypatch.setenv("SANDBOX_HUGGINGFACE_TOKEN", "tok")
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "tok")
    monkeypatch.delenv("SANDBOX_ENABLE_OPENAI", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    sentinel = object()

    def fake_pipeline(task: str, model: str, use_auth_token: str):
        assert task == "text-generation"
        assert model == "foo"
        assert use_auth_token == "tok"
        return sentinel

    monkeypatch.setattr(gsp, "pipeline", fake_pipeline)
    monkeypatch.setattr(gsp, "_seed_generator_from_history", lambda gen: None)
    settings = gsp.get_settings(refresh=True)
    assert settings.huggingface_token == "tok"
    cfg = gsp.get_config(refresh=True)
    assert cfg.enabled_backends[0] == "transformers"
    gen = gsp._load_generator(cfg)
    assert gen is sentinel


def test_openai_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setenv("SANDBOX_ENABLE_OPENAI", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.delenv("SANDBOX_ENABLE_TRANSFORMERS", raising=False)

    sentinel = object()

    async def fake_openai():
        return sentinel

    monkeypatch.setattr(gsp, "_load_openai_generator", fake_openai)
    cfg = gsp.get_config(refresh=True)
    assert cfg.enabled_backends == ("openai",)
    gen = gsp._load_generator(cfg)
    assert gen is sentinel


def test_fallback_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setenv("SANDBOX_ENABLE_TRANSFORMERS", "1")
    monkeypatch.setenv("SANDBOX_STUB_FALLBACK_MODEL", "stub")
    monkeypatch.delenv("SANDBOX_STUB_MODEL", raising=False)
    monkeypatch.delenv("SANDBOX_HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("SANDBOX_ENABLE_OPENAI", raising=False)

    sentinel = object()

    async def fake_fallback(cfg):
        assert cfg.fallback_model == "stub"
        return sentinel

    monkeypatch.setattr(gsp, "_load_fallback_pipeline", fake_fallback)
    cfg = gsp.get_config(refresh=True)
    assert cfg.enabled_backends == ("fallback",)
    gen = gsp._load_generator(cfg)
    assert gen is sentinel


def test_no_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    monkeypatch.delenv("SANDBOX_ENABLE_TRANSFORMERS", raising=False)
    monkeypatch.delenv("SANDBOX_ENABLE_OPENAI", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SANDBOX_STUB_MODEL", raising=False)
    monkeypatch.delenv("SANDBOX_HUGGINGFACE_TOKEN", raising=False)

    cfg = gsp.get_config(refresh=True)
    assert cfg.enabled_backends == ()
    with pytest.raises(RuntimeError):
        gsp._load_generator(cfg)

