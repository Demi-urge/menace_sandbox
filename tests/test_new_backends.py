import anthropic_client
import local_backend
from llm_router import client_from_settings
from sandbox_settings import SandboxSettings
from llm_interface import Prompt
import requests
import pytest
import types


class DummyResp:
    def __init__(self, status: int, data: dict):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):  # pragma: no cover - simple stub
        if self.status_code >= 400:
            raise requests.HTTPError("boom", response=self)


def test_anthropic_client_via_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "prompts.db"))
    resp = DummyResp(
        200,
        {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
    )
    monkeypatch.setattr(
        anthropic_client.requests,
        "post",
        lambda url, headers=None, json=None, timeout=None: resp,
    )
    settings = SandboxSettings(preferred_llm_backend="anthropic")
    client = client_from_settings(settings)
    res = client.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder"),
        context_builder=_BUILDER,
    )
    assert res.text == "ok"
    assert res.prompt_tokens == 1
    assert res.completion_tokens == 2


def test_mixtral_local_backend(monkeypatch):
    resp = DummyResp(200, {"text": "local"})
    monkeypatch.setattr(
        local_backend.requests.Session,
        "post",
        lambda self, url, json=None, timeout=None: resp,
    )
    settings = SandboxSettings(preferred_llm_backend="mixtral")
    client = client_from_settings(settings)
    result = client.generate(
        Prompt(text="hello", vector_confidence=0.5, origin="context_builder"),
        context_builder=_BUILDER,
    )
    assert result.text == "local"
    assert client.model == "mixtral"


def test_llama3_local_backend(monkeypatch):
    resp = DummyResp(200, {"text": "llama"})
    monkeypatch.setattr(
        local_backend.requests.Session,
        "post",
        lambda self, url, json=None, timeout=None: resp,
    )
    settings = SandboxSettings(preferred_llm_backend="llama3")
    client = client_from_settings(settings)
    res = client.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder"),
        context_builder=_BUILDER,
    )
    assert res.text == "llama"
    assert client.model == "llama3"


def test_rest_backend_retries_and_latency(monkeypatch):
    """_RESTBackend retries failed posts and records latency and tokens."""

    calls = {"n": 0}

    def fake_post(self, url, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResp(500, {})
        return DummyResp(200, {"text": "ok"})

    sleeps: list[float] = []
    monkeypatch.setattr(
        local_backend.rate_limit,
        "sleep_with_backoff",
        lambda attempt, base=1.0, max_delay=60.0: sleeps.append(
            min(base * (2**attempt), max_delay)
        ),
    )
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    counter = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(local_backend.time, "perf_counter", lambda: next(counter))
    monkeypatch.setattr(local_backend.requests.Session, "post", fake_post)

    backend = local_backend.OllamaBackend(model="m", base_url="http://x")
    result = backend.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder"),
        context_builder=_BUILDER,
    )
    assert result.text == "ok"
    assert result.latency_ms == 1000.0
    assert result.prompt_tokens == local_backend.rate_limit.estimate_tokens("hi", model="m")
    assert result.completion_tokens == local_backend.rate_limit.estimate_tokens("ok", model="m")
    assert sleeps == [1.0]
    assert calls["n"] == 2


def test_rest_backend_propagates_failure(monkeypatch):
    monkeypatch.setattr(
        local_backend.rate_limit,
        "sleep_with_backoff",
        lambda *_a, **_k: None,
    )
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")

    def boom(self, url, json=None, timeout=None):
        return DummyResp(500, {})

    monkeypatch.setattr(local_backend.requests.Session, "post", boom)
    counter = iter([0.0, 1.0])
    monkeypatch.setattr(local_backend.time, "perf_counter", lambda: next(counter))

    backend = local_backend.OllamaBackend(model="m", base_url="http://x")
    with pytest.raises(local_backend._RetryableHTTPError):
        backend.generate(
            Prompt(text="hi", vector_confidence=0.5, origin="context_builder"),
            context_builder=_BUILDER,
        )


@pytest.mark.parametrize(
    "factory, name",
    [
        (local_backend.mixtral_client, "mixtral"),
        (local_backend.llama3_client, "llama3"),
    ],
)
def test_local_clients_log(monkeypatch, factory, name):
    """Local client factories log interactions when ``log_prompts`` is True."""

    resp = DummyResp(200, {"text": "ok"})
    monkeypatch.setattr(
        local_backend.requests.Session,
        "post",
        lambda self, url, json=None, timeout=None: resp,
    )

    logged: dict[str, str] = {}

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, result, backend=None):
            logged["prompt"] = prompt.text
            logged["backend"] = backend

    import prompt_db

    monkeypatch.setattr(prompt_db, "PromptDB", DummyDB)

    client = factory()
    client.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder"),
        context_builder=_BUILDER,
    )
    assert logged == {"prompt": "hi", "backend": name}
_BUILDER = types.SimpleNamespace(roi_tracker=None)
