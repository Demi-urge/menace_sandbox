import anthropic_client
import local_backend
from llm_router import client_from_settings
from sandbox_settings import SandboxSettings
from llm_interface import Prompt
import requests


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
        {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 2}},
    )
    monkeypatch.setattr(anthropic_client.requests, "post", lambda url, headers=None, json=None, timeout=None: resp)
    settings = SandboxSettings(preferred_llm_backend="anthropic")
    client = client_from_settings(settings)
    res = client.generate(Prompt(text="hi"))
    assert res.text == "ok"
    assert res.prompt_tokens == 1
    assert res.completion_tokens == 2


def test_mixtral_local_backend(monkeypatch):
    resp = DummyResp(200, {"text": "local"})
    monkeypatch.setattr(local_backend.requests, "post", lambda url, json=None, timeout=None: resp)
    settings = SandboxSettings(preferred_llm_backend="mixtral")
    client = client_from_settings(settings)
    result = client.generate(Prompt(text="hello"))
    assert result.text == "local"
    assert client.model == "mixtral"


def test_llama3_local_backend(monkeypatch):
    resp = DummyResp(200, {"text": "llama"})
    monkeypatch.setattr(local_backend.requests, "post", lambda url, json=None, timeout=None: resp)
    settings = SandboxSettings(preferred_llm_backend="llama3")
    client = client_from_settings(settings)
    res = client.generate(Prompt(text="hi"))
    assert res.text == "llama"
    assert client.model == "llama3"
