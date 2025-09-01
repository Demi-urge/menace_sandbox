import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import requests
import pytest

import llm_config
from llm_interface import LLMClient, LLMResult, Prompt, OpenAIProvider, rate_limit
from anthropic_client import AnthropicClient
from local_backend import OllamaBackend, VLLMBackend


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    cfg = llm_config.LLMConfig(model="test-model", api_key="key", max_retries=2, tokens_per_minute=0)
    monkeypatch.setattr(llm_config, "get_config", lambda: cfg)
    return cfg


def test_parse_fn_handling():
    class Dummy(LLMClient):
        def __init__(self):
            super().__init__("dummy", log_prompts=False)

        def _generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="123", raw={"backend": "dummy"})

    client = Dummy()
    prompt = Prompt("hello")

    res = client.generate(prompt, parse_fn=int)
    assert res.parsed == 123

    res_err = client.generate(prompt, parse_fn=lambda x: int("bad"))
    assert res_err.parsed is None


def test_backend_retry_fallback():
    class FailBackend:
        model = "fail"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt: Prompt) -> LLMResult:
            self.calls += 1
            raise RuntimeError("boom")

    class OkBackend:
        model = "ok"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt: Prompt) -> LLMResult:
            self.calls += 1
            return LLMResult(text="ok")

    b1 = FailBackend()
    b2 = OkBackend()
    client = LLMClient(model="router", backends=[b1, b2], log_prompts=False)
    result = client.generate(Prompt("hi"))
    assert result.text == "ok"
    assert b1.calls == 1 and b2.calls == 1


def test_promptdb_logging(monkeypatch):
    logged = []

    class DummyDB:
        def __init__(self, model):
            pass

        def log(self, prompt, result, backend=None):
            logged.append((prompt, result, backend))

    import llm_interface

    import prompt_db

    monkeypatch.setattr(prompt_db, "PromptDB", DummyDB)

    class Dummy(LLMClient):
        def __init__(self):
            super().__init__("dummy")

        def _generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="txt", raw={"backend": "dummy"})

    client = Dummy()
    client.generate(Prompt("x"))
    assert logged and logged[0][2] == "dummy"


def test_openai_provider_retries(monkeypatch):
    backoff = []
    monkeypatch.setattr(rate_limit, "sleep_with_backoff", lambda attempt: backoff.append(attempt))

    provider = OpenAIProvider(model="gpt", api_key="k", max_retries=2)

    calls = []

    def fake_post(url, headers=None, json=None, timeout=30):
        calls.append(1)
        if len(calls) == 1:
            class Resp:
                status_code = 500

                def json(self):
                    return {}

                def raise_for_status(self):
                    pass

            return Resp()

        class Resp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

            def raise_for_status(self):
                pass

        return Resp()

    monkeypatch.setattr(provider._session, "post", fake_post)
    result = provider._generate(Prompt("hi"))
    assert result.text == "ok"
    assert len(calls) == 2
    assert backoff == [0]


def test_anthropic_client_retries(monkeypatch):
    import anthropic_client as ac

    backoff = []
    monkeypatch.setattr(ac.rate_limit, "sleep_with_backoff", lambda attempt: backoff.append(attempt))

    client = AnthropicClient(model="m", api_key="k", max_retries=2)

    calls = []

    def fake_post(url, headers=None, json=None, timeout=30):
        calls.append(1)
        if len(calls) == 1:
            class Resp:
                status_code = 500

                def raise_for_status(self):
                    pass

                def json(self):
                    return {}

            return Resp()

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"content": [{"text": "ok"}], "usage": {}}

        return Resp()

    monkeypatch.setattr(ac.requests, "post", fake_post)
    result = client.generate(Prompt("hi"))
    assert result.text == "ok"
    assert len(calls) == 2
    assert backoff == [0]


def test_openai_cost_calculation(monkeypatch):
    cfg = llm_config.LLMConfig(
        model="m",
        api_key="k",
        max_retries=1,
        tokens_per_minute=0,
        pricing={"m": {"input": 0.1, "output": 0.2}},
    )
    monkeypatch.setattr(llm_config, "get_config", lambda: cfg)
    provider = OpenAIProvider(model="m", api_key="k", max_retries=1)

    def fake_post(url, headers=None, json=None, timeout=30):
        class Resp:
            status_code = 200

            def json(self):
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                }

            def raise_for_status(self):
                pass

        return Resp()

    monkeypatch.setattr(provider._session, "post", fake_post)
    res = provider._generate(Prompt("hi"))
    assert res.cost == pytest.approx(2 * 0.1 + 3 * 0.2)


def test_anthropic_cost_calculation(monkeypatch):
    cfg = llm_config.LLMConfig(
        model="m",
        api_key="k",
        max_retries=1,
        tokens_per_minute=0,
        pricing={"m": {"input": 0.1, "output": 0.2}},
    )
    monkeypatch.setattr(llm_config, "get_config", lambda: cfg)
    import anthropic_client as ac

    client = AnthropicClient(model="m", api_key="k", max_retries=1)

    def fake_post(url, headers=None, json=None, timeout=30):
        class Resp:
            status_code = 200

            def json(self):
                return {
                    "content": [{"text": "ok"}],
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                }

            def raise_for_status(self):
                pass

        return Resp()

    monkeypatch.setattr(ac.requests, "post", fake_post)
    res = client.generate(Prompt("hi"))
    assert res.cost == pytest.approx(2 * 0.1 + 3 * 0.2)


def test_ollama_backend_retries(monkeypatch):
    import local_backend as lb

    backoff = []
    monkeypatch.setattr(lb.rate_limit, "sleep_with_backoff", lambda attempt: backoff.append(attempt))

    backend = OllamaBackend(model="m", base_url="http://x")

    calls = []

    def fake_post(payload):
        calls.append(1)
        if len(calls) == 1:
            raise requests.RequestException("boom")
        return {"text": "ok"}

    monkeypatch.setattr(backend, "_post", fake_post)
    result = backend.generate(Prompt("hi"))
    assert result.text == "ok"
    assert len(calls) == 2
    assert backoff == [0]


def test_vllm_backend_retries(monkeypatch):
    import local_backend as lb

    backoff = []
    monkeypatch.setattr(lb.rate_limit, "sleep_with_backoff", lambda attempt: backoff.append(attempt))

    backend = VLLMBackend(model="m", base_url="http://x")

    calls = []

    def fake_post(payload):
        calls.append(1)
        if len(calls) == 1:
            raise requests.RequestException("boom")
        return {"text": "ok"}

    monkeypatch.setattr(backend, "_post", fake_post)
    result = backend.generate(Prompt("hi"))
    assert result.text == "ok"
    assert len(calls) == 2
    assert backoff == [0]
