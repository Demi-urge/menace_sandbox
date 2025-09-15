import json
import requests
import rate_limit

import local_backend
from local_backend import OllamaBackend
from llm_interface import Prompt, LLMClient, OpenAIProvider
from context_builder_util import create_context_builder


class DummyResp:
    def __init__(self, status: int, data: dict):
        self.status_code = status
        self._data = data

    def json(self):  # pragma: no cover - simple stub
        return self._data

    def raise_for_status(self):  # pragma: no cover - simple stub
        if self.status_code >= 400:
            raise requests.HTTPError("boom", response=self)


def test_openai_success_logging_and_parsing(monkeypatch, tmp_path):
    monkeypatch.setattr(OpenAIProvider, "generate", LLMClient.generate)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "prompts.db"))
    client = OpenAIProvider(model="gpt-test")

    resp = DummyResp(200, {"choices": [{"message": {"content": json.dumps({"a": 1})}}]})
    monkeypatch.setattr(
        client._session,
        "post",
        lambda url, headers=None, json=None, timeout=None: resp,
    )

    prompt = Prompt(
        text="hi",
        metadata={"tags": ["t"], "vector_confidences": [0.9]},
        origin="context_builder",
    )
    builder = create_context_builder()
    result = client.generate(prompt, parse_fn=json.loads, context_builder=builder)
    assert result.parsed == {"a": 1}
    row = client.db.conn.execute(
        "SELECT response_text, response_parsed FROM prompts",
    ).fetchone()
    assert json.loads(row[0]) == {"a": 1}
    assert json.loads(row[1]) == {"a": 1}


def test_openai_rate_limit_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(OpenAIProvider, "generate", LLMClient.generate)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "prompts.db"))
    client = OpenAIProvider(model="gpt-test")
    sleeps: list[float] = []
    monkeypatch.setattr(
        rate_limit,
        "sleep_with_backoff",
        lambda attempt, base=1.0, max_delay=60.0: sleeps.append(
            min(base * (2**attempt), max_delay)
        ),
    )

    responses = [
        DummyResp(429, {}),
        DummyResp(200, {"choices": [{"message": {"content": "ok"}}]}),
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(client._session, "post", fake_post)

    res = client.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder"),
        context_builder=create_context_builder(),
    )
    assert res.text == "ok"
    assert sleeps == [1.0]
    assert not responses


def test_client_fallback_to_local_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(OpenAIProvider, "generate", LLMClient.generate)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "prompts.db"))

    openai_backend = OpenAIProvider(model="gpt-test")
    openai_backend._log_prompts = False
    openai_backend.db = None
    monkeypatch.setattr(
        rate_limit,
        "sleep_with_backoff",
        lambda attempt, base=1.0, max_delay=60.0: None,
    )

    def failing_post(url, headers=None, json=None, timeout=None):
        raise requests.RequestException("boom")

    monkeypatch.setattr(openai_backend._session, "post", failing_post)

    local = OllamaBackend(model="local", base_url="http://llm")

    resp = DummyResp(200, {"text": json.dumps({"b": 2})})

    def local_post(url, json=None, timeout=None):
        return resp

    monkeypatch.setattr(
        local_backend.requests.Session,
        "post",
        lambda self, url, json=None, timeout=None: local_post(
            url, json=json, timeout=timeout
        ),
    )

    client = LLMClient(backends=[openai_backend, local])

    builder = create_context_builder()
    result = client.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder"),
        parse_fn=json.loads,
        context_builder=builder,
    )
    assert result.parsed == {"b": 2}
    expected_prompt_tokens = rate_limit.estimate_tokens("hi", model="local")
    expected_completion_tokens = rate_limit.estimate_tokens(
        json.dumps({"b": 2}), model="local"
    )
    assert result.prompt_tokens == expected_prompt_tokens
    assert result.completion_tokens == expected_completion_tokens
    row = client.db.conn.execute(
        "SELECT response_text, response_parsed FROM prompts",
    ).fetchone()
    assert json.loads(row[0]) == {"b": 2}
    assert json.loads(row[1]) == {"b": 2}
