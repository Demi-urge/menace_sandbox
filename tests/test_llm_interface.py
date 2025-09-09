import retry_utils
from db_router import DBRouter
from llm_interface import Prompt, LLMResult, LLMClient
from typing import Any
import random
import time
from llm_router import LLMRouter
from prompt_db import PromptDB
from completion_parsers import parse_json
import asyncio
from context_builder_util import create_context_builder


def test_promptdb_logs_to_memory(tmp_path):
    router = DBRouter("prompts", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = PromptDB(model="test", router=router)
    prompt = Prompt(
        text="hi", examples=["ex"], outcome_tags=["tag"], vector_confidences=[0.4]
    )
    result = LLMResult(raw={"r": 1}, text="res")
    db.log(prompt, result)
    row = db.conn.execute(
        "SELECT text, examples, vector_confidences, outcome_tags, response_text, model FROM prompts"
    ).fetchone()
    assert row == (
        "hi",
        "[\"ex\"]",
        "[0.4]",
        "[\"tag\"]",
        "res",
        "test",
    )


def test_retry_with_backoff(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def func():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("fail")
        return "ok"

    monkeypatch.setattr(retry_utils.time, "sleep", lambda s: sleeps.append(s))
    result = retry_utils.with_retry(func, attempts=3, delay=1.0, exc=ValueError)
    assert result == "ok"
    assert sleeps == [1.0, 2.0]
    assert calls["count"] == 3


class FailingClient(LLMClient):
    def __init__(self):
        super().__init__("fail", log_prompts=False)

    def _generate(self, prompt: Prompt) -> LLMResult:
        raise RuntimeError("boom")


class EchoClient(LLMClient):
    def __init__(self):
        super().__init__("echo", log_prompts=False)
        self.calls = 0

    def _generate(self, prompt: Prompt) -> LLMResult:
        self.calls += 1
        return LLMResult(text="local")


def test_router_fallback_on_error():
    router = LLMRouter(remote=FailingClient(), local=EchoClient(), size_threshold=1)
    res = router.generate(Prompt(text="hi"))
    assert res.text == "local"


def test_openai_provider_retry_and_logging(monkeypatch):
    """OpenAIProvider retries on 429 responses and logs the interaction."""

    from llm_interface import OpenAIProvider, requests, Prompt, LLMResult, rate_limit
    monkeypatch.setattr(OpenAIProvider, "generate", LLMClient.generate)

    # Capture sleep calls to assert backoff behaviour
    sleeps: list[float] = []
    monkeypatch.setattr(
        rate_limit,
        "sleep_with_backoff",
        lambda attempt, base=1.0, max_delay=60.0: sleeps.append(
            min(base * (2**attempt), max_delay)
        ),
    )
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    # Stub PromptDB to record log invocations
    logged: list[tuple[Prompt, LLMResult, str | None]] = []

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, result, backend=None):
            logged.append((prompt, result, backend))

    import prompt_db as pdb

    monkeypatch.setattr(pdb, "PromptDB", DummyDB)

    # Fake HTTP responses: first a rate limit, then success
    class Resp:
        def __init__(self, status, data):
            self.status_code = status
            self.ok = status == 200
            self._data = data

        def json(self):  # pragma: no cover - simple stub
            return self._data

        def raise_for_status(self):  # pragma: no cover - simple stub
            if not self.ok:
                raise requests.HTTPError("boom", response=self)

    responses = [
        Resp(429, {}),
        Resp(
            200,
            {
                "choices": [{"message": {"content": "{\"a\":1}"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ),
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(max_retries=2)
    monkeypatch.setattr(provider._session, "post", fake_post)
    counter = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(counter))

    prompt = Prompt(text="hi", metadata={"tags": ["t"], "vector_confidences": [0.7]})
    result = provider.generate(prompt)

    # Second response consumed and logged
    assert not responses
    assert sleeps == [1.0]
    assert result.parsed == {"a": 1}
    assert result.raw["backend"] == "openai"
    assert logged and logged[0][0] is prompt
    assert logged[0][2] == "openai"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert result.latency_ms == 1000.0


def test_router_fallback_logs(monkeypatch):
    """LLMRouter falls back to local backend and logs the successful prompt."""

    logged: list[str] = []

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, *_a, backend=None, **_k):
            logged.append(backend)

    import llm_router as lr

    monkeypatch.setattr(lr, "PromptDB", DummyDB)

    class BoomClient(LLMClient):
        def __init__(self):
            super().__init__("boom")

        def _generate(self, prompt: Prompt) -> LLMResult:
            raise RuntimeError("fail")

    class LocalClient(LLMClient):
        def __init__(self):
            super().__init__("local")

        def _generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="ok")

    router = LLMRouter(remote=BoomClient(), local=LocalClient(), size_threshold=1)
    res = router.generate(Prompt(text="task", metadata={"tags": ["x"]}))
    assert res.text == "ok"
    assert res.raw["backend"] == "local"
    assert logged == ["local"]


def test_client_backends_fallback():
    """LLMClient tries secondary backends when the first fails."""

    class BoomBackend:
        model = "boom"

        def generate(self, prompt: Prompt) -> LLMResult:
            raise RuntimeError("fail")

    class LocalBackend:
        model = "local"

        def generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="local")

    client = LLMClient(backends=[BoomBackend(), LocalBackend()], log_prompts=False)
    res = client.generate(Prompt(text="hi"))
    assert res.text == "local"


def test_client_small_task_uses_local():
    """Prompts flagged as small_task bypass the first backend."""

    class RemoteBackend:
        model = "remote"

        def generate(self, prompt: Prompt) -> LLMResult:  # pragma: no cover - should not run
            raise AssertionError("should not be called")

    class LocalBackend:
        model = "local"

        def generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="ok")

    client = LLMClient(backends=[RemoteBackend(), LocalBackend()], log_prompts=False)
    res = client.generate(Prompt(text="task", metadata={"small_task": True}))
    assert res.text == "ok"


def test_generate_applies_parse_fn():
    class Dummy(LLMClient):
        def __init__(self):
            super().__init__("dummy", log_prompts=False)

        def _generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
            return LLMResult(text="{\"a\":1}")

    client = Dummy()
    builder = create_context_builder()
    res = client.generate(
        Prompt(text="hi"), parse_fn=parse_json, context_builder=builder
    )
    assert res.parsed == {"a": 1}


def test_generate_parse_fn_error_ignored():
    class Dummy(LLMClient):
        def __init__(self):
            super().__init__("dummy", log_prompts=False)

        def _generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="oops")

    def bad(_text: str):  # pragma: no cover - intentional failure
        raise ValueError("fail")

    client = Dummy()
    res = client.generate(Prompt(text="hi"), parse_fn=bad)
    assert res.parsed is None


def test_successful_call_logs_and_returns_raw_and_parsed(monkeypatch):
    """LLMClient logs prompts and exposes raw and parsed responses."""

    logged: list[tuple[Prompt, LLMResult, str | None]] = []

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt: Prompt, result: LLMResult, backend=None) -> None:
            logged.append((prompt, result, backend))

    import prompt_db

    monkeypatch.setattr(prompt_db, "PromptDB", DummyDB)

    class DummyClient(LLMClient):
        def __init__(self):
            super().__init__("dummy")

        def _generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(raw={"x": 1, "backend": "dummy"}, text="{\"a\":1}")

    client = DummyClient()
    prompt = Prompt(text="hi")
    result = client.generate(prompt, parse_fn=parse_json)

    assert result.raw == {"x": 1, "backend": "dummy"}
    assert result.parsed == {"a": 1}
    assert logged == [(prompt, result, "dummy")]


def test_openai_provider_retries_on_server_error(monkeypatch):
    """Server errors trigger retries with exponential backoff."""

    from llm_interface import OpenAIProvider, requests, rate_limit
    monkeypatch.setattr(OpenAIProvider, "generate", LLMClient.generate)

    sleeps: list[float] = []
    monkeypatch.setattr(
        rate_limit,
        "sleep_with_backoff",
        lambda attempt, base=1.0, max_delay=60.0: sleeps.append(
            min(base * (2**attempt), max_delay)
        ),
    )

    class Resp:
        def __init__(self, status: int, data):
            self.status_code = status
            self.ok = status == 200
            self._data = data

        def json(self):  # pragma: no cover - simple stub
            return self._data

        def raise_for_status(self):  # pragma: no cover - simple stub
            if not self.ok:
                raise requests.HTTPError("boom", response=self)

    responses = [
        Resp(500, {}),
        Resp(200, {"choices": [{"message": {"content": "{\"b\":2}"}}]}),
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(max_retries=2)
    monkeypatch.setattr(provider._session, "post", fake_post)

    result = provider.generate(Prompt(text="hi"))

    assert result.parsed == {"b": 2}
    assert result.raw["choices"][0]["message"]["content"] == "{\"b\":2}"
    assert sleeps == [1.0]
    assert not responses


def test_prompt_serialization_roundtrip():
    import json
    from dataclasses import asdict

    prompt = Prompt(
        user="hello",
        system="sys",
        examples=["ex"],
        vector_confidence=0.5,
        tags=["t"],
        metadata={"k": "v"},
    )

    data = asdict(prompt)
    restored = Prompt(**json.loads(json.dumps(data)))

    assert asdict(restored) == data


def _setup_fake_httpx(monkeypatch):
    import types
    import llm_interface as llmi

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}"
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}"
            yield "data: [DONE]"

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None, json=None, timeout=None):
            return FakeStream()

    monkeypatch.setattr(llmi, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))
    return llmi


def test_openai_provider_async_stream(monkeypatch):
    _setup_fake_httpx(monkeypatch)
    from llm_interface import OpenAIProvider, Prompt

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider()

    chunks: list[str] = []

    async def run():
        async for part in provider.async_generate(Prompt(text="hi")):
            chunks.append(part)

    asyncio.run(run())
    assert chunks == ["Hel", "lo"]


def test_openai_provider_async_logging(monkeypatch):
    llmi = _setup_fake_httpx(monkeypatch)
    from llm_interface import OpenAIProvider, Prompt

    logged: list[tuple[Prompt, Any, str | None]] = []

    class DummyDB:
        def __init__(self, *a, **k):
            pass

        def log(self, prompt, result, backend=None):
            logged.append((prompt, result, backend))

    import prompt_db as pdb

    monkeypatch.setattr(pdb, "PromptDB", DummyDB)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider()

    async def run():
        async for _ in provider.async_generate(Prompt(text="hi")):
            pass

    asyncio.run(run())
    assert logged and logged[0][1].text == "Hello"
    assert logged[0][2] == "openai"


def test_openai_provider_streaming_sync_wrapper(monkeypatch):
    _setup_fake_httpx(monkeypatch)
    from llm_interface import OpenAIProvider, Prompt

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider()

    result = provider.generate(Prompt(text="hi"))
    assert result.text == "Hello"


def _setup_fake_local_httpx(monkeypatch):
    import types
    import local_backend as lb

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield '{"response":"Hel"}'
            yield '{"response":"lo"}'
            yield '{"done": true}'

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json=None, timeout=None):
            return FakeStream()

    monkeypatch.setattr(lb, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))
    return lb


def test_rest_backend_async_stream(monkeypatch):
    lb = _setup_fake_local_httpx(monkeypatch)
    from llm_interface import Prompt, LLMClient

    backend = lb.OllamaBackend(model="m", base_url="http://x")
    client = LLMClient(model="m", backends=[backend], log_prompts=False)

    chunks: list[str] = []

    async def run():
        async for part in client.async_generate(Prompt(text="hi")):
            chunks.append(part)

    asyncio.run(run())
    assert chunks == ["Hel", "lo"]


def test_rest_backend_async_logging(monkeypatch):
    lb = _setup_fake_local_httpx(monkeypatch)
    from llm_interface import Prompt, LLMClient

    backend = lb.OllamaBackend(model="m", base_url="http://x")

    logged: list[tuple[Prompt, Any, str | None]] = []

    class DummyDB:
        def __init__(self, *a, **k):
            pass

        def log(self, prompt, result, backend=None):
            logged.append((prompt, result, backend))

    import prompt_db as pdb

    monkeypatch.setattr(pdb, "PromptDB", DummyDB)
    client = LLMClient(model="m", backends=[backend], log_prompts=True)

    async def run():
        async for _ in client.async_generate(Prompt(text="hi")):
            pass

    asyncio.run(run())
    assert logged and logged[0][1].text == "Hello"
    assert logged[0][2] == "ollama"


def _setup_fake_sse_httpx(monkeypatch):
    import types
    import local_backend as lb

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield 'data: {"delta":{"content":"Hel"}}'
            yield 'data: {"delta":{"content":"lo"}}'
            yield 'data: [DONE]'

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json=None, timeout=None):
            return FakeStream()

    monkeypatch.setattr(lb, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))
    return lb


def test_rest_backend_async_stream_sse(monkeypatch):
    lb = _setup_fake_sse_httpx(monkeypatch)
    from llm_interface import Prompt, LLMClient

    backend = lb.VLLMBackend(model="m", base_url="http://x")
    client = LLMClient(model="m", backends=[backend], log_prompts=False)

    chunks: list[str] = []

    async def run():
        async for part in client.async_generate(Prompt(text="hi")):
            chunks.append(part)

    asyncio.run(run())
    assert chunks == ["Hel", "lo"]
