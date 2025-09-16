import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import asyncio
import types

import requests
import pytest

import llm_config
from llm_interface import LLMClient, LLMResult, Prompt, OpenAIProvider, rate_limit
from anthropic_client import AnthropicClient
from local_backend import OllamaBackend, VLLMBackend


def _builder():
    return types.SimpleNamespace(roi_tracker=None)


PROMPT_META = {"vector_confidences": [0.5]}


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    cfg = llm_config.LLMConfig(model="test-model", api_key="key", max_retries=2, tokens_per_minute=0)
    monkeypatch.setattr(llm_config, "get_config", lambda: cfg)
    return cfg


def test_requires_origin():
    class Dummy(LLMClient):
        def __init__(self):
            super().__init__("dummy", log_prompts=False)

        def _generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
            return LLMResult(text="ok", raw={"backend": "dummy"})

    client = Dummy()
    with pytest.raises(ValueError):
        client.generate(Prompt("hi"), context_builder=_builder())
    with pytest.raises(ValueError):
        client.generate(Prompt("hi", origin="unknown"), context_builder=_builder())
    assert (
        client.generate(
            Prompt(
                "hi",
                origin="context_builder",
                metadata=PROMPT_META,
            ),
            context_builder=_builder(),
        ).text
        == "ok"
    )

def test_parse_fn_handling():
    class Dummy(LLMClient):
        def __init__(self):
            super().__init__("dummy", log_prompts=False)

        def _generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
            return LLMResult(text="123", raw={"backend": "dummy"})

    client = Dummy()
    prompt = Prompt("hello", origin="context_builder", metadata=PROMPT_META)

    res = client.generate(prompt, parse_fn=int, context_builder=_builder())
    assert res.parsed == 123

    res_err = client.generate(
        prompt, parse_fn=lambda x: int("bad"), context_builder=_builder()
    )
    assert res_err.parsed is None


def test_backend_retry_fallback():
    class FailBackend:
        model = "fail"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
            self.calls += 1
            raise RuntimeError("boom")

    class OkBackend:
        model = "ok"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
            self.calls += 1
            return LLMResult(text="ok")

    b1 = FailBackend()
    b2 = OkBackend()
    client = LLMClient(model="router", backends=[b1, b2], log_prompts=False)
    result = client.generate(
        Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
    )
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

        def _generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
            return LLMResult(text="txt", raw={"backend": "dummy"})

    client = Dummy()
    client.generate(Prompt("x", origin="context_builder", metadata=PROMPT_META), context_builder=_builder())
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
    result = provider._generate(
        Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
    )
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
    result = client.generate(
        Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
    )
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
    res = provider._generate(
        Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
    )
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
    res = client.generate(
        Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
    )
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
    result = backend.generate(
        Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
    )
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
    result = backend.generate(
        Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
    )
    assert result.text == "ok"
    assert len(calls) == 2
    assert backoff == [0]


def test_local_weights_streaming(monkeypatch):
    import asyncio
    import queue
    import types
    import sys
    import importlib

    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def __call__(self, text, return_tensors=None):
            return {"input_ids": [0]}

    class DummyModel:
        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def generate(self, *args, **kwargs):
            streamer = kwargs.get("streamer")
            if streamer is not None:
                streamer.put("hello ")
                streamer.put("world")
                streamer.end()
            else:
                return [[0]]

    class DummyStreamer:
        def __init__(self, *args, **kwargs):
            self.q = queue.Queue()

        def put(self, val):
            self.q.put(val)

        def end(self):
            self.q.put(None)

        def __iter__(self):
            return self

        def __next__(self):
            item = self.q.get()
            if item is None:
                raise StopIteration
            return item

    dummy_tf = types.ModuleType("transformers")
    dummy_tf.AutoTokenizer = DummyTokenizer
    dummy_tf.AutoModelForCausalLM = DummyModel
    dummy_tf.TextIteratorStreamer = DummyStreamer

    monkeypatch.setitem(sys.modules, "transformers", dummy_tf)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    pb = importlib.import_module("private_backend")
    monkeypatch.setattr(pb, "torch", None)
    client = pb.local_weights_client(model="m")

    builder = _builder()

    async def collect():
        parts = []
        async for part in client.async_generate(
            Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=builder
        ):
            parts.append(part)
        return "".join(parts)

    text = asyncio.run(collect())
    assert text == "hello world"


def test_openai_generate_inside_running_loop(monkeypatch):
    provider = OpenAIProvider(model="gpt", api_key="k")
    monkeypatch.setattr(OpenAIProvider, "_log", lambda *a, **k: None)

    async def fake_async_gen(self, prompt, *, context_builder):
        yield "hi"

    monkeypatch.setattr(OpenAIProvider, "_async_generate", fake_async_gen)

    async def run():
        return await provider.generate(
            Prompt("hi", origin="context_builder", metadata=PROMPT_META), context_builder=_builder()
        )

    result = asyncio.run(run())
    assert result.text == "hi"
