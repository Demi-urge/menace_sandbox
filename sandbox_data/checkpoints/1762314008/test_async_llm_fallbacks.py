import asyncio
import types
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from llm_interface import Prompt, LLMClient, LLMResult
from llm_router import LLMRouter


class _AsyncStubClient(LLMClient):
    def __init__(self, text: str, *, fail: bool = False) -> None:
        super().__init__(text, log_prompts=False)
        self.text = text
        self.fail = fail

    def _generate(
        self, prompt: Prompt, *, context_builder
    ) -> LLMResult:  # pragma: no cover - unused
        raise NotImplementedError

    async def async_generate(self, prompt: Prompt, *, context_builder):
        if self.fail:
            raise RuntimeError("boom")
        yield self.text[: len(self.text) // 2]
        yield self.text[len(self.text) // 2:]


def test_router_async_fallback():
    remote = _AsyncStubClient("remote")
    local = _AsyncStubClient("local", fail=True)
    router = LLMRouter(remote=remote, local=local, size_threshold=5)

    chunks: list[str] = []
    builder = types.SimpleNamespace(roi_tracker=None)

    async def run() -> None:
        async for part in router.async_generate(
            Prompt(text="hi", origin="context_builder", metadata={"vector_confidences": [0.5]}), context_builder=builder
        ):
            chunks.append(part)

    asyncio.run(run())
    assert "".join(chunks) == "remote"


def _setup_fake_local_httpx(monkeypatch):
    import local_backend as lb

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield '{"response":"ok"}'
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
            if "fail" in url:
                raise RuntimeError("boom")
            return FakeStream()

    monkeypatch.setattr(
        lb,
        "httpx",
        types.SimpleNamespace(AsyncClient=FakeClient),
    )
    monkeypatch.setattr(lb.rate_limit, "sleep_with_backoff", lambda *_: None)
    monkeypatch.setattr(
        lb.llm_config,
        "get_config",
        lambda: types.SimpleNamespace(
            max_retries=1, tokens_per_minute=1000, pricing={}
        ),
    )
    return lb


def test_local_backend_async_fallback(monkeypatch):
    lb = _setup_fake_local_httpx(monkeypatch)

    backend1 = lb.OllamaBackend(model="m", base_url="http://fail")
    backend2 = lb.OllamaBackend(model="m", base_url="http://ok")
    client = LLMClient(model="m", backends=[backend1, backend2], log_prompts=False)

    chunks: list[str] = []
    builder = types.SimpleNamespace(roi_tracker=None)

    async def run() -> None:
        async for part in client.async_generate(
            Prompt(text="hi", origin="context_builder", metadata={"vector_confidences": [0.5]}), context_builder=builder
        ):
            chunks.append(part)

    asyncio.run(run())
    assert chunks == ["ok"]
