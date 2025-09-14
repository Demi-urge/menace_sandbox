import pytest
from llm_interface import Prompt, LLMResult
import codex_fallback_handler as cf


class DummyBuilder:
    def build_prompt(self, user: str, *, intent_metadata=None):
        return Prompt(user, metadata=intent_metadata or {})


def test_fallback_generation_requires_context_builder(monkeypatch):
    calls = {}

    class DummyLLMClient:
        def __init__(self, model: str) -> None:
            pass

        def generate(self, prompt, *, context_builder=None):
            if context_builder is None:
                raise RuntimeError("missing context_builder")
            calls["ctx"] = context_builder
            return LLMResult(text="ok", raw={})

    monkeypatch.setattr(cf, "LLMClient", DummyLLMClient)

    builder = DummyBuilder()
    prompt = Prompt("hi")
    client = cf._ContextClient(model="m", context_builder=builder)

    # Omitting the context builder should raise.
    with pytest.raises(RuntimeError):
        client._client.generate(prompt)

    # The wrapper must forward the builder to the client.
    result = client.generate(prompt)
    assert result.text == "ok"
    assert calls["ctx"] is builder
