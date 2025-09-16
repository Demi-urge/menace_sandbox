import pytest
from llm_interface import Prompt, LLMResult
import codex_fallback_handler as cf


class DummyBuilder:
    def build_prompt(self, user: str, *, intent=None, **kwargs):
        return Prompt(user, metadata=intent or {})


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

    # The wrapper should return the enriched prompt from the builder.
    enriched_prompt = client.generate(prompt)
    assert isinstance(enriched_prompt, Prompt)
    assert enriched_prompt is not prompt

    # Omitting the context builder should raise when calling the LLM directly.
    with pytest.raises(RuntimeError):
        client._client.generate(enriched_prompt)

    # Rerouting via the handler must forward the context builder to the client.
    result = cf.reroute_to_fallback_model(prompt, context_builder=builder)
    assert result.text == "ok"
    assert calls["ctx"] is builder
