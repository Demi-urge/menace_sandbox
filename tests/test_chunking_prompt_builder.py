import types
import sys

import pytest

import chunking as pc


def test_summarize_snippet_invokes_build_prompt(monkeypatch):
    # Ensure micro-model summariser does not short-circuit
    monkeypatch.setitem(
        sys.modules,
        "micro_models.diff_summarizer",
        types.SimpleNamespace(summarize_diff=lambda a, b: ""),
    )
    # Bypass cache interactions
    monkeypatch.setattr(pc, "_load_snippet_summary", lambda digest: None)
    monkeypatch.setattr(pc, "_store_snippet_summary", lambda digest, summary: None)

    prompts = {}

    class DummyBuilder:
        def build(self, text: str) -> str:  # pragma: no cover - simple stub
            return ""

        def build_prompt(self, text, *, intent=None, intent_metadata=None, top_k=0):
            prompts["built"] = types.SimpleNamespace(user=text, metadata={})
            return prompts["built"]

    called = {}

    def fake_generate(prompt, *, context_builder):  # pragma: no cover - simple stub
        called["prompt"] = prompt
        return types.SimpleNamespace(text="summary")

    builder = DummyBuilder()
    llm = types.SimpleNamespace(generate=fake_generate)

    result = pc.summarize_snippet("example", llm, context_builder=builder)

    assert result == "summary"
    assert "built" in prompts, "context_builder.build_prompt was not invoked"
    assert called["prompt"] is prompts["built"], "llm.generate did not receive built prompt"


def test_summarize_snippet_requires_context_builder():
    with pytest.raises(ValueError, match="context_builder is required"):
        pc.summarize_snippet("example", context_builder=None)
