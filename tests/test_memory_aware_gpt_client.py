from types import SimpleNamespace

import pytest

import memory_aware_gpt_client as magc
from context_builder import PromptBuildError
from llm_interface import LLMResult
from prompt_types import Prompt


class DummyKnowledge:
    def __init__(self):
        self.logged = []

    def build_context(self, key: str, limit: int = 5) -> str:
        return (
            "### Feedback\n- fb1\n\n### Error fixes\n- fix1\n\n### Improvement paths\n- imp1"
        )

    def log(self, prompt: str, resp: str, tags):
        self.logged.append((prompt, resp, tags))


def test_context_injection_and_logging():
    client = SimpleNamespace()
    recorded = {}

    def fake_generate(prompt, **kwargs):
        recorded["prompt"] = prompt
        recorded["kwargs"] = kwargs
        return LLMResult(text="response")

    client.generate = fake_generate
    knowledge = DummyKnowledge()

    builder = SimpleNamespace(
        build_prompt=lambda q, **k: Prompt(user=q)
    )

    magc.ask_with_memory(
        client,
        "mod.act",
        "Do it",
        memory=knowledge,
        context_builder=builder,
        tags=["feedback"],
    )

    sent_prompt = recorded["prompt"]
    assert isinstance(sent_prompt, Prompt)
    joined = "\n".join(sent_prompt.examples)
    assert "fb1" in joined
    assert "fix1" in joined
    assert "imp1" in joined
    assert knowledge.logged and knowledge.logged[0][0].endswith("Do it")
    assert recorded["kwargs"]["tags"] == ["feedback", "module:mod", "action:act"]


def test_context_builder_failure_raises_and_no_call():
    class FailingBuilder:
        def build_prompt(self, *args, **kwargs):
            raise RuntimeError("boom")

    knowledge = DummyKnowledge()
    called = False

    def fake_generate(*a, **k):
        nonlocal called
        called = True

    client = SimpleNamespace(generate=fake_generate)
    builder = FailingBuilder()

    with pytest.raises(RuntimeError):
        magc.ask_with_memory(
            client,
            "mod.act",
            "Do it",
            memory=knowledge,
            context_builder=builder,
        )

    assert not called


def test_missing_context_builder_raises():
    knowledge = DummyKnowledge()
    called = False

    def fake_generate(*a, **k):
        nonlocal called
        called = True

    client = SimpleNamespace(generate=fake_generate)

    with pytest.raises(PromptBuildError) as exc_info:
        magc.ask_with_memory(
            client,
            "mod.act",
            "Do it",
            memory=knowledge,
            context_builder=None,  # type: ignore[arg-type]
        )

    assert not called
    assert isinstance(exc_info.value.__cause__, AttributeError)
