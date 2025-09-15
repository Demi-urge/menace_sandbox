from types import SimpleNamespace

import pytest

import memory_aware_gpt_client as magc
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

    def fake_ask(prompt, **kw):
        recorded["prompt"] = prompt
        return {"choices": [{"message": {"content": "response"}}]}

    client.ask = fake_ask
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


def test_context_builder_failure_raises_and_no_call():
    class FailingBuilder:
        def build_prompt(self, *args, **kwargs):
            raise RuntimeError("boom")

    knowledge = DummyKnowledge()
    called = False

    def fake_ask(*a, **k):
        nonlocal called
        called = True

    client = SimpleNamespace(ask=fake_ask)
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

    def fake_ask(*a, **k):
        nonlocal called
        called = True

    client = SimpleNamespace(ask=fake_ask)

    with pytest.raises(AttributeError):
        magc.ask_with_memory(
            client,
            "mod.act",
            "Do it",
            memory=knowledge,
            context_builder=None,  # type: ignore[arg-type]
        )

    assert not called
