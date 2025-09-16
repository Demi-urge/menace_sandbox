from types import SimpleNamespace

import pytest

import memory_aware_gpt_client as magc
from context_builder import PromptBuildError, handle_failure
from llm_interface import LLMResult
from prompt_types import Prompt
from memory_logging import ensure_tags


class DummyKnowledge:
    def __init__(self):
        self.logged = []

    def build_context(self, key: str, limit: int = 5) -> str:
        return (
            "### Feedback\n- fb1\n\n### Error fixes\n- fix1\n\n### Improvement paths\n- imp1"
        )

    def log(self, prompt: str, resp: str, tags):
        self.logged.append((prompt, resp, tags))


def _generate_with_memory(
    client,
    *,
    key: str,
    prompt_text: str,
    memory: DummyKnowledge,
    context_builder,
    tags: list[str] | None = None,
):
    try:
        mem_ctx = memory.build_context(key, limit=5)
    except Exception:
        mem_ctx = ""

    intent_meta: dict[str, str] = {"user_query": prompt_text}
    if mem_ctx:
        intent_meta["memory_context"] = mem_ctx

    if context_builder is None:
        handle_failure(
            "ContextBuilder.build_prompt failed in tests",  # pragma: no cover - defensive
            AttributeError("context_builder is None"),
        )

    try:
        prompt_obj = context_builder.build_prompt(
            prompt_text,
            intent_metadata=intent_meta,
            session_id="session-id",
        )
    except PromptBuildError:
        raise
    except Exception as exc:  # pragma: no cover - safety net
        handle_failure(
            "ContextBuilder.build_prompt failed in tests",
            exc,
        )

    full_tags = ensure_tags(key, tags)
    prompt_obj.metadata.setdefault("intent_metadata", {}).update(intent_meta)
    prompt_obj.metadata.setdefault("tags", full_tags)

    result = client.generate(
        prompt_obj,
        context_builder=context_builder,
        tags=full_tags,
    )
    text = getattr(result, "text", None)
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    memory.log(prompt_text, text, full_tags)
    return prompt_obj, text, full_tags


def test_context_injection_and_logging():
    client = SimpleNamespace()
    recorded = {}

    def fake_generate(prompt, **kwargs):
        recorded["prompt"] = prompt
        recorded["kwargs"] = kwargs
        return LLMResult(text="response")

    client.generate = fake_generate
    knowledge = DummyKnowledge()

    def build_prompt(query: str, *, intent_metadata=None, session_id=None):
        prompt = Prompt(user=query)
        prompt.metadata.setdefault("session_id", session_id or "")
        if intent_metadata:
            prompt.metadata.update(intent_metadata)
            memory_ctx = intent_metadata.get("memory_context")
            if memory_ctx:
                prompt.examples.append(memory_ctx)
        return prompt

    builder = SimpleNamespace(build_prompt=build_prompt)

    _, _, full_tags = _generate_with_memory(
        client,
        key="mod.act",
        prompt_text="Do it",
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
    assert recorded["kwargs"]["tags"] == full_tags


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

    with pytest.raises(PromptBuildError) as exc_info:
        _generate_with_memory(
            client,
            key="mod.act",
            prompt_text="Do it",
            memory=knowledge,
            context_builder=builder,
        )

    assert not called
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_missing_context_builder_raises():
    knowledge = DummyKnowledge()
    called = False

    def fake_generate(*a, **k):
        nonlocal called
        called = True

    client = SimpleNamespace(generate=fake_generate)

    with pytest.raises(PromptBuildError) as exc_info:
        _generate_with_memory(
            client,
            key="mod.act",
            prompt_text="Do it",
            memory=knowledge,
            context_builder=None,
        )

    assert not called
    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_ask_with_memory_removed():
    with pytest.raises(AttributeError):
        magc.ask_with_memory  # type: ignore[attr-defined]
