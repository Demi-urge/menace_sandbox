import types
import pytest

pytest.skip("optional dependencies not installed", allow_module_level=True)

from menace_sandbox.gpt_memory import GPTMemoryManager
import db_router
import menace_sandbox.gpt_memory as gm
from knowledge_retriever import (
    get_feedback,
    get_error_fixes,
    get_improvement_paths,
    recent_feedback,
    recent_error_fix,
    recent_improvement_path,
)
from types import SimpleNamespace

from context_builder import PromptBuildError, handle_failure
from llm_interface import LLMResult
from local_knowledge_module import LocalKnowledgeModule
from log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH
from memory_logging import ensure_tags
from prompt_types import Prompt


class DummyModel:
    def encode(self, text):
        text = text.lower()
        if "password" in text or "credential" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


class DummyClient:
    def __init__(self):
        self.prompts = []
        self.next_response = ""

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return LLMResult(text=self.next_response)


def _generate_with_memory(
    client,
    *,
    key: str,
    prompt_text: str,
    memory: LocalKnowledgeModule,
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
    except Exception as exc:  # pragma: no cover - defensive
        handle_failure(
            "ContextBuilder.build_prompt failed in tests",
            exc,
        )

    full_tags = ensure_tags(key, tags)
    result = client.generate(
        prompt_obj,
        context_builder=context_builder,
        tags=full_tags,
    )
    text = getattr(result, "text", None)
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    memory.log(prompt_text, text, full_tags)
    return prompt_obj


def test_memory_aware_client_persists_across_runs(tmp_path):
    db = tmp_path / "mem.db"
    embedder = DummyModel()

    client = DummyClient()
    db_router.GLOBAL_ROUTER = None
    gm.GLOBAL_ROUTER = None
    mgr = GPTMemoryManager(db_path=str(db), embedder=embedder)
    module = LocalKnowledgeModule(manager=mgr)

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

    client.next_response = "Great success"
    _generate_with_memory(
        client,
        key="auth.reset_password",
        prompt_text="reset my password",
        memory=module,
        context_builder=builder,
        tags=[FEEDBACK],
    )

    client.next_response = "This fixes the error."
    _generate_with_memory(
        client,
        key="auth.reset_password",
        prompt_text="credential bug encountered",
        memory=module,
        context_builder=builder,
        tags=[ERROR_FIX],
    )

    client.next_response = "An improvement is to apply a patch."
    _generate_with_memory(
        client,
        key="auth.reset_password",
        prompt_text="any improvement suggestions?",
        memory=module,
        context_builder=builder,
        tags=[IMPROVEMENT_PATH],
    )
    mgr.close()

    db_router.GLOBAL_ROUTER = None
    gm.GLOBAL_ROUTER = None
    mgr2 = GPTMemoryManager(db_path=str(db), embedder=embedder)
    module2 = LocalKnowledgeModule(manager=mgr2)

    assert mgr2.search_context(
        "auth.reset_password", tags=[FEEDBACK], use_embeddings=False
    ) == []

    fb = [e.response for e in get_feedback(mgr2, "auth.reset_password") if "insight" not in e.tags]
    fixes = [
        e.response
        for e in get_error_fixes(mgr2, "auth.reset_password")
        if "insight" not in e.tags
    ]
    improvs = [
        e.response
        for e in get_improvement_paths(mgr2, "auth.reset_password")
        if "insight" not in e.tags
    ]

    assert fb == ["Great success"]
    assert fixes == ["This fixes the error."]
    assert improvs == ["An improvement is to apply a patch."]

    client2 = DummyClient()
    client2.next_response = "final"
    _generate_with_memory(
        client2,
        key="auth.reset_password",
        prompt_text="what next?",
        memory=module2,
        context_builder=builder,
    )
    sent_prompt = client2.prompts[0]
    context_blob = "\n".join(sent_prompt.examples)
    assert "Great success" in context_blob
    assert "This fixes the error." in context_blob
    assert "An improvement is to apply a patch." in context_blob

    assert mgr2.search_context("what next?", limit=1)

    service = module2.knowledge
    assert recent_feedback(service)
    assert recent_error_fix(service)
    assert recent_improvement_path(service)
    mgr2.close()

