from types import SimpleNamespace

import db_router
import menace_sandbox.gpt_memory as gm
from context_builder import PromptBuildError, handle_failure
from menace_sandbox.gpt_memory import (
    FEEDBACK,
    ERROR_FIX,
    IMPROVEMENT_PATH,
    INSIGHT,
    _summarise_text,
)
from llm_interface import LLMResult
from local_knowledge_module import LocalKnowledgeModule
from memory_logging import ensure_tags
from prompt_types import Prompt


class DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        resp = self.responses[self.calls]
        self.calls += 1
        return LLMResult(text=resp)


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
    return text


def test_memory_continuity_across_sessions(tmp_path):
    db_file = tmp_path / "mem.db"
    key = "mod.act"

    prompts = {
        FEEDBACK: f"{key} feedback prompt",
        ERROR_FIX: f"{key} fix prompt",
        IMPROVEMENT_PATH: f"{key} improvement prompt",
    }
    responses = {
        FEEDBACK: "feedback resp",
        ERROR_FIX: "fix resp",
        IMPROVEMENT_PATH: "improve resp",
    }

    client = DummyClient(
        [responses[FEEDBACK], responses[ERROR_FIX], responses[IMPROVEMENT_PATH]]
    )
    db_router.GLOBAL_ROUTER = None
    gm.GLOBAL_ROUTER = None
    module_a = LocalKnowledgeModule(db_path=db_file)

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

    _generate_with_memory(
        client,
        key=key,
        prompt_text=prompts[FEEDBACK],
        memory=module_a,
        context_builder=builder,
        tags=[FEEDBACK],
    )
    _generate_with_memory(
        client,
        key=key,
        prompt_text=prompts[ERROR_FIX],
        memory=module_a,
        context_builder=builder,
        tags=[ERROR_FIX],
    )
    _generate_with_memory(
        client,
        key=key,
        prompt_text=prompts[IMPROVEMENT_PATH],
        memory=module_a,
        context_builder=builder,
        tags=[IMPROVEMENT_PATH],
    )

    module_a.refresh()

    fb_insight = _summarise_text(f"{prompts[FEEDBACK]} {responses[FEEDBACK]}")
    fix_insight = _summarise_text(f"{prompts[ERROR_FIX]} {responses[ERROR_FIX]}")
    imp_insight = _summarise_text(
        f"{prompts[IMPROVEMENT_PATH]} {responses[IMPROVEMENT_PATH]}"
    )
    module_a.memory.close()

    db_router.GLOBAL_ROUTER = None
    gm.GLOBAL_ROUTER = None
    module_b = LocalKnowledgeModule(db_path=db_file)

    fb_entries = [e for e in module_b.memory.retrieve("", tags=[FEEDBACK]) if INSIGHT not in e.tags]
    assert [e.response for e in fb_entries] == [responses[FEEDBACK]]
    assert FEEDBACK in fb_entries[0].tags

    fix_entries = [e for e in module_b.memory.retrieve("", tags=[ERROR_FIX]) if INSIGHT not in e.tags]
    assert [e.response for e in fix_entries] == [responses[ERROR_FIX]]
    assert ERROR_FIX in fix_entries[0].tags

    imp_entries = [
        e for e in module_b.memory.retrieve("", tags=[IMPROVEMENT_PATH]) if INSIGHT not in e.tags
    ]
    assert [e.response for e in imp_entries] == [responses[IMPROVEMENT_PATH]]
    assert IMPROVEMENT_PATH in imp_entries[0].tags

    ctx = module_b.build_context(key)
    assert fb_insight in ctx
    assert fix_insight in ctx
    assert imp_insight in ctx

    module_b.memory.close()
