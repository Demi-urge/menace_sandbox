from types import SimpleNamespace

from memory_aware_gpt_client import ask_with_memory
from local_knowledge_module import LocalKnowledgeModule
from gpt_memory import (
    FEEDBACK,
    ERROR_FIX,
    IMPROVEMENT_PATH,
    INSIGHT,
    _summarise_text,
)


class DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def ask(self, messages, **kwargs):
        resp = self.responses[self.calls]
        self.calls += 1
        return {"choices": [{"message": {"content": resp}}]}


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
    module_a = LocalKnowledgeModule(db_path=db_file)
    builder = SimpleNamespace(build=lambda *a, **k: "")

    ask_with_memory(
        client,
        key,
        prompts[FEEDBACK],
        memory=module_a,
        context_builder=builder,
        tags=[FEEDBACK],
    )
    ask_with_memory(
        client,
        key,
        prompts[ERROR_FIX],
        memory=module_a,
        context_builder=builder,
        tags=[ERROR_FIX],
    )
    ask_with_memory(
        client,
        key,
        prompts[IMPROVEMENT_PATH],
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
