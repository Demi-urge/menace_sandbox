import sys
import types

import pytest

# Ensure lightweight stubs for modules expected by chatgpt_idea_bot imports
sys.modules.setdefault(
    "menace_sandbox.database_manager",
    types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: []),
)
sys.modules.setdefault(
    "menace_sandbox.database_management_bot",
    types.SimpleNamespace(DatabaseManagementBot=object),
)
sys.modules.setdefault(
    "menace_sandbox.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "menace_sandbox.memory_logging",
    types.SimpleNamespace(log_with_tags=lambda *a, **k: None),
)
sys.modules.setdefault(
    "menace_sandbox.memory_aware_gpt_client",
    types.SimpleNamespace(ask_with_memory=lambda *a, **k: {}),
)
sys.modules.setdefault(
    "menace_sandbox.local_knowledge_module",
    types.SimpleNamespace(LocalKnowledgeModule=lambda *a, **k: types.SimpleNamespace(memory=None)),
)
sys.modules.setdefault(
    "menace_sandbox.knowledge_retriever",
    types.SimpleNamespace(
        get_feedback=lambda *a, **k: [],
        get_improvement_paths=lambda *a, **k: [],
        get_error_fixes=lambda *a, **k: [],
    ),
)
sys.modules.setdefault(
    "menace_sandbox.run_autonomous", types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None)
)
sys.modules.setdefault(
    "menace_sandbox.sandbox_runner", types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None)
)
sys.modules.setdefault(
    "governed_retrieval",
    types.SimpleNamespace(govern_retrieval=lambda *a, **k: None, redact=lambda x: x),
)

import menace_sandbox.chatgpt_idea_bot as cib  # noqa: E402
import menace_sandbox.newsreader_bot as nrb  # noqa: E402
from menace_sandbox.codex_fallback_handler import _ContextClient  # noqa: E402
from context_builder import PromptBuildError  # noqa: E402
from menace_sandbox.chunking import summarize_snippet  # noqa: E402
from prompt_types import Prompt


class FailingBuilder:
    def refresh_db_weights(self) -> None:  # pragma: no cover - simple stub
        pass

    def build_prompt(self, *args, **kwargs):  # pragma: no cover - raises for tests
        raise RuntimeError("builder exploded")


def test_chatgpt_client_build_prompt_with_memory_raises_prompt_build_error(monkeypatch):
    builder = FailingBuilder()
    client = cib.ChatGPTClient(context_builder=builder)

    with pytest.raises(PromptBuildError):
        client.build_prompt_with_memory(["alpha"], context_builder=builder)


def test_chatgpt_build_prompt_propagates_prompt_build_error(monkeypatch):
    builder = FailingBuilder()
    client = cib.ChatGPTClient(context_builder=builder)

    with pytest.raises(PromptBuildError):
        cib.build_prompt(client, builder, ["alpha"])


def test_newsreader_monetise_event_raises_prompt_build_error(monkeypatch):
    builder = FailingBuilder()
    client = cib.ChatGPTClient(context_builder=builder)
    event = nrb.Event("title", "summary", "source", "now")

    with pytest.raises(PromptBuildError):
        nrb.monetise_event(client, event)


def test_codex_fallback_client_raises_prompt_build_error(monkeypatch):
    builder = FailingBuilder()
    client = _ContextClient(model="gpt", context_builder=builder)

    with pytest.raises(PromptBuildError):
        client.generate(Prompt(user="demo"))


def test_chunking_summary_raises_prompt_build_error(monkeypatch):
    module = types.SimpleNamespace(summarize_diff=lambda *_a, **_k: "")
    monkeypatch.setitem(sys.modules, "micro_models.diff_summarizer", module)

    builder = FailingBuilder()
    llm = types.SimpleNamespace(generate=lambda *a, **k: types.SimpleNamespace(text=""))

    with pytest.raises(PromptBuildError):
        summarize_snippet("def demo():\n    pass\n", llm, context_builder=builder)

