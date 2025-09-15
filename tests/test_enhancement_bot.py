"""Tests for the enhancement bot context injection."""

import sys
import types
from pathlib import Path
import importlib

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub heavy dependencies so the module imports without optional components
ceb_stub = types.SimpleNamespace(
    EnhancementDB=object,
    EnhancementHistory=object,
    Enhancement=object,
)
sys.modules.setdefault("menace_sandbox.chatgpt_enhancement_bot", ceb_stub)
sys.modules.setdefault("chatgpt_enhancement_bot", ceb_stub)

code_db_stub = types.SimpleNamespace(CodeDB=object)
sys.modules.setdefault("menace_sandbox.code_database", code_db_stub)
sys.modules.setdefault("code_database", code_db_stub)

# Avoid pulling in heavy transformer stack during imports
transformers_stub = types.SimpleNamespace(AutoTokenizer=None)
sys.modules.setdefault("transformers", transformers_stub)

# Ensure snippet_compressor resolves relative imports correctly
sc_mod = importlib.import_module("menace_sandbox.snippet_compressor")
sys.modules.setdefault("snippet_compressor", sc_mod)

# Provide minimal vector_service stub
vector_service_pkg = types.ModuleType("vector_service")
context_builder_mod = types.SimpleNamespace(ContextBuilder=object)
vector_service_pkg.context_builder = context_builder_mod
embedding_stub = types.SimpleNamespace(
    EmbeddingBackfill=types.SimpleNamespace(watch_events=lambda *a, **k: None)
)
vector_service_pkg.embedding_backfill = embedding_stub
vector_service_pkg.EmbeddingBackfill = embedding_stub.EmbeddingBackfill
vector_service_pkg.EmbeddableDBMixin = object
sys.modules.setdefault("vector_service", vector_service_pkg)
sys.modules.setdefault("vector_service.context_builder", context_builder_mod)
sys.modules.setdefault("vector_service.embedding_backfill", embedding_stub)

# Micro model stubs
diff_stub = types.SimpleNamespace(summarize_diff=lambda *a, **k: "")
micro_pkg = types.ModuleType("menace_sandbox.micro_models")
sys.modules.setdefault("menace_sandbox.micro_models", micro_pkg)
sys.modules.setdefault("menace_sandbox.micro_models.diff_summarizer", diff_stub)

from menace_sandbox.enhancement_bot import EnhancementBot  # noqa: E402
from llm_interface import LLMClient, LLMResult, Prompt  # noqa: E402


def test_codex_summarize_injects_context():
    calls: dict[str, str | int] = {}

    prompts: dict[str, object] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build(self, desc: str, *, top_k: int = 5) -> str:
            calls["desc"] = desc
            calls["top_k"] = top_k
            return "CTX"

        def build_prompt(self, query, *, intent=None, intent_metadata=None, top_k=0):
            meta = intent or intent_metadata or {}
            prompts["built"] = Prompt(query, metadata={"intent": meta})
            return prompts["built"]

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            self.prompt = None
            self.ctx = None
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt, *, context_builder=None):  # type: ignore[override]
            self.prompt = prompt
            self.ctx = context_builder
            return LLMResult(text="summary")

    builder = DummyBuilder()
    llm = DummyLLM()
    bot = EnhancementBot(context_builder=builder, llm_client=llm)
    res = bot._codex_summarize("before", "after", hint="diff", confidence=1.0)
    assert res == "summary"
    assert calls["desc"] == "diff"
    assert calls["top_k"] == 5
    assert llm.prompt is prompts["built"]
    meta = prompts["built"].metadata["intent"]
    assert meta["retrieved_context"] == "CTX"
    assert meta["refactor_summary"] == "diff"
    assert "before_hash" in meta and "after_hash" in meta
    assert llm.ctx is builder


def test_codex_summarize_compresses_context():
    long_ctx = "X" * 1000
    from snippet_compressor import compress_snippets
    expected = compress_snippets({"snippet": long_ctx})["snippet"]

    prompts: dict[str, Prompt] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build(self, desc: str, *, top_k: int = 5) -> str:  # noqa: D401 - simple stub
            return long_ctx

        def build_prompt(self, query, *, intent=None, intent_metadata=None, top_k=0):
            meta = intent or intent_metadata or {}
            prompts["built"] = Prompt(query, metadata={"intent": meta})
            return prompts["built"]

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            self.prompt = None
            self.ctx = None
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt, *, context_builder=None):  # type: ignore[override]
            self.prompt = prompt
            self.ctx = context_builder
            return LLMResult(text="summary")

    bot = EnhancementBot(context_builder=DummyBuilder(), llm_client=DummyLLM())
    bot._codex_summarize("before", "after", hint="diff", confidence=1.0)
    assert bot.llm_client and bot.llm_client.prompt  # type: ignore[attr-defined]
    meta = prompts["built"].metadata["intent"]
    assert meta["retrieved_context"] == expected
    assert meta["retrieved_context"] != long_ctx


def test_codex_summarize_builds_prompt():
    prompts: dict[str, Prompt] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build(self, *_a, **_k):  # pragma: no cover - simple stub
            return ""

        def build_prompt(self, query, *, intent=None, intent_metadata=None, top_k=0):
            prompts["built"] = Prompt(query, metadata={"intent": intent or intent_metadata or {}})
            return prompts["built"]

    called: dict[str, object] = {}

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt, *, context_builder=None):  # type: ignore[override]
            called["prompt"] = prompt
            return LLMResult(text="ok")

    bot = EnhancementBot(context_builder=DummyBuilder(), llm_client=DummyLLM())
    bot._codex_summarize("before", "after", hint="diff")
    assert "built" in prompts, "context_builder.build_prompt was not invoked"
    assert called["prompt"] is prompts["built"], "llm_client.generate did not receive built prompt"


def test_requires_context_builder():
    with pytest.raises(ValueError):
        EnhancementBot(context_builder=None)  # type: ignore[arg-type]
