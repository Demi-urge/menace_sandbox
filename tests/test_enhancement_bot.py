"""Tests for the enhancement bot context injection."""

import sys
import types
from pathlib import Path

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

# Micro model stubs
diff_stub = types.SimpleNamespace(summarize_diff=lambda a, b: "")
micro_pkg = types.ModuleType("menace_sandbox.micro_models")
sys.modules.setdefault("menace_sandbox.micro_models", micro_pkg)
sys.modules.setdefault("menace_sandbox.micro_models.diff_summarizer", diff_stub)

from menace_sandbox.enhancement_bot import EnhancementBot  # noqa: E402
from llm_interface import LLMClient, LLMResult  # noqa: E402


def test_codex_summarize_injects_context():
    calls: dict[str, str | int] = {}

    class DummyBuilder:
        def refresh_db_weights(self):
            return {}

        def build(self, desc: str, *, top_k: int = 5) -> str:
            calls["desc"] = desc
            calls["top_k"] = top_k
            return "CTX"

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            self.prompt = None
            super().__init__(model="dummy", backends=[])

        def generate(self, prompt):  # type: ignore[override]
            self.prompt = prompt
            return LLMResult(text="summary")

    builder = DummyBuilder()
    llm = DummyLLM()
    bot = EnhancementBot(context_builder=builder, llm_client=llm)
    res = bot._codex_summarize("before", "after", hint="diff", confidence=1.0)
    assert res == "summary"
    assert calls["desc"] == "diff"
    assert calls["top_k"] == 5
    assert llm.prompt and "CTX" in llm.prompt.user


def test_requires_context_builder():
    with pytest.raises(ValueError):
        EnhancementBot(context_builder=None)  # type: ignore[arg-type]
