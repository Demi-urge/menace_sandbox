import sys
import types
from typing import Any, Dict, List

# Stub heavy dependencies before importing PromptEngine
sys.modules.setdefault("gpt_memory", types.SimpleNamespace(GPTMemoryManager=object))
sys.modules.setdefault("code_database", types.SimpleNamespace(PatchHistoryDB=object))

from prompt_engine import PromptEngine  # noqa: E402
import prompt_engine as pe  # noqa: E402
import chunking as pc  # noqa: E402


class DummyRetriever:
    def __init__(self, records: List[Dict[str, Any]] | None = None) -> None:
        self.records = records or []

    def search(self, query: str, top_k: int):
        return self.records


def test_prompt_engine_auto_summarises_when_limit_exceeded(monkeypatch):
    code = "def big():\n" + "    x = 0\n" * 2000

    def fake_split(text: str, limit: int):
        Chunk = types.SimpleNamespace
        return [
            Chunk(text="chunkA", start_line=1, end_line=1, hash="h1", token_count=10),
            Chunk(text="chunkB", start_line=2, end_line=2, hash="h2", token_count=10),
        ]

    monkeypatch.setattr(pe, "split_into_chunks", fake_split)

    calls: list[str] = []

    def fake_summarize(text: str, llm: object, context_builder=None) -> str:
        calls.append(text)
        return f"sum:{text}"

    monkeypatch.setattr(pc, "summarize_code", fake_summarize)
    monkeypatch.setattr(pe, "summarize_code", fake_summarize)

    records = [{"score": 0.9, "metadata": {"summary": "irrelevant", "tests_passed": True}}]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        confidence_threshold=-1.0,
        token_threshold=50,
        chunk_token_threshold=20,
        llm=object(),
        context_builder=object(),
    )

    prompt = engine.build_prompt(
        "do something", context=code, context_builder=engine.context_builder
    )
    assert calls == ["chunkA", "chunkB"]
    assert "sum:chunkA" in prompt and "sum:chunkB" in prompt
    assert "x = 0" not in prompt
