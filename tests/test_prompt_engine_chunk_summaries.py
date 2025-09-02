import sys
import types
from pathlib import Path
from typing import Any, Dict, List

# Stub heavy dependencies before importing PromptEngine
sys.modules.setdefault("gpt_memory", types.SimpleNamespace(GPTMemoryManager=object))
sys.modules.setdefault("code_database", types.SimpleNamespace(PatchHistoryDB=object))

from prompt_engine import PromptEngine  # noqa: E402
import prompt_chunking as pc  # noqa: E402


class DummyRetriever:
    def __init__(self, records: List[Dict[str, Any]] | None = None) -> None:
        self.records = records or []

    def search(self, query: str, top_k: int):
        return self.records


def test_prompt_engine_uses_file_summaries_when_limit_exceeded(tmp_path, monkeypatch):
    file = tmp_path / "big.py"
    code = "def big():\n" + "    x = 0\n" * 2000
    file.write_text(code)

    summaries = [{"summary": "sumA"}, {"summary": "sumB"}]
    called: list[tuple[Path, int]] = []

    def fake_get_chunk_summaries(path: Path, limit: int):
        called.append((path, limit))
        return summaries

    monkeypatch.setattr(pc, "get_chunk_summaries", fake_get_chunk_summaries)

    records = [{"score": 0.9, "metadata": {"summary": "irrelevant", "tests_passed": True}}]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        confidence_threshold=-1.0,
        token_threshold=50,
        chunk_token_threshold=20,
    )

    if pc._count_tokens(code) > engine.token_threshold:
        chunks = pc.get_chunk_summaries(file, engine.chunk_token_threshold)
        ctx = engine._trim_tokens("\n".join(c["summary"] for c in chunks), engine.token_threshold)
    else:
        ctx = code

    prompt = engine.build_prompt("do something", context=ctx)
    assert called
    assert "sumA" in prompt and "sumB" in prompt
    assert "x = 0" not in prompt
