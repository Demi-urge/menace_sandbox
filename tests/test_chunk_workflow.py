from pathlib import Path
from typing import Any, Dict, List
import sys
import types

import chunking
from prompt_engine import PromptEngine
import chunking as pc


# Lightweight llm_interface stub to avoid heavy dependencies
llm_stub = types.ModuleType("llm_interface")


class _LLMClient:
    def __init__(self, model: str = "") -> None:
        self.model = model

    def generate(self, prompt):
        raise NotImplementedError


class _LLMResult:
    def __init__(self, text: str = "") -> None:
        self.text = text


llm_stub.LLMClient = _LLMClient
llm_stub.LLMResult = _LLMResult
llm_stub.Completion = _LLMResult


class _Prompt:
    def __init__(self, system: str = "", user: str = "", **kwargs) -> None:
        self.system = system
        self.user = user


llm_stub.Prompt = _Prompt
sys.modules.setdefault("llm_interface", llm_stub)


# Minimal vector_service stubs for PromptEngine
vec_mod = types.ModuleType("vector_service")
vec_mod.CognitionLayer = object
vec_mod.PatchLogger = object
vec_mod.VectorServiceError = Exception
sys.modules.setdefault("vector_service", vec_mod)
sys.modules.setdefault(
    "vector_service.retriever", types.ModuleType("vector_service.retriever")
)
sys.modules.setdefault(
    "vector_service.decorators", types.ModuleType("vector_service.decorators")
)

# Stub heavy dependencies for PromptEngine
sys.modules.setdefault(
    "gpt_memory", types.SimpleNamespace(GPTMemoryManager=object)
)
sys.modules.setdefault(
    "code_database", types.SimpleNamespace(PatchHistoryDB=object)
)


def test_chunk_file_ast_and_token_limits(tmp_path, monkeypatch):
    code = (
        "def a():\n    return 1\n\n"
        "def b():\n    return 2\n\n"
        "def c():\n    return 3\n"
    )
    path = tmp_path / "sample.py"  # path-ignore
    path.write_text(code)
    monkeypatch.setattr(chunking, "_count_tokens", lambda t: len(t.split()))
    chunks = chunking.chunk_file(path, 5)
    assert all(len(ch.text.split()) <= 10 for ch in chunks)
    assert [ch.start_line for ch in chunks] == [1, 4, 7]
    assert [ch.end_line for ch in chunks] == [2, 5, 8]


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, context_builder=None):  # pragma: no cover - simple stub
        self.calls += 1
        return types.SimpleNamespace(text="stub")


class DummyBuilder:
    def build(self, text: str):  # pragma: no cover - simple stub
        return ""


def test_summary_cache_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(chunking, "SNIPPET_CACHE_DIR", tmp_path)
    llm = DummyLLM()
    builder = DummyBuilder()
    code = "print('hello')"
    s1 = chunking.summarize_code(code, llm, context_builder=builder)
    assert s1 == "stub"
    assert llm.calls == 1
    s2 = chunking.summarize_code(code, llm, context_builder=builder)
    assert s2 == "stub"
    assert llm.calls == 1


class DummyRetriever:
    def __init__(self, records: List[Dict[str, Any]] | None = None) -> None:
        self.records = records or []

    def search(self, query: str, top_k: int):
        return self.records


def test_prompt_from_summaries_under_limit(tmp_path, monkeypatch):
    file = tmp_path / "big.py"  # path-ignore
    code = "def big():\n" + "    x = 0\n" * 2000
    file.write_text(code)

    summaries = [{"summary": "sumA"}, {"summary": "sumB"}]

    def fake_get_chunk_summaries(path: Path, limit: int, **kwargs):
        assert path == file
        assert limit == 20
        return summaries

    monkeypatch.setattr(pc, "get_chunk_summaries", fake_get_chunk_summaries)

    records = [{"score": 0.9, "metadata": {"summary": "irrelevant", "tests_passed": True}}]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        confidence_threshold=-1.0,
        token_threshold=50,
        chunk_token_threshold=20,
        context_builder=object(),
    )

    if pc._count_tokens(code) > engine.token_threshold:
        chunks = pc.get_chunk_summaries(file, engine.chunk_token_threshold)
        context = engine._trim_tokens(
            "\n".join(c["summary"] for c in chunks), engine.token_threshold
        )
    else:
        context = code

    prompt = engine.build_prompt(
        "do something", context=context, context_builder=engine.context_builder
    )
    assert "sumA" in prompt.user and "sumB" in prompt.user
    assert "x = 0" not in prompt.user
    assert pc._count_tokens(prompt.user) <= engine.token_threshold


def test_patch_application_on_chunked_file(tmp_path, monkeypatch):
    code = (
        "def a():\n    pass\n\n"
        "def b():\n    pass\n"
    )
    path = tmp_path / "big.py"  # path-ignore
    path.write_text(code)
    monkeypatch.setattr(chunking, "_count_tokens", lambda t: len(t.split()))
    chunks = chunking.chunk_file(path, 5)
    patches = ["# patch 1", "# patch 2"]
    lines = path.read_text().splitlines()
    for patch, chunk in reversed(list(zip(patches, chunks))):
        lines.insert(chunk.end_line, patch)
    path.write_text("\n".join(lines))
    text = path.read_text()
    assert "# patch 1" in text and "# patch 2" in text
