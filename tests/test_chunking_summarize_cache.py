import hashlib

from llm_interface import LLMClient, LLMResult

import chunking as pc


class DummyLLM(LLMClient):
    def __init__(self):
        super().__init__(model="dummy")
        self.calls = 0

    def _generate(self, prompt, context_builder=None):  # pragma: no cover - simple stub
        self.calls += 1
        return LLMResult(text="stub summary")


class DummyBuilder:
    def build(self, text: str):  # pragma: no cover - simple stub
        return ""


def test_summarize_code_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "SNIPPET_CACHE_DIR", tmp_path)
    llm = DummyLLM()
    builder = DummyBuilder()
    code = "print('hello')"

    summary1 = pc.summarize_code(code, llm, context_builder=builder)
    assert summary1 == "stub summary"
    assert llm.calls == 1

    summary2 = pc.summarize_code(code, llm, context_builder=builder)
    assert summary2 == "stub summary"
    assert llm.calls == 1

    digest = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
    assert pc._load_snippet_summary(digest) == "stub summary"
