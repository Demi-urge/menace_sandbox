from typing import Any, Dict, List

from prompt_engine import PromptEngine


class DummyRetriever:
    def __init__(self, records: List[Dict[str, Any]] | None = None) -> None:
        self.records = records or []

    def search(self, query: str, top_k: int):
        return self.records


def test_build_prompt_includes_summaries():
    records = [{"score": 0.9, "metadata": {"summary": "irrelevant", "tests_passed": True}}]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        confidence_threshold=-1.0,
        context_builder=object(),
    )
    prompt = engine.build_prompt(
        "do things", summaries=["sumA", "sumB"], context_builder=engine.context_builder
    )
    text = str(prompt)
    assert "sumA" in text and "sumB" in text


def test_build_prompt_skips_summaries_when_absent():
    records = [{"score": 0.9, "metadata": {"summary": "irrelevant", "tests_passed": True}}]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        confidence_threshold=-1.0,
        context_builder=object(),
    )
    prompt = engine.build_prompt("do things", context_builder=engine.context_builder)
    text = str(prompt)
    assert "irrelevant" in text
    assert "sumA" not in text and "sumB" not in text
