from prompt_engine import PromptEngine, DEFAULT_TEMPLATE
from vector_service.roi_tags import RoiTag
from typing import Any, Dict, List
import logging


class DummyRetriever:
    def __init__(self, records):
        self.records = records

    def search(self, query, top_k):
        return self.records[:top_k]


def _record(score: float, **meta: Any) -> Dict[str, Any]:
    return {"score": score, "metadata": meta}


def test_retrieval_snippets_included():
    records = [
        _record(
            1.0,
            summary="fixed bug",
            diff="changed logic",
            outcome="works",
            tests_passed=True,
        )
    ]
    engine = PromptEngine(retriever=DummyRetriever(records))
    prompt = engine.build_prompt("desc")
    assert "Successful example:" in prompt
    assert "Code summary: fixed bug" in prompt
    assert "Diff summary: changed logic" in prompt
    assert "Outcome: works (tests passed)" in prompt


def test_orders_by_roi_and_timestamp():
    records = [
        _record(1.0, roi_tag=RoiTag.LOW_ROI.value, summary="low", tests_passed=True),
        _record(1.0, roi_tag=RoiTag.HIGH_ROI.value, summary="high", tests_passed=True),
        _record(1.0, ts=1, summary="old fail", tests_passed=False),
        _record(1.0, ts=2, summary="new fail", tests_passed=False),
    ]
    engine = PromptEngine(retriever=DummyRetriever(records))
    prompt = engine.build_prompt("desc")
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    assert prompt.index("Code summary: new fail") < prompt.index("Code summary: old fail")


def test_build_snippets_sorted_by_score():
    patches = [
        {
            "metadata": {
                "summary": "bad",
                "roi_tag": RoiTag.BUG_INTRODUCED.value,
                "tests_passed": True,
            }
        },
        {
            "metadata": {
                "summary": "good",
                "roi_tag": RoiTag.HIGH_ROI.value,
                "tests_passed": True,
            }
        },
    ]
    engine = PromptEngine()
    lines = engine.build_snippets(patches)
    text = "\n".join(lines)
    assert text.index("Code summary: good") < text.index("Code summary: bad")


def test_fallback_on_low_confidence(caplog, monkeypatch):
    records: List[Dict[str, Any]] = []
    engine = PromptEngine(retriever=DummyRetriever(records))
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    with caplog.at_level(logging.INFO):
        prompt = engine.build_prompt("desc")
    assert prompt == DEFAULT_TEMPLATE
    assert "falling back" in caplog.text.lower()


def test_retry_trace_included():
    records = [_record(1.0, summary="foo", tests_passed=True)]
    engine = PromptEngine(retriever=DummyRetriever(records))
    trace = "Traceback: fail"
    prompt = engine.build_prompt("desc", retry_info=trace)
    expected = "Previous attempt failed with:\nTraceback: fail\nTry a different approach."
    assert expected in prompt


def test_retry_trace_idempotent():
    records = [_record(1.0, summary="foo", tests_passed=True)]
    engine = PromptEngine(retriever=DummyRetriever(records))
    trace = (
        "Previous attempt failed with:\nTraceback: boom\nTry a different approach."
    )
    prompt = engine.build_prompt("desc", retry_info=trace)
    assert prompt.count("Traceback: boom") == 1
    assert prompt.count("Previous attempt failed with:") == 1
