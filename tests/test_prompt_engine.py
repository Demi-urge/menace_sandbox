from typing import Any, Dict, List

from prompt_engine import PromptEngine, DEFAULT_TEMPLATE
from vector_service.roi_tags import RoiTag


class DummyRetriever:
    """Simple retriever returning predefined records."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def search(self, query: str, top_k: int):  # pragma: no cover - trivial
        return self.records[:top_k]


def _record(score: float, **meta: Any) -> Dict[str, Any]:
    """Return retriever record with ``score`` and ``metadata``."""

    return {"score": score, "metadata": meta}


def test_prompt_engine_ranks_snippets_by_roi_and_timestamp():
    records = [
        _record(0.9, roi_tag=RoiTag.LOW_ROI.value, summary="low", tests_passed=True, ts=1),
        _record(0.8, roi_tag=RoiTag.HIGH_ROI.value, summary="high", tests_passed=True, ts=2),
        _record(0.2, summary="old fail", tests_passed=False, ts=1),
        _record(0.1, summary="new fail", tests_passed=False, ts=2),
    ]
    engine = PromptEngine(retriever=DummyRetriever(records), top_n=4)
    prompt = engine.build_prompt("desc")
    assert "Successful example:" in prompt
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    assert prompt.index("Code summary: new fail") < prompt.index("Code summary: old fail")


def test_prompt_engine_falls_back_when_confidence_low(monkeypatch):
    records = [_record(0.0, summary="bad", tests_passed=True)]
    engine = PromptEngine(retriever=DummyRetriever(records))
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    prompt = engine.build_prompt("desc")
    assert prompt == DEFAULT_TEMPLATE


def test_prompt_engine_includes_failure_trace():
    records = [_record(1.0, summary="foo", tests_passed=True)]
    engine = PromptEngine(retriever=DummyRetriever(records))
    trace = "Traceback: fail"
    prompt = engine.build_prompt("goal", retry_info=trace)
    expected = (
        "Previous attempt failed with:\n"
        "Traceback: fail\n"
        "Try a different approach."
    )
    assert expected in prompt
