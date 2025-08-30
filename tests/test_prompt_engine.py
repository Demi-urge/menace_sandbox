from typing import Any, Dict, List

from prompt_engine import PromptEngine, DEFAULT_TEMPLATE
from vector_service.retriever import FallbackResult


class DummyRetriever:
    """Basic retriever returning predefined records."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def search(self, query: str, top_k: int):  # pragma: no cover - trivial
        return self.records[:top_k]


class DummyFallbackRetriever:
    """Retriever that always returns a fallback result."""

    def search(self, query: str, top_k: int):  # pragma: no cover - trivial
        return FallbackResult("low_confidence", [], confidence=0.1)


def _record(score: float, **meta: Any) -> Dict[str, Any]:
    """Helper to build retriever records with ``score`` and metadata."""

    return {"score": score, "metadata": meta}


def test_prompt_engine_sections_and_ranking():
    records = [
        _record(0.9, raroi=0.4, summary="low", tests_passed=True, ts=1),
        _record(0.8, raroi=0.9, summary="high", tests_passed=True, ts=2),
        _record(0.7, summary="fail", tests_passed=False, ts=2),
    ]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        top_n=3,
        confidence_threshold=0.0,
    )
    prompt = engine.build_prompt("desc")

    # Sections from build_snippets are present
    assert "Successful example:" in prompt
    assert "Avoid pattern:" in prompt

    # Ranking respects ROI values
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    # Failure snippet appears after successes
    assert prompt.rindex("Code summary: fail") > prompt.index("Avoid pattern:")


def test_prompt_engine_handles_retry_trace():
    records = [_record(1.0, summary="foo", tests_passed=True, raroi=0.5)]
    engine = PromptEngine(patch_retriever=DummyRetriever(records))
    trace = "Previous failure:\nTraceback: fail\nPlease attempt a different solution."
    prompt = engine.build_prompt("goal", retry_trace=trace)

    assert prompt.count("Previous failure:") == 1
    assert "Traceback: fail" in prompt
    assert prompt.strip().endswith("Please attempt a different solution.")


def test_prompt_engine_uses_template_on_low_confidence(monkeypatch):
    records = [_record(0.0, summary="bad", tests_passed=True)]
    engine = PromptEngine(patch_retriever=DummyRetriever(records))
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    prompt = engine.build_prompt("goal")
    assert prompt == DEFAULT_TEMPLATE


def test_prompt_engine_handles_fallback_result(monkeypatch):
    engine = PromptEngine(patch_retriever=DummyFallbackRetriever())
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    prompt = engine.build_prompt("goal")
    assert prompt == DEFAULT_TEMPLATE


def test_weighted_scoring_alters_ordering():
    records = [
        _record(0.0, raroi=1.0, summary="roi", tests_passed=True, ts=1),
        _record(0.0, raroi=0.2, summary="recent", tests_passed=True, ts=2),
    ]
    engine = PromptEngine(
        patch_retriever=DummyRetriever(records),
        top_n=2,
        roi_weight=1.0,
        recency_weight=0.0,
        confidence_threshold=-10,
    )

    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "roi"

    engine.roi_weight = 0.0
    engine.recency_weight = 1.0
    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "recent"


def test_roi_tag_weights_adjust_ranking():
    records = [
        _record(
            0.0,
            raroi=0.5,
            summary="good",
            tests_passed=True,
            ts=1,
            roi_tag="high-ROI",
        ),
        _record(
            0.0,
            raroi=0.5,
            summary="bad",
            tests_passed=True,
            ts=1,
            roi_tag="bug-introduced",
        ),
    ]
    engine = PromptEngine(
        patch_retriever=DummyRetriever(records),
        top_n=2,
        roi_weight=0.0,
        recency_weight=0.0,
        confidence_threshold=-10,
    )

    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "good"

    engine.roi_tag_weights = {"high-ROI": -1.0, "bug-introduced": 1.0}
    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "bad"

