import logging
import types
import sys
from typing import Any, Dict, List

# Stub vector_service to avoid heavy dependencies during import
sys.modules.setdefault("vector_service", types.ModuleType("vector_service"))

import prompt_engine as pe


class RoiTag:
    HIGH_ROI = types.SimpleNamespace(value="high-ROI")
    BUG_INTRODUCED = types.SimpleNamespace(value="bug-introduced")
    SUCCESS = types.SimpleNamespace(value="success")
    LOW_ROI = types.SimpleNamespace(value="low-ROI")
    NEEDS_REVIEW = types.SimpleNamespace(value="needs-review")
    BLOCKED = types.SimpleNamespace(value="blocked")

    @classmethod
    def validate(cls, value):
        return types.SimpleNamespace(value=value)

pe.RoiTag = RoiTag
PromptEngine = pe.PromptEngine
DEFAULT_TEMPLATE = pe.DEFAULT_TEMPLATE


class DummyBuilder:
    def __init__(self, *_, **__):
        self.roi_tracker = None

    def _count_tokens(self, text: str) -> int:
        return len(str(text).split())


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
            raroi=0.5,
            roi_tag=RoiTag.HIGH_ROI.value,
        )
    ]
    engine = PromptEngine(
        retriever=DummyRetriever(records), context_builder=DummyBuilder()
    )
    prompt = engine.build_prompt("desc", context_builder=engine.context_builder)
    assert "Given the following pattern, desc" in prompt.text
    assert "Given the following pattern:" in prompt.text
    assert "Code summary: fixed bug" in prompt.text
    assert "Diff summary: changed logic" in prompt.text
    assert "Outcome: works (tests passed)" in prompt.text
    assert prompt.vector_confidences and prompt.vector_confidences[0] > 0
    assert RoiTag.HIGH_ROI.value in prompt.outcome_tags


def test_custom_headers_codex_style():
    records = [
        _record(1.0, summary="ok", tests_passed=True, ts=1),
        _record(1.0, summary="fail", tests_passed=False, ts=2),
    ]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        context_builder=DummyBuilder(),
        confidence_threshold=0.0,
        success_header="Correct example:",
        failure_header="Incorrect example:",
        trainer=object(),
    )
    prompt = engine.build_prompt("desc", context_builder=engine.context_builder)
    text = str(prompt)
    assert "Correct example:" in text
    assert "Incorrect example:" in text


def test_orders_by_roi_and_timestamp():
    records = [
        _record(
            1.0,
            roi_tag=RoiTag.LOW_ROI.value,
            summary="low",
            tests_passed=True,
            raroi=0.4,
        ),
        _record(
            1.0,
            roi_tag=RoiTag.HIGH_ROI.value,
            summary="high",
            tests_passed=True,
            raroi=0.9,
        ),
        _record(1.0, ts=1, summary="old fail", tests_passed=False),
        _record(1.0, ts=2, summary="new fail", tests_passed=False),
    ]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        context_builder=DummyBuilder(),
        confidence_threshold=-1.0,
    )
    prompt = engine.build_prompt("desc", context_builder=engine.context_builder)
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    assert prompt.index("Code summary: new fail") < prompt.index("Code summary: old fail")


def test_build_snippets_sorted_by_score():
    patches = [
        {
            "metadata": {
                "summary": "bad",
                "roi_tag": RoiTag.BUG_INTRODUCED.value,
                "tests_passed": True,
                "raroi": -1.0,
            }
        },
        {
            "metadata": {
                "summary": "good",
                "roi_tag": RoiTag.HIGH_ROI.value,
                "tests_passed": True,
                "raroi": 1.0,
            }
        },
    ]
    engine = PromptEngine(confidence_threshold=-2.0, context_builder=DummyBuilder())
    lines = engine.build_snippets(patches)
    text = "\n".join(lines)
    assert text.index("Code summary: good") < text.index("Code summary: bad")


def test_fallback_on_low_confidence(caplog, monkeypatch):
    records: List[Dict[str, Any]] = []
    engine = PromptEngine(
        retriever=DummyRetriever(records), context_builder=DummyBuilder()
    )
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    with caplog.at_level(logging.INFO):
        prompt = engine.build_prompt("desc", context_builder=engine.context_builder)
    assert prompt.user == DEFAULT_TEMPLATE
    assert "falling back" in caplog.text.lower()


def test_retry_trace_included():
    records = [_record(1.0, summary="foo", tests_passed=True, raroi=0.5)]
    engine = PromptEngine(
        retriever=DummyRetriever(records), context_builder=DummyBuilder()
    )
    trace = "Traceback: fail"
    prompt = engine.build_prompt("desc", retry_trace=trace, context_builder=engine.context_builder)
    expected = "Previous failure:\nTraceback: fail\nPlease attempt a different solution."
    assert expected in prompt.text


def test_retry_trace_idempotent():
    records = [_record(1.0, summary="foo", tests_passed=True, raroi=0.5)]
    engine = PromptEngine(
        retriever=DummyRetriever(records), context_builder=DummyBuilder()
    )
    trace = (
        "Previous failure:\nTraceback: boom\nPlease attempt a different solution."
    )
    prompt = engine.build_prompt("desc", retry_trace=trace, context_builder=engine.context_builder)
    assert prompt.text.count("Traceback: boom") == 1
    assert prompt.text.count("Previous failure:") == 1


def test_roi_tag_positive_effect_on_score():
    engine = PromptEngine(
        retriever=DummyRetriever([]), context_builder=DummyBuilder()
    )
    baseline = engine._score_snippet({})
    high = engine._score_snippet({"roi_tag": RoiTag.HIGH_ROI.value})
    assert high > baseline


def test_roi_tag_negative_effect_on_score():
    engine = PromptEngine(
        retriever=DummyRetriever([]), context_builder=DummyBuilder()
    )
    baseline = engine._score_snippet({})
    bad = engine._score_snippet({"roi_tag": RoiTag.BUG_INTRODUCED.value})
    assert bad < baseline
