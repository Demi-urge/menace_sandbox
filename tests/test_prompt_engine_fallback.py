from typing import Any, Dict, List, Tuple
import logging

from prompt_engine import PromptEngine, DEFAULT_TEMPLATE


class DummyRetriever:
    def __init__(self, records: List[Dict[str, Any]], boom: bool = False):
        self.records = records
        self.boom = boom

    def search(self, query: str, top_k: int):
        if self.boom:
            raise RuntimeError("boom")
        return self.records[:top_k]


def _record(score: float, **meta: Any) -> Dict[str, Any]:
    return {"score": score, "metadata": meta}


def test_construct_prompt_orders_by_roi_and_timestamp():
    records = [
        _record(1.0, roi_delta=0.1, summary="low", tests_passed=True),
        _record(1.0, roi_delta=0.9, summary="high", tests_passed=True),
        _record(1.0, ts=1, summary="old fail", tests_passed=False),
        _record(1.0, ts=2, summary="new fail", tests_passed=False),
    ]
    engine = PromptEngine(retriever=DummyRetriever(records))
    prompt = engine.build_prompt("desc")
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    assert prompt.index("Code summary: new fail") < prompt.index("Code summary: old fail")


def test_construct_prompt_fallback_on_low_confidence(monkeypatch, caplog):
    engine = PromptEngine(retriever=DummyRetriever([]))
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    events: List[Tuple[str, Dict[str, Any]]] = []
    monkeypatch.setattr("prompt_engine.audit_log_event", lambda e, d: events.append((e, d)))
    with caplog.at_level(logging.INFO):
        prompt = engine.build_prompt("desc")
    assert prompt == DEFAULT_TEMPLATE
    assert "falling back" in caplog.text.lower()
    assert events and events[0][0] == "prompt_engine_fallback"
    assert events[0][1]["reason"] == "low_confidence"


def test_construct_prompt_fallback_on_retrieval_error(monkeypatch, caplog):
    engine = PromptEngine(retriever=DummyRetriever([], boom=True))
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    events: List[Tuple[str, Dict[str, Any]]] = []
    monkeypatch.setattr("prompt_engine.audit_log_event", lambda e, d: events.append((e, d)))
    with caplog.at_level(logging.ERROR):
        prompt = engine.build_prompt("desc")
    assert prompt == DEFAULT_TEMPLATE
    assert "boom" in caplog.text.lower()
    assert events and events[0][1]["reason"] == "retrieval_error"
