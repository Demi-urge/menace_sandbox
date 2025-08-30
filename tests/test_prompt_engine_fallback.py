import logging
from prompt_engine import PromptEngine, DEFAULT_TEMPLATE


def test_construct_prompt_orders_by_roi_and_timestamp(monkeypatch):
    records = [
        {"metadata": {"roi_delta": 0.1, "summary": "low", "tests_passed": True}},
        {"metadata": {"roi_delta": 0.9, "summary": "high", "tests_passed": True}},
        {"metadata": {"ts": 1, "summary": "old fail", "tests_passed": False}},
        {"metadata": {"ts": 2, "summary": "new fail", "tests_passed": False}},
    ]

    def fake_fetch(q, n):
        return records, 0.9

    monkeypatch.setattr(PromptEngine, "_fetch_patches", staticmethod(fake_fetch))
    prompt = PromptEngine.construct_prompt("desc")
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    assert prompt.index("Code summary: new fail") < prompt.index("Code summary: old fail")


def test_construct_prompt_fallback_on_low_confidence(monkeypatch, caplog):
    monkeypatch.setattr(
        PromptEngine,
        "_fetch_patches",
        staticmethod(lambda q, n: ([], 0.0)),
    )
    monkeypatch.setattr(PromptEngine, "_static_prompt", staticmethod(lambda: DEFAULT_TEMPLATE))
    events = []
    monkeypatch.setattr("prompt_engine.audit_log_event", lambda e, d: events.append((e, d)))
    with caplog.at_level(logging.INFO):
        prompt = PromptEngine.construct_prompt("desc")
    assert prompt == DEFAULT_TEMPLATE
    assert "falling back" in caplog.text.lower()
    assert events and events[0][0] == "prompt_engine_fallback"
    assert events[0][1]["reason"] == "low_confidence"


def test_construct_prompt_fallback_on_retrieval_error(monkeypatch, caplog):
    def boom(q, n):
        raise RuntimeError("boom")

    monkeypatch.setattr(PromptEngine, "_fetch_patches", staticmethod(boom))
    monkeypatch.setattr(PromptEngine, "_static_prompt", staticmethod(lambda: DEFAULT_TEMPLATE))
    events = []
    monkeypatch.setattr("prompt_engine.audit_log_event", lambda e, d: events.append((e, d)))
    with caplog.at_level(logging.ERROR):
        prompt = PromptEngine.construct_prompt("desc")
    assert prompt == DEFAULT_TEMPLATE
    assert "boom" in caplog.text.lower()
    assert events and events[0][1]["reason"] == "retrieval_error"
