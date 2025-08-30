import logging
from prompt_engine import PromptEngine, DEFAULT_TEMPLATE


def test_retrieval_snippets_included(monkeypatch):
    records = [{"metadata": {
        "summary": "fixed bug",
        "diff": "changed logic",
        "outcome": "works",
        "tests_passed": True,
    }}]
    monkeypatch.setattr(PromptEngine, "_fetch_patches", staticmethod(lambda q, n: (records, 1.0)))
    prompt = PromptEngine.construct_prompt("desc")
    assert "Code summary: fixed bug" in prompt
    assert "Diff summary: changed logic" in prompt
    assert "Outcome: works (tests passed)" in prompt


def test_orders_by_roi_and_timestamp(monkeypatch):
    records = [
        {"metadata": {"roi_delta": 0.1, "summary": "low", "tests_passed": True}},
        {"metadata": {"roi_delta": 0.9, "summary": "high", "tests_passed": True}},
        {"metadata": {"ts": 1, "summary": "old fail", "tests_passed": False}},
        {"metadata": {"ts": 2, "summary": "new fail", "tests_passed": False}},
    ]
    monkeypatch.setattr(PromptEngine, "_fetch_patches", staticmethod(lambda q, n: (records, 1.0)))
    prompt = PromptEngine.construct_prompt("desc")
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    assert prompt.index("Code summary: new fail") < prompt.index("Code summary: old fail")


def test_fallback_on_low_confidence(monkeypatch, caplog):
    monkeypatch.setattr(PromptEngine, "_fetch_patches", staticmethod(lambda q, n: ([], 0.0)))
    with caplog.at_level(logging.INFO):
        prompt = PromptEngine.construct_prompt("desc")
    assert prompt == DEFAULT_TEMPLATE
    assert "falling back" in caplog.text.lower()


def test_retry_trace_included(monkeypatch):
    records = [{"metadata": {"summary": "foo", "tests_passed": True}}]
    monkeypatch.setattr(PromptEngine, "_fetch_patches", staticmethod(lambda q, n: (records, 1.0)))
    trace = "Traceback: fail"
    prompt = PromptEngine.construct_prompt("desc", retry_trace=trace)
    expected = f"Previous attempt failed with {trace}; seek alternative solution."
    assert expected in prompt
