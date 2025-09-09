import types
from code_summarizer import summarize_code


class DummyBuilder:
    def __init__(self):
        self.calls = 0

    def build_context(self, *args, **kwargs):  # pragma: no cover - simple stub
        self.calls += 1
        return ""


def test_summarize_code_micro_model(monkeypatch):
    """Ensure the micro-model path is used and honours token limits."""

    def fake_summarize_diff(before: str, after: str, max_new_tokens: int = 128) -> str:
        assert before == ""
        assert after == "print('hi')"
        return "one two three four five six seven"

    monkeypatch.setattr(
        "micro_models.diff_summarizer.summarize_diff", fake_summarize_diff
    )

    class DummyClient:
        def __init__(self, *a, **kw):  # pragma: no cover - should not be called
            raise AssertionError("LLM client should not be used")

    monkeypatch.setattr("local_client.OllamaClient", DummyClient)

    summary = summarize_code("print('hi')", context_builder=DummyBuilder(), max_summary_tokens=5)
    assert summary.split() == ["one", "two", "three", "four", "five"]


def test_summarize_code_llm_client(monkeypatch):
    """Fallback to a local LLM when the micro model is unavailable."""

    monkeypatch.setattr(
        "micro_models.diff_summarizer.summarize_diff", lambda *a, **k: ""
    )

    class DummyResult(types.SimpleNamespace):
        text: str = "alpha beta gamma delta epsilon zeta eta"

    class DummyClient:
        def __init__(self, *a, **k):
            pass

        def generate(self, prompt, context_builder=None):  # pragma: no cover - trivial
            return DummyResult()

    monkeypatch.setattr("local_client.OllamaClient", DummyClient)

    summary = summarize_code("print('hi')", context_builder=DummyBuilder(), max_summary_tokens=5)
    assert summary.split() == ["alpha", "beta", "gamma", "delta", "epsilon"]


def test_summarize_code_heuristic(monkeypatch):
    monkeypatch.setattr("micro_models.diff_summarizer.summarize_diff", lambda *a, **k: "")

    class DummyClient:
        def __init__(self, *a, **k):
            pass

        def generate(self, prompt):
            return types.SimpleNamespace(text="")

    monkeypatch.setattr("local_client.OllamaClient", DummyClient)

    code = "# header\n\nclass Foo:\n    pass\n"
    summary = summarize_code(code, context_builder=DummyBuilder(), max_summary_tokens=10)
    assert summary.startswith("class Foo")
