import types
from code_summarizer import summarize_code


class DummyBuilder:
    def build_prompt(self, text, **kwargs):  # pragma: no cover - simple stub
        return types.SimpleNamespace(user=text, metadata={}, examples=[])


def test_summarize_code_micro_model(monkeypatch):
    """Ensure the micro-model path is used and honours token limits."""

    def fake_summarize_diff(
        before: str, after: str, max_new_tokens: int = 128, *, context_builder=None
    ) -> str:
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


def test_summarize_code_enriched_context(monkeypatch):
    monkeypatch.setattr("micro_models.diff_summarizer.summarize_diff", lambda *a, **k: "")

    captured = {}

    class DummyClient:
        def __init__(self, *a, **k):
            pass

        def generate(self, prompt, context_builder=None):
            captured["prompt"] = prompt
            return types.SimpleNamespace(text="summary")

    monkeypatch.setattr("local_client.OllamaClient", DummyClient)

    class RichBuilder:
        def build_prompt(self, text, *, intent=None, top_k=5):
            from prompt_types import Prompt

            meta = {
                "vector_confidences": [0.7],
                "vectors": [("s", "id", 0.7)],
            }
            prompt = Prompt(user=text, metadata=meta)
            prompt.vector_confidence = 0.7
            return prompt

    builder = RichBuilder()
    summary = summarize_code("print('hi')", context_builder=builder, max_summary_tokens=5)
    assert summary == "summary"
    prompt = captured["prompt"]
    assert prompt.metadata["vectors"] == [("s", "id", 0.7)]
    assert prompt.metadata["vector_confidences"] == [0.7]
    assert prompt.vector_confidence == 0.7


def test_summarize_code_no_builder_falls_back(monkeypatch):
    monkeypatch.setattr("micro_models.diff_summarizer.summarize_diff", lambda *a, **k: "")

    class DummyClient:
        def __init__(self, *a, **k):  # pragma: no cover - should not run
            raise AssertionError("LLM client should not be used")

    monkeypatch.setattr("local_client.OllamaClient", DummyClient)

    code = "# header\n\nclass Foo:\n    pass\n"
    summary = summarize_code(code, context_builder=None, max_summary_tokens=10)
    assert summary.startswith("class Foo")
