import pytest

from llm_interface import LLMClient, LLMResult, Prompt


class DummyClient(LLMClient):
    def __init__(self):
        super().__init__("dummy", log_prompts=False)

    def _generate(self, prompt, *, context_builder):  # type: ignore[override]
        return LLMResult(text="ok")


def test_generate_requires_context_builder_markers():
    client = DummyClient()
    with pytest.raises(ValueError):
        client.generate(Prompt(text="hi"), context_builder=object())

    prompt = Prompt(text="hi", metadata={"vector_confidences": [1.0]})
    res = client.generate(prompt, context_builder=object())
    assert res.text == "ok"
