import pytest

from local_model_wrapper import LocalModelWrapper
from prompt_types import Prompt


class DummyModel:
    def generate(self, input_ids, **_):  # pragma: no cover - simple stub
        return [input_ids[0] + [" out"]]


class DummyTokenizer:
    def encode(self, text, return_tensors=None):  # pragma: no cover - simple stub
        return [[text]]

    def decode(self, tokens, skip_special_tokens=True):  # pragma: no cover - simple stub
        return "".join(tokens)


def test_wrapper_requires_context_builder():
    wrapper = LocalModelWrapper(DummyModel(), DummyTokenizer())
    with pytest.raises(TypeError):
        wrapper.generate("hi")  # type: ignore[misc]


def test_string_prompt_uses_builder():
    wrapper = LocalModelWrapper(DummyModel(), DummyTokenizer())

    class Builder:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def build_prompt(self, query: str, **_):
            self.queries.append(query)
            return Prompt(user=query, examples=["ctx"])

    builder = Builder()
    out = wrapper.generate("hi", context_builder=builder)
    assert builder.queries == ["hi"]
    assert out.strip() == "out"
