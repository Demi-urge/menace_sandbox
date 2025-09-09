import pytest

from local_model_wrapper import LocalModelWrapper


class DummyModel:
    def generate(self, input_ids, **_):  # pragma: no cover - simple stub
        return [input_ids]


class DummyTokenizer:
    def encode(self, text, return_tensors=None):  # pragma: no cover - simple stub
        return text

    def decode(self, tokens, skip_special_tokens=True):  # pragma: no cover - simple stub
        return tokens


def test_wrapper_requires_context_builder():
    wrapper = LocalModelWrapper(DummyModel(), DummyTokenizer())
    with pytest.raises(TypeError):
        wrapper.generate("hi")  # type: ignore[misc]
