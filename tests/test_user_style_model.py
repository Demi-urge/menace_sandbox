from __future__ import annotations
import menace.user_style_model as usm


def test_generate_requires_context_builder():
    class DummyDB:
        def fetch(self, n):  # pragma: no cover - simple stub
            return []

    model = usm.UserStyleModel(DummyDB())

    class DummyModel:
        def generate(self, **_):
            return [[0]]

    class DummyTokenizer:
        def __call__(self, text, return_tensors=None):  # pragma: no cover - simple stub
            return {"input_ids": [1]}

        def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - simple stub
            return "out"

    model.model = DummyModel()
    model.tokenizer = DummyTokenizer()

    try:
        model.generate("hi")  # type: ignore[misc]
    except TypeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("context_builder was not required")


def test_generate_injects_context(monkeypatch):
    captured: dict[str, str] = {}

    class DummyModel:
        def generate(self, **_):
            return [[0]]

    class DummyTokenizer:
        def __call__(self, text, return_tensors=None):
            captured["prompt"] = text
            return {"input_ids": [1]}

        def decode(self, ids, skip_special_tokens=True):
            return "out"

    class DummyBuilder:
        def build(self, query, include_vectors=False):  # pragma: no cover - simple stub
            return "context"

    def fake_compress(meta, **_):
        return {"snippet": "COMP-" + meta.get("snippet", "")}

    monkeypatch.setattr(usm, "compress_snippets", fake_compress)

    class DummyDB:
        def fetch(self, n):  # pragma: no cover - simple stub
            return []

    model = usm.UserStyleModel(DummyDB())
    model.model = DummyModel()
    model.tokenizer = DummyTokenizer()

    result = model.generate("hello", context_builder=DummyBuilder())
    assert result == "out"
    assert captured["prompt"] == "COMP-context\n\nhello"

