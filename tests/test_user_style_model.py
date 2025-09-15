from __future__ import annotations
import sys
import types
from prompt_types import Prompt


class _MirrorStub:
    def fetch(self, n):  # pragma: no cover - simple stub
        return []


dummy_mb = types.ModuleType("menace.mirror_bot")
dummy_mb.MirrorDB = _MirrorStub
sys.modules.setdefault("menace.mirror_bot", dummy_mb)

import menace.user_style_model as usm


def test_generate_requires_context_builder():
    class DummyDB:
        def fetch(self, n):  # pragma: no cover - simple stub
            return []

    model = usm.UserStyleModel(DummyDB())

    class DummyModel:
        def generate(self, input_ids, **_):
            return [input_ids]

    class DummyTokenizer:
        def encode(self, text, return_tensors=None):  # pragma: no cover - simple stub
            return [[1]]

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
        def generate(self, input_ids, **_):
            return [input_ids]

    class DummyTokenizer:
        def encode(self, text, return_tensors=None):
            captured["prompt"] = text
            return [[1]]

        def decode(self, ids, skip_special_tokens=True):
            return "out"

    class DummyBuilder:
        def build_prompt(self, query, **_):  # pragma: no cover - simple stub
            return Prompt(user=query, examples=["context"])

    class DummyDB:
        def fetch(self, n):  # pragma: no cover - simple stub
            return []

    model = usm.UserStyleModel(DummyDB())
    model.model = DummyModel()
    model.tokenizer = DummyTokenizer()

    result = model.generate("hello", context_builder=DummyBuilder())
    assert result == "out"
    assert captured["prompt"] == "context\n\nhello"
