import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import neurosales.response_generation as rg  # noqa: E402
from neurosales.response_generation import (  # noqa: E402
    ResponseCandidateGenerator,
    redundancy_filter,
)


def test_redundancy_filter_removes_duplicates():
    res = redundancy_filter(["hello world", "hello world", "hello there"], threshold=0.5)
    assert len(res) == 2
    assert res[0] == "hello world"


def test_generate_candidates_pool():
    class DummyBuilder:
        def build(self, message, **_):
            return ""

        def refresh_db_weights(self):
            return None

    gen = ResponseCandidateGenerator(context_builder=DummyBuilder())
    gen.add_past_response("Sure, I can assist you.")
    gen.add_past_response("Let me show you how to proceed.")
    candidates = gen.generate_candidates("I need help", ["Hi"], "helper")
    assert candidates
    assert len(candidates) == len(set(candidates))


def test_dynamic_candidates_include_context(monkeypatch):
    class DummyBuilder:
        def __init__(self):
            self.calls = []

        def build(self, message, **_):
            self.calls.append(message)
            return "RAWCTX"

        def refresh_db_weights(self):
            return None

    class DummyTokenizer:
        def encode(self, prompt, return_tensors=None):
            return types.SimpleNamespace(prompt=prompt, shape=(1, len(prompt.split())))

        def decode(self, output, skip_special_tokens=True):
            return output.prompt

    class DummyModel:
        def generate(self, input_ids, **_):
            return [input_ids]

    def fake_compress(meta, **_):
        txt = meta.get("snippet", "")
        return {"snippet": txt.replace("RAW", "COMP")}

    monkeypatch.setattr(rg, "compress_snippets", fake_compress)

    builder = DummyBuilder()
    gen = ResponseCandidateGenerator(context_builder=builder)
    gen.tokenizer = DummyTokenizer()
    gen.model = DummyModel()
    res = gen._dynamic_candidates("hello", ["hi"], "arch", n=1)
    assert builder.calls == ["hello"]
    assert "COMPCTX" in res[0]
    assert "RAWCTX" not in res[0]
