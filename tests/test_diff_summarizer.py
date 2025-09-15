import micro_models.diff_summarizer as ds
from prompt_types import Prompt


class DummyTokenizer:
    def __init__(self):
        self.prompt = None

    def __call__(self, prompt, return_tensors=None):  # pragma: no cover - simple stub
        self.prompt = prompt
        return {"input_ids": [0]}

    def decode(self, tokens, skip_special_tokens=True):  # pragma: no cover - simple stub
        return "Summary: ok"


class DummyModel:
    def __init__(self):
        self.device = "cpu"

    def generate(self, **kwargs):  # pragma: no cover - simple stub
        return [[1]]


def test_context_inclusion(monkeypatch):
    tok = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr(ds, "_load_model", lambda: (tok, model))
    monkeypatch.setattr(ds, "torch", None)

    captured = {}

    class DummyBuilder:
        def build_prompt(self, goal, *, context, intent_metadata=None, top_k=0, **kwargs):
            captured.update(
                goal=goal,
                context=context,
                intent=intent_metadata,
                top_k=top_k,
                builder=self,
            )
            return Prompt(user=f"retrieved\n{context}", system="PAY")

    builder = DummyBuilder()
    res = ds.summarize_diff("before", "after", context_builder=builder)
    assert res.startswith("ok")
    assert tok.prompt.startswith("PAY\n")
    assert "retrieved" in tok.prompt
    assert captured["context"] == "before\nafter"
    assert captured["intent"]["hint"] == "after"
    assert captured["builder"] is builder
    assert captured["top_k"] == 5


def test_hint_weighting(monkeypatch):
    tok = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr(ds, "_load_model", lambda: (tok, model))
    monkeypatch.setattr(ds, "torch", None)

    topks = []

    class DummyBuilder:
        def build_prompt(self, goal, *, context, intent_metadata=None, top_k=0, **kwargs):
            topks.append(top_k)
            return Prompt(user="", system="")

    builder = DummyBuilder()
    ds.summarize_diff("before", "after", context_builder=builder)
    ds.summarize_diff("", "", context_builder=builder)
    assert topks == [5, 0]
