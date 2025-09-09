import micro_models.diff_summarizer as ds


class DummyBuilder:
    def __init__(self):
        self.calls = []

    def build_context(self, text):
        self.calls.append(text)
        return "EXTRA"


class DummyTokenizer:
    def __init__(self):
        self.prompt = None

    def __call__(self, prompt, return_tensors=None):
        self.prompt = prompt
        return {"input_ids": [0]}

    def decode(self, tokens, skip_special_tokens=True):
        return "Summary: ok"


class DummyModel:
    def __init__(self):
        self.device = "cpu"

    def generate(self, **kwargs):
        return [[1]]


def test_context_builder_prepended(monkeypatch):
    tok = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr(ds, "_load_model", lambda: (tok, model))
    monkeypatch.setattr(ds, "torch", None)

    builder = DummyBuilder()
    res = ds.summarize_diff("before", "after", context_builder=builder)
    assert res == "ok"
    assert builder.calls and builder.calls[0] == "before\nafter"
    assert tok.prompt.startswith("Context:\nEXTRA\nSummarize")
