import json

import action_justifier as aj


class DummyTokenizer:
    last_prompt = None

    @classmethod
    def from_pretrained(cls, model_path, local_files_only=True):
        return cls()

    def encode(self, prompt, return_tensors=None):
        DummyTokenizer.last_prompt = prompt
        return [prompt]

    def decode(self, tokens, skip_special_tokens=True):
        return tokens


class DummyModel:
    def __init__(self):
        self.calls = 0

    @classmethod
    def from_pretrained(cls, model_path, local_files_only=True):
        return cls()

    def generate(self, tokens, max_new_tokens=60, do_sample=False):
        self.calls += 1
        return [tokens[0] + " explanation"]


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(aj, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(aj, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(aj, "AutoModelForCausalLM", DummyModel)
    monkeypatch.setattr(aj, "compress_snippets", lambda meta, **_: meta)
    monkeypatch.setattr(aj, "_CACHE_DIR", tmp_path)


class Builder:
    def __init__(self):
        self.queries = []

    def build(self, query, **_):
        self.queries.append(query)
        return "builder-context"


def test_llm_justification_includes_vector_context(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    builder = Builder()
    settings = {"model_path": "dummy"}
    action_log = {"action_type": "x", "action_description": "y"}
    payload = {
        "action_log": action_log,
        "violation_flags": [],
        "risk_score": 0.1,
        "domain": "dom",
    }

    res = aj._llm_justification(action_log, [], 0.1, "dom", settings, context_builder=builder)
    assert res == "explanation"
    assert builder.queries == [json.dumps(payload)]
    assert DummyTokenizer.last_prompt.startswith("builder-context\n\nAction type:")


def test_llm_justification_cache_hit(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    builder = Builder()
    settings = {"model_path": "dummy"}
    action_log = {"action_type": "x"}

    model = DummyModel()
    monkeypatch.setattr(
        aj.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: model),
    )

    res1 = aj._llm_justification(action_log, [], 0.2, "dom", settings, context_builder=builder)
    assert res1 == "explanation"
    res2 = aj._llm_justification(action_log, [], 0.2, "dom", settings, context_builder=builder)
    assert res2 == "explanation"
    assert model.calls == 1
