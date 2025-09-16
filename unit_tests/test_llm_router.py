import sys, pathlib, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_interface import LLMClient, LLMResult, Prompt
from llm_router import LLMRouter


def _builder():
    return types.SimpleNamespace(roi_tracker=None)


META = {"vector_confidences": [0.5]}


class DummyBackend(LLMClient):
    def __init__(self, model: str, *, fail: bool = False):
        super().__init__(model, log_prompts=False)
        self.fail = fail
        self.calls = 0

    def _generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("fail")
        return LLMResult(text=self.model, raw={"model": self.model, "backend": self.model})


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    logged = []

    class DummyDB:
        def __init__(self, model):
            self.logged = logged
            self.latencies = {}

        def log(self, prompt, result, backend=None):
            self.logged.append((prompt, result, backend))

        def fetch_logs(self, limit=None, model=None, tag=None):
            if model is None:
                return []
            lat = self.latencies.get(model)
            if lat is None:
                return []
            return [{"backend": model, "latency_ms": lat}]

    import llm_router

    monkeypatch.setattr(llm_router, "PromptDB", DummyDB)
    return logged


def test_routing_by_prompt_size():
    remote = DummyBackend("remote")
    local = DummyBackend("local")
    router = LLMRouter(remote=remote, local=local, size_threshold=10)

    small = Prompt("hi", origin="context_builder", metadata=META)
    large = Prompt("x" * 100, origin="context_builder", metadata=META)
    builder = _builder()

    res_small = router.generate(small, context_builder=builder)
    assert res_small.text == "local"
    assert local.calls == 1 and remote.calls == 0

    res_large = router.generate(large, context_builder=builder)
    assert res_large.text == "remote"
    assert remote.calls == 1


def test_routing_by_roi_tags():
    remote = DummyBackend("remote")
    local = DummyBackend("local")
    router = LLMRouter(remote=remote, local=local, size_threshold=10)
    builder = _builder()

    prompt_low = Prompt(
        "x" * 100, tags=["low_roi"], origin="context_builder", metadata=META
    )
    result_low = router.generate(prompt_low, context_builder=builder)
    assert result_low.text == "local"
    assert local.calls == 1 and remote.calls == 0

    remote.calls = local.calls = 0
    prompt_high = Prompt(
        "hi", tags=["high_roi"], origin="context_builder", metadata=META
    )
    result_high = router.generate(prompt_high, context_builder=builder)
    assert result_high.text == "remote"
    assert remote.calls == 1 and local.calls == 0

    remote.calls = local.calls = 0
    prompt_meta = Prompt(
        "hi",
        metadata={"tags": ["high_roi"], "vector_confidences": META["vector_confidences"]},
        origin="context_builder",
    )
    result_meta = router.generate(prompt_meta, context_builder=builder)
    assert result_meta.text == "remote"
    assert remote.calls == 1 and local.calls == 0


def test_fallback_on_error():
    remote = DummyBackend("remote", fail=True)
    local = DummyBackend("local")
    router = LLMRouter(remote=remote, local=local, size_threshold=0)
    builder = _builder()

    result = router.generate(
        Prompt("hi", origin="context_builder", metadata=META), context_builder=builder
    )
    assert result.text == "local"
    assert remote.calls == 1 and local.calls == 1


def test_token_cost_preference(monkeypatch):
    remote = DummyBackend("gpt-4o")
    local = DummyBackend("claude-3-sonnet")
    router = LLMRouter(remote=remote, local=local, size_threshold=10)

    prompt = Prompt("x" * 100, origin="context_builder", metadata=META)
    res = router.generate(prompt, context_builder=_builder())
    assert res.text == "claude-3-sonnet"


def test_latency_preference(monkeypatch):
    remote = DummyBackend("gpt-4o")
    local = DummyBackend("claude-3-sonnet")
    router = LLMRouter(remote=remote, local=local, size_threshold=10)
    # Equal pricing so latency drives decision
    monkeypatch.setattr("llm_pricing.get_input_rate", lambda model, overrides=None: 0.0)
    router.db.latencies = {"gpt-4o": 200.0, "claude-3-sonnet": 50.0}

    prompt = Prompt("x" * 100, origin="context_builder", metadata=META)
    res = router.generate(prompt, context_builder=_builder())
    assert res.text == "claude-3-sonnet"


def test_vector_confidence_bias(monkeypatch):
    remote = DummyBackend("gpt-4o")
    local = DummyBackend("claude-3-sonnet")
    router = LLMRouter(remote=remote, local=local, size_threshold=10)
    monkeypatch.setattr("llm_pricing.get_input_rate", lambda model, overrides=None: 0.0)

    high = Prompt(
        "hi", vector_confidence=0.9, origin="context_builder", metadata=META
    )
    res_high = router.generate(high, context_builder=_builder())
    assert res_high.text == "gpt-4o"

    low = Prompt(
        "x" * 100, vector_confidence=0.1, origin="context_builder", metadata=META
    )
    res_low = router.generate(low, context_builder=_builder())
    assert res_low.text == "claude-3-sonnet"
