import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_interface import LLMClient, LLMResult, Prompt
from llm_router import LLMRouter


class DummyBackend(LLMClient):
    def __init__(self, model: str, *, fail: bool = False):
        super().__init__(model, log_prompts=False)
        self.fail = fail
        self.calls = 0

    def _generate(self, prompt: Prompt) -> LLMResult:
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

        def log(self, prompt, result, backend=None):
            self.logged.append((prompt, result, backend))

    import llm_router

    monkeypatch.setattr(llm_router, "PromptDB", DummyDB)
    return logged


def test_routing_by_prompt_size():
    remote = DummyBackend("remote")
    local = DummyBackend("local")
    router = LLMRouter(remote=remote, local=local, size_threshold=10)

    small = Prompt("hi")
    large = Prompt("x" * 100)

    res_small = router.generate(small)
    assert res_small.text == "local"
    assert local.calls == 1 and remote.calls == 0

    res_large = router.generate(large)
    assert res_large.text == "remote"
    assert remote.calls == 1


def test_routing_by_roi_tags():
    remote = DummyBackend("remote")
    local = DummyBackend("local")
    router = LLMRouter(remote=remote, local=local, size_threshold=10)

    prompt_low = Prompt("x" * 100, tags=["low_roi"])
    result_low = router.generate(prompt_low)
    assert result_low.text == "local"
    assert local.calls == 1 and remote.calls == 0

    remote.calls = local.calls = 0
    prompt_high = Prompt("hi", tags=["high_roi"])
    result_high = router.generate(prompt_high)
    assert result_high.text == "remote"
    assert remote.calls == 1 and local.calls == 0

    remote.calls = local.calls = 0
    prompt_meta = Prompt("hi", metadata={"tags": ["high_roi"]})
    result_meta = router.generate(prompt_meta)
    assert result_meta.text == "remote"
    assert remote.calls == 1 and local.calls == 0


def test_fallback_on_error():
    remote = DummyBackend("remote", fail=True)
    local = DummyBackend("local")
    router = LLMRouter(remote=remote, local=local, size_threshold=0)

    result = router.generate(Prompt("hi"))
    assert result.text == "local"
    assert remote.calls == 1 and local.calls == 1
