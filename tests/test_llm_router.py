from llm_interface import Prompt, LLMResult, LLMClient
from llm_router import LLMRouter


class StubClient(LLMClient):
    def __init__(self, text: str, *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    def generate(self, prompt: Prompt) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return LLMResult(raw={}, text=self.text)


def test_router_uses_local_for_small_prompts():
    local = StubClient("local")
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(Prompt(text="hi"))
    assert res.text == "local"
    assert local.calls == 1
    assert remote.calls == 0

def test_router_uses_remote_for_large_prompts():
    local = StubClient("local")
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(Prompt(text="this is long"))
    assert res.text == "remote"
    assert remote.calls == 1

def test_router_fallback_on_failure():
    local = StubClient("local", fail=True)
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(Prompt(text="hi"))

    assert res.text == "remote"
    assert remote.calls == 1
