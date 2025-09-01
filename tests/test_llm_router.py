from llm_interface import Prompt, LLMResult, LLMClient
from llm_router import LLMRouter, client_from_settings
from sandbox_settings import SandboxSettings
import llm_router


class StubClient(LLMClient):
    def __init__(self, text: str, *, fail: bool = False) -> None:
        super().__init__(text, log_prompts=False)
        self.text = text
        self.fail = fail
        self.calls = 0

    def _generate(self, prompt: Prompt) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return LLMResult(raw={"backend": self.model}, text=self.text)


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
    res = router.generate(Prompt(text="this is a long prompt for testing"))
    assert res.text == "remote"
    assert remote.calls == 1


def test_router_fallback_on_failure():
    local = StubClient("local", fail=True)
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(Prompt(text="hi"))

    assert res.text == "remote"
    assert remote.calls == 1


def test_router_respects_roi_tags():
    local = StubClient("local")
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(Prompt(text="hi", tags=["high_roi"]))
    assert res.text == "remote"
    res = router.generate(Prompt(text="this is a long prompt for testing", tags=["low_roi"]))
    assert res.text == "local"


def test_router_avoids_recent_failures(monkeypatch):
    local = StubClient("local")
    remote = StubClient("remote", fail=True)
    router = LLMRouter(remote=remote, local=local, size_threshold=5, failure_cooldown=10)

    now = [0.0]
    monkeypatch.setattr(llm_router.time, "time", lambda: now[0])

    router.generate(Prompt(text="this is a long prompt for testing"))
    assert remote.calls == 1

    remote.fail = False
    now[0] = 1.0
    router.generate(Prompt(text="this is a long prompt for testing"))
    assert remote.calls == 1
    assert local.calls == 2


def test_router_logs_backend_choice(monkeypatch):
    logged: list[str] = []

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, result, backend=None):
            logged.append(backend)

    import llm_router as lr

    monkeypatch.setattr(lr, "PromptDB", DummyDB)
    router = LLMRouter(remote=StubClient("remote"), local=StubClient("local"), size_threshold=5)
    res = router.generate(Prompt(text="hi"))
    assert logged == ["local"]
    assert res.raw["backend"] == "local"


def custom_factory() -> StubClient:
    return StubClient("custom")


def test_client_from_settings_dynamic_backend(monkeypatch):
    settings = SandboxSettings(
        preferred_llm_backend="custom",
        available_backends={"custom": "tests.test_llm_router.custom_factory"},
    )
    client = client_from_settings(settings)
    res = client.generate(Prompt(text="hi"))
    assert res.text == "custom"
