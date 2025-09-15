from typing import Any
from llm_interface import Prompt, LLMResult, LLMClient
from llm_router import LLMRouter, client_from_settings
from sandbox_settings import SandboxSettings
import llm_router
from llm_registry import backend, create_backend


class StubClient(LLMClient):
    def __init__(self, text: str, *, fail: bool = False) -> None:
        super().__init__(text, log_prompts=False)
        self.text = text
        self.fail = fail
        self.calls = 0

    def _generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return LLMResult(raw={"backend": self.model}, text=self.text)


def test_router_uses_local_for_small_prompts():
    local = StubClient("local")
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder")
    )
    assert res.text == "local"
    assert local.calls == 1
    assert remote.calls == 0


def test_router_uses_remote_for_large_prompts():
    local = StubClient("local")
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(
        Prompt(
            text="this is a long prompt for testing",
            vector_confidence=0.5,
            origin="context_builder",
        )
    )
    assert res.text == "remote"
    assert remote.calls == 1


def test_router_fallback_on_failure():
    local = StubClient("local", fail=True)
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder")
    )

    assert res.text == "remote"
    assert remote.calls == 1


def test_router_respects_roi_tags():
    local = StubClient("local")
    remote = StubClient("remote")
    router = LLMRouter(remote=remote, local=local, size_threshold=5)
    res = router.generate(
        Prompt(
            text="hi",
            tags=["high_roi"],
            vector_confidence=0.5,
            origin="context_builder",
        )
    )
    assert res.text == "remote"
    res = router.generate(
        Prompt(
            text="this is a long prompt for testing",
            tags=["low_roi"],
            vector_confidence=0.5,
            origin="context_builder",
        )
    )
    assert res.text == "local"


def test_router_avoids_recent_failures(monkeypatch):
    local = StubClient("local")
    remote = StubClient("remote", fail=True)
    router = LLMRouter(remote=remote, local=local, size_threshold=5, failure_cooldown=10)

    now = [0.0]
    monkeypatch.setattr(llm_router.time, "time", lambda: now[0])

    router.generate(
        Prompt(
            text="this is a long prompt for testing",
            vector_confidence=0.5,
            origin="context_builder",
        )
    )
    assert remote.calls == 1

    remote.fail = False
    now[0] = 1.0
    router.generate(
        Prompt(
            text="this is a long prompt for testing",
            vector_confidence=0.5,
            origin="context_builder",
        )
    )
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
    res = router.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder")
    )
    assert logged == ["local"]
    assert res.raw["backend"] == "local"


def test_router_logs_prompt_metadata(monkeypatch):
    logged: dict[str, Any] = {}

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, result, backend=None):
            logged["prompt"] = prompt
            logged["raw"] = result.raw
            logged["backend"] = backend

    import llm_router as lr

    monkeypatch.setattr(lr, "PromptDB", DummyDB)
    router = LLMRouter(remote=StubClient("remote"), local=StubClient("local"), size_threshold=5)
    prompt = Prompt(
        text="hi",
        examples=["ex"],
        outcome_tags=["tag"],
        vector_confidence=0.1,
        origin="context_builder",
    )
    router.generate(prompt)

    assert logged["backend"] == "local"
    assert logged["prompt"].examples == ["ex"]
    assert logged["prompt"].vector_confidence == 0.1
    assert logged["prompt"].outcome_tags == ["tag"]
    assert logged["raw"]["tags"] == ["tag"]
    assert logged["raw"]["vector_confidence"] == 0.1


def custom_factory() -> StubClient:
    return StubClient("custom")


@backend("decorator")
def decorator_factory() -> StubClient:
    return StubClient("decorator")


def test_client_from_settings_dynamic_backend(monkeypatch):
    settings = SandboxSettings(
        preferred_llm_backend="custom",
        available_backends={"custom": "tests.test_llm_router.custom_factory"},
    )
    client = client_from_settings(settings)
    res = client.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder")
    )
    assert res.text == "custom"


def test_registry_decorator_helper():
    client = create_backend("decorator")
    res = client.generate(
        Prompt(text="hi", vector_confidence=0.5, origin="context_builder")
    )
    assert res.text == "decorator"
