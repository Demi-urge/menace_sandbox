import retry_utils
from db_router import DBRouter
from llm_interface import Prompt, LLMResult, LLMClient
import random
from llm_router import LLMRouter
from prompt_db import PromptDB


def test_promptdb_logs_to_memory(tmp_path):
    router = DBRouter("prompts", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = PromptDB(model="test", router=router)
    prompt = Prompt(
        text="hi", examples=["ex"], outcome_tags=["tag"], vector_confidences=[0.4]
    )
    result = LLMResult(raw={"r": 1}, text="res")
    db.log(prompt, result)
    row = db.conn.execute(
        "SELECT text, examples, vector_confidences, outcome_tags, response_text, model FROM prompts"
    ).fetchone()
    assert row == (
        "hi",
        "[\"ex\"]",
        "[0.4]",
        "[\"tag\"]",
        "res",
        "test",
    )


def test_retry_with_backoff(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def func():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("fail")
        return "ok"

    monkeypatch.setattr(retry_utils.time, "sleep", lambda s: sleeps.append(s))
    result = retry_utils.with_retry(func, attempts=3, delay=1.0, exc=ValueError)
    assert result == "ok"
    assert sleeps == [1.0, 2.0]
    assert calls["count"] == 3


class FailingClient(LLMClient):
    def __init__(self):
        super().__init__("fail", log_prompts=False)

    def _generate(self, prompt: Prompt) -> LLMResult:
        raise RuntimeError("boom")


class EchoClient(LLMClient):
    def __init__(self):
        super().__init__("echo", log_prompts=False)
        self.calls = 0

    def _generate(self, prompt: Prompt) -> LLMResult:
        self.calls += 1
        return LLMResult(text="local")


def test_router_fallback_on_error():
    router = LLMRouter(remote=FailingClient(), local=EchoClient(), size_threshold=1)
    res = router.generate(Prompt(text="hi"))
    assert res.text == "local"


def test_openai_provider_retry_and_logging(monkeypatch):
    """OpenAIProvider retries on 429 responses and logs the interaction."""

    from llm_interface import OpenAIProvider, requests, time as llm_time, Prompt, LLMResult

    # Capture sleep calls to assert backoff behaviour
    sleeps: list[float] = []
    monkeypatch.setattr(llm_time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    # Stub PromptDB to record log invocations
    logged: list[tuple[Prompt, LLMResult]] = []

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, result):
            logged.append((prompt, result))

    import prompt_db

    monkeypatch.setattr(prompt_db, "PromptDB", DummyDB)

    # Fake HTTP responses: first a rate limit, then success
    class Resp:
        def __init__(self, status, data):
            self.status_code = status
            self.ok = status == 200
            self._data = data

        def json(self):  # pragma: no cover - simple stub
            return self._data

        def raise_for_status(self):  # pragma: no cover - simple stub
            if not self.ok:
                raise requests.HTTPError("boom", response=self)

    responses = [Resp(429, {}), Resp(200, {"choices": [{"message": {"content": "{\"a\":1}"}}]})]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(max_retries=2)
    monkeypatch.setattr(provider._session, "post", fake_post)

    prompt = Prompt(text="hi", metadata={"tags": ["t"], "vector_confidences": [0.7]})
    result = provider.generate(prompt)

    # Second response consumed and logged
    assert not responses
    assert sleeps == [1.0]
    assert result.parsed == {"a": 1}
    assert logged and logged[0][0] is prompt


def test_router_fallback_logs(monkeypatch):
    """LLMRouter falls back to local backend and logs the successful prompt."""

    logged: list[str] = []

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, *_a, **_k):
            logged.append(prompt.text)

    import prompt_db

    monkeypatch.setattr(prompt_db, "PromptDB", DummyDB)

    class BoomClient(LLMClient):
        def __init__(self):
            super().__init__("boom")

        def _generate(self, prompt: Prompt) -> LLMResult:
            raise RuntimeError("fail")

    class LocalClient(LLMClient):
        def __init__(self):
            super().__init__("local")

        def _generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="ok")

    router = LLMRouter(remote=BoomClient(), local=LocalClient(), size_threshold=1)
    res = router.generate(Prompt(text="task", metadata={"tags": ["x"]}))
    assert res.text == "ok"


def test_client_backends_fallback():
    """LLMClient tries secondary backends when the first fails."""

    class BoomBackend:
        model = "boom"

        def generate(self, prompt: Prompt) -> LLMResult:
            raise RuntimeError("fail")

    class LocalBackend:
        model = "local"

        def generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="local")

    client = LLMClient(backends=[BoomBackend(), LocalBackend()], log_prompts=False)
    res = client.generate(Prompt(text="hi"))
    assert res.text == "local"


def test_client_small_task_uses_local():
    """Prompts flagged as small_task bypass the first backend."""

    class RemoteBackend:
        model = "remote"

        def generate(self, prompt: Prompt) -> LLMResult:  # pragma: no cover - should not run
            raise AssertionError("should not be called")

    class LocalBackend:
        model = "local"

        def generate(self, prompt: Prompt) -> LLMResult:
            return LLMResult(text="ok")

    client = LLMClient(backends=[RemoteBackend(), LocalBackend()], log_prompts=False)
    res = client.generate(Prompt(text="task", metadata={"small_task": True}))
    assert res.text == "ok"
