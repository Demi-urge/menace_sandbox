import sqlite3

import retry_utils
from llm_interface import Prompt, LLMResult, LLMClient
from llm_router import LLMRouter
from prompt_db import PromptDB


class MemoryRouter:
    """Minimal router returning an in-memory SQLite connection."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")

    def get_connection(self, table_name: str, operation: str = "write"):
        return self.conn


def test_promptdb_logs_to_memory():
    db = PromptDB(model="test", router=MemoryRouter())
    prompt = Prompt(text="hi", examples=["ex"])
    result = LLMResult(raw={"r": 1}, text="res")
    db.log_prompt(prompt, result, ["tag"], 0.4)
    row = db.conn.execute(
        "SELECT text, examples, confidence, tags, response_text, model FROM prompts"
    ).fetchone()
    assert row == (
        "hi",
        "[\"ex\"]",
        0.4,
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
    def generate(self, prompt: Prompt) -> LLMResult:
        raise RuntimeError("boom")


class EchoClient(LLMClient):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: Prompt) -> LLMResult:
        self.calls += 1
        return LLMResult(text="local")


def test_router_fallback_on_error():
    router = LLMRouter(remote=FailingClient(), local=EchoClient(), size_threshold=1)
    res = router.generate(Prompt(text="hi"))
    assert res.text == "local"
