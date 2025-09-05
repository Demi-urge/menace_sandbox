import json

import menace.codex_fallback_handler as cf
from menace.prompt_types import Prompt
from menace.llm_interface import LLMResult


def test_queue_failed_writes_jsonl(tmp_path):
    path = tmp_path / "queue.jsonl"
    cf.queue_failed(Prompt("hello"), "boom", path=path)

    data = json.loads(path.read_text().strip())
    assert data["prompt"] == "hello"
    assert data["reason"] == "boom"


def test_handle_reroutes(monkeypatch):
    prompt = Prompt("hi")

    called = {}

    def fake_reroute(p: Prompt) -> LLMResult:
        called["prompt"] = p.user
        return LLMResult(text="ok", raw={"model": "gpt-3.5-turbo"})

    monkeypatch.setattr(cf, "reroute_to_gpt35", fake_reroute)

    result = cf.handle(prompt, "oops")
    assert called["prompt"] == "hi"
    assert isinstance(result, LLMResult)
    assert result.text == "ok"
    assert result.raw["model"] == "gpt-3.5-turbo"


def test_handle_queues_on_failure(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.jsonl"
    monkeypatch.setattr(cf, "_QUEUE_FILE", queue_path)

    def boom(_: Prompt) -> LLMResult:
        raise RuntimeError("fail")

    monkeypatch.setattr(cf, "reroute_to_gpt35", boom)

    result = cf.handle(Prompt("bye"), "bad news")
    assert isinstance(result, LLMResult)
    assert result.text == ""
    assert result.raw["reason"] == "bad news"

    record = json.loads(queue_path.read_text().strip())
    assert record["prompt"] == "bye"
    assert record["reason"] == "bad news"


def test_handle_returns_reason_on_empty_completion(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.jsonl"
    monkeypatch.setattr(cf, "_QUEUE_FILE", queue_path)

    def empty(_: Prompt) -> LLMResult:
        return LLMResult(text="", raw={"model": "gpt-3.5-turbo"})

    monkeypatch.setattr(cf, "reroute_to_gpt35", empty)

    result = cf.handle(Prompt("zap"), "no text")
    assert isinstance(result, LLMResult)
    assert result.text == ""
    assert result.raw["reason"] == "no text"

    record = json.loads(queue_path.read_text().strip())
    assert record["prompt"] == "zap"
    assert record["reason"] == "no text"

