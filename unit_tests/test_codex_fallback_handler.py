import json
import types

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

    cf._settings = types.SimpleNamespace(codex_fallback_model="alt-model")

    def fake_reroute(p: Prompt) -> LLMResult:
        called["prompt"] = p.user
        return LLMResult(text="ok", raw={"model": cf._settings.codex_fallback_model})

    monkeypatch.setattr(cf, "reroute_to_fallback_model", fake_reroute)

    result = cf.handle(prompt, "oops")
    assert called["prompt"] == "hi"
    assert isinstance(result, LLMResult)
    assert result.text == "ok"
    assert result.raw["model"] == "alt-model"


def test_handle_queues_on_failure(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.jsonl"
    monkeypatch.setattr(cf, "_QUEUE_FILE", queue_path)

    def boom(_: Prompt) -> LLMResult:
        raise RuntimeError("fail")

    monkeypatch.setattr(cf, "reroute_to_fallback_model", boom)

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
        return LLMResult(text="", raw={"model": cf._settings.codex_fallback_model})

    monkeypatch.setattr(cf, "reroute_to_fallback_model", empty)

    result = cf.handle(Prompt("zap"), "no text")
    assert isinstance(result, LLMResult)
    assert result.text == ""
    assert result.raw["reason"] == "no text"

    record = json.loads(queue_path.read_text().strip())
    assert record["prompt"] == "zap"
    assert record["reason"] == "no text"


def test_reroute_uses_configured_model(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, *, model: str):
            captured["model"] = model
            self.model = model

        def generate(self, prompt: Prompt) -> LLMResult:
            captured["prompt"] = prompt.user
            return LLMResult(text="ok", raw={"model": self.model})

    monkeypatch.setattr(cf, "LLMClient", DummyClient)
    cf._settings = types.SimpleNamespace(codex_fallback_model="dummy-model")

    result = cf.reroute_to_fallback_model(Prompt("hi"))
    assert captured["model"] == "dummy-model"
    assert captured["prompt"] == "hi"
    assert result.raw["model"] == "dummy-model"
