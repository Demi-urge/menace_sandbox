import json
import types

import menace.codex_fallback_handler as cf
from menace.prompt_types import Prompt
from menace.llm_interface import LLMResult


class DummyBuilder:
    def refresh_db_weights(self, *a, **k):
        return None

    def build(self, *a, **k):
        return ""


def test_queue_failed_writes_jsonl(tmp_path):
    path = tmp_path / "queue.jsonl"
    cf._settings = types.SimpleNamespace(codex_retry_queue_path=str(path))
    cf.queue_failed(Prompt("hello"), "boom")

    data = json.loads(path.read_text().strip())
    assert data["prompt"] == "hello"
    assert data["reason"] == "boom"


def test_handle_reroutes(monkeypatch, tmp_path):
    prompt = Prompt("hi")

    called = {}

    cf._settings = types.SimpleNamespace(
        codex_fallback_model="alt-model",
        codex_retry_queue_path=str(tmp_path / "queue.jsonl"),
        codex_fallback_strategy="reroute",
    )

    def fake_reroute(p: Prompt, *, context_builder=None) -> LLMResult:
        called["prompt"] = p.user
        return LLMResult(text="ok", raw={"model": cf._settings.codex_fallback_model})

    monkeypatch.setattr(cf, "reroute_to_fallback_model", fake_reroute)

    result = cf.handle(prompt, "oops", context_builder=DummyBuilder())
    assert called["prompt"] == "hi"
    assert isinstance(result, LLMResult)
    assert result.text == "ok"
    assert result.raw["model"] == "alt-model"


def test_handle_queues_on_failure(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.jsonl"
    cf._settings = types.SimpleNamespace(
        codex_retry_queue_path=str(queue_path), codex_fallback_strategy="reroute"
    )

    def boom(_: Prompt, *, context_builder=None) -> LLMResult:
        raise RuntimeError("fail")

    monkeypatch.setattr(cf, "reroute_to_fallback_model", boom)

    result = cf.handle(Prompt("bye"), "bad news", context_builder=DummyBuilder())
    assert isinstance(result, LLMResult)
    assert result.text == ""
    assert result.raw["reason"] == "bad news"

    record = json.loads(queue_path.read_text().strip())
    assert record["prompt"] == "bye"
    assert record["reason"] == "bad news"


def test_handle_returns_reason_on_empty_completion(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.jsonl"
    cf._settings = types.SimpleNamespace(
        codex_retry_queue_path=str(queue_path),
        codex_fallback_model="m",
        codex_fallback_strategy="reroute",
    )

    def empty(_: Prompt, *, context_builder=None) -> LLMResult:
        return LLMResult(text="", raw={"model": cf._settings.codex_fallback_model})

    monkeypatch.setattr(cf, "reroute_to_fallback_model", empty)

    result = cf.handle(Prompt("zap"), "no text", context_builder=DummyBuilder())
    assert isinstance(result, LLMResult)
    assert result.text == ""
    assert result.raw["reason"] == "no text"

    record = json.loads(queue_path.read_text().strip())
    assert record["prompt"] == "zap"
    assert record["reason"] == "no text"


def test_handle_queue_strategy(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.jsonl"
    called = {}
    cf._settings = types.SimpleNamespace(
        codex_retry_queue_path=str(queue_path), codex_fallback_strategy="queue"
    )

    def fake_reroute(p: Prompt, *, context_builder=None) -> LLMResult:
        called["prompt"] = p.user
        return LLMResult(text="ok")

    monkeypatch.setattr(cf, "reroute_to_fallback_model", fake_reroute)

    result = cf.handle(Prompt("hi"), "boom", context_builder=DummyBuilder())
    assert "prompt" not in called
    assert result.text == ""
    record = json.loads(queue_path.read_text().strip())
    assert record["prompt"] == "hi"
    assert record["reason"] == "boom"


def test_reroute_uses_configured_model(monkeypatch, tmp_path):
    captured = {}

    class DummyClient:
        def __init__(self, *, model: str):
            captured["model"] = model
            self.model = model

        def generate(self, prompt: Prompt) -> LLMResult:
            captured["prompt"] = prompt.user
            return LLMResult(text="ok", raw={"model": self.model})

    monkeypatch.setattr(cf, "LLMClient", DummyClient)
    cf._settings = types.SimpleNamespace(
        codex_fallback_model="dummy-model", codex_retry_queue_path=str(tmp_path / "queue.jsonl")
    )

    result = cf.reroute_to_fallback_model(Prompt("hi"), context_builder=DummyBuilder())
    assert captured["model"] == "dummy-model"
    assert captured["prompt"] == "hi"
    assert result.raw["model"] == "dummy-model"
