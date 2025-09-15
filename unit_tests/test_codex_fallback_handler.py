import json
import types
import sys

sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))

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
    captured = {}

    class DummyClient:
        def __init__(self, *, model: str) -> None:
            self.model = model

        def generate(self, prompt: Prompt, *, context_builder=None) -> LLMResult:
            captured["prompt"] = prompt
            captured["builder"] = context_builder
            return LLMResult(text="ok", raw={"model": self.model})

    class Builder:
        def refresh_db_weights(self, *a, **k):
            return None

        def build_prompt(self, query: str, **kwargs) -> Prompt:
            return Prompt(
                f"ctx\n\n{query}",
                system=kwargs.get("system", ""),
                examples=list(kwargs.get("examples", []) or []),
                tags=list(kwargs.get("tags", []) or []),
            )

    builder = Builder()

    monkeypatch.setattr(cf, "LLMClient", DummyClient)

    cf._settings = types.SimpleNamespace(
        codex_fallback_model="alt-model",
        codex_retry_queue_path=str(tmp_path / "queue.jsonl"),
        codex_fallback_strategy="reroute",
    )

    p = Prompt("hi", system="sys", examples=["e1"])
    p.tags = ["t1"]

    result = cf.handle(p, "oops", context_builder=builder)
    assert captured["prompt"].user == "ctx\n\nhi"
    assert captured["prompt"].system == "sys"
    assert captured["prompt"].examples == ["e1"]
    assert captured["prompt"].tags == ["t1"]
    assert captured["builder"] is builder
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

        def generate(self, prompt: Prompt, *, context_builder=None) -> LLMResult:
            captured["prompt"] = prompt.user
            captured["builder"] = context_builder
            return LLMResult(text="ok", raw={"model": self.model})

    monkeypatch.setattr(cf, "LLMClient", DummyClient)
    cf._settings = types.SimpleNamespace(
        codex_fallback_model="dummy-model", codex_retry_queue_path=str(tmp_path / "queue.jsonl")
    )

    builder = DummyBuilder()
    result = cf.reroute_to_fallback_model(Prompt("hi"), context_builder=builder)
    assert captured["model"] == "dummy-model"
    assert captured["prompt"] == "hi"
    assert captured["builder"] is builder
    assert result.raw["model"] == "dummy-model"


def test_reroute_includes_retrieved_context(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, *, model: str):
            self.model = model

        def generate(self, prompt: Prompt, *, context_builder=None) -> LLMResult:
            captured["prompt"] = prompt
            captured["builder"] = context_builder
            captured["system"] = prompt.system
            captured["examples"] = prompt.examples
            captured["tags"] = getattr(prompt, "tags", [])
            return LLMResult(text="ok")

    class Builder:
        def refresh_db_weights(self, *a, **k):
            return None

        def build_prompt(self, query: str, **kwargs) -> Prompt:
            captured["args"] = (
                kwargs.get("system"),
                kwargs.get("examples"),
                kwargs.get("tags"),
            )
            return Prompt(
                f"ctx\n\n{query}",
                system=kwargs.get("system", ""),
                examples=list(kwargs.get("examples", []) or []),
                tags=list(kwargs.get("tags", []) or []),
            )

    builder = Builder()
    monkeypatch.setattr(cf, "LLMClient", DummyClient)
    cf._settings = types.SimpleNamespace(codex_fallback_model="dummy")

    p = Prompt("hi", system="sys", examples=["e1"])
    p.tags = ["t1"]
    cf.reroute_to_fallback_model(p, context_builder=builder)
    assert captured["prompt"].user == "ctx\n\nhi"
    assert captured["builder"] is builder
    assert captured["args"] == ("sys", ["e1"], ["t1"])
    assert captured["system"] == "sys"
    assert captured["examples"] == ["e1"]
    assert captured["tags"] == ["t1"]
