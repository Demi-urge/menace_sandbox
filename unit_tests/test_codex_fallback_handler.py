import json

import menace.codex_fallback_handler as cf
from menace.prompt_types import Prompt


def test_queue_failed_writes_jsonl(tmp_path):
    path = tmp_path / "queue.jsonl"
    cf.queue_failed(Prompt("hello"), "boom", path=path)

    data = json.loads(path.read_text().strip())
    assert data["prompt"] == "hello"
    assert data["reason"] == "boom"


def test_handle_reroutes(monkeypatch):
    prompt = Prompt("hi")

    called = {}

    def fake_reroute(p: Prompt) -> str:
        called["prompt"] = p.user
        return "ok"

    monkeypatch.setattr(cf, "reroute_to_gpt35", fake_reroute)

    result = cf.handle(prompt, "oops")
    assert called["prompt"] == "hi"
    assert result == "ok"


def test_handle_queues_on_failure(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.jsonl"
    monkeypatch.setattr(cf, "_QUEUE_FILE", queue_path)

    def boom(_: Prompt) -> str:
        raise RuntimeError("fail")

    monkeypatch.setattr(cf, "reroute_to_gpt35", boom)

    result = cf.handle(Prompt("bye"), "bad news")
    assert result == ""

    record = json.loads(queue_path.read_text().strip())
    assert record["prompt"] == "bye"
    assert record["reason"] == "bad news"

