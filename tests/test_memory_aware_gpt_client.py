from types import SimpleNamespace
import memory_aware_gpt_client as magc


def test_context_injection_and_logging(monkeypatch):
    client = SimpleNamespace()
    recorded = {}

    def fake_ask(msgs, **kw):
        recorded['messages'] = msgs
        return {"choices": [{"message": {"content": "response"}}]}

    client.ask = fake_ask

    fb = [SimpleNamespace(prompt="p1", response="fb1")]
    fixes = [SimpleNamespace(prompt="p2", response="fix1")]
    improvs = [SimpleNamespace(prompt="p3", response="imp1")]

    monkeypatch.setattr(magc, "get_feedback", lambda m, k, limit=5: fb)
    monkeypatch.setattr(magc, "get_error_fixes", lambda m, k, limit=3: fixes)
    monkeypatch.setattr(magc, "get_improvement_paths", lambda m, k, limit=3: improvs)

    log_calls = []

    def fake_log(mem, prompt, resp, tags):
        log_calls.append((prompt, resp, tags))

    monkeypatch.setattr(magc, "log_with_tags", fake_log)

    magc.ask_with_memory(client, "mod.act", "Do it", memory=object(), tags=["feedback"])

    sent_prompt = recorded['messages'][0]['content']
    assert "fb1" in sent_prompt
    assert "fix1" in sent_prompt
    assert "imp1" in sent_prompt
    assert log_calls and log_calls[0][0].endswith("Do it")
