from types import SimpleNamespace
import memory_aware_gpt_client as magc


class DummyKnowledge:
    def __init__(self):
        self.logged = []

    def build_context(self, key: str, limit: int = 5) -> str:
        return (
            "### Feedback\n- fb1\n\n### Error fixes\n- fix1\n\n### Improvement paths\n- imp1"
        )

    def log(self, prompt: str, resp: str, tags):
        self.logged.append((prompt, resp, tags))


def test_context_injection_and_logging():
    client = SimpleNamespace()
    recorded = {}

    def fake_ask(msgs, **kw):
        recorded["messages"] = msgs
        return {"choices": [{"message": {"content": "response"}}]}

    client.ask = fake_ask
    knowledge = DummyKnowledge()
    builder = SimpleNamespace(build=lambda *a, **k: "")

    magc.ask_with_memory(
        client,
        "mod.act",
        "Do it",
        memory=knowledge,
        context_builder=builder,
        tags=["feedback"],
    )

    sent_prompt = recorded["messages"][0]["content"]
    assert "fb1" in sent_prompt
    assert "fix1" in sent_prompt
    assert "imp1" in sent_prompt
    assert knowledge.logged and knowledge.logged[0][0].endswith("Do it")
