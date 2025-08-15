import menace.chatgpt_idea_bot as cib
from log_tags import IMPROVEMENT_PATH


class DummyMemory:
    def __init__(self) -> None:
        self.entries = []
        self.fetch_calls = []

    def log_interaction(self, prompt: str, response: str, tags):
        self.entries.append((prompt, response, list(tags)))

    def fetch_context(self, tags):
        self.fetch_calls.append(list(tags))
        for prompt, response, stored_tags in reversed(self.entries):
            if set(tags) & set(stored_tags):
                return response
        return ""


def test_memory_based_context(monkeypatch):
    mem = DummyMemory()
    mem.log_interaction("Initial feedback", "Improve caching strategy", tags=[IMPROVEMENT_PATH])

    client = cib.ChatGPTClient(gpt_memory=mem)
    client.session = None  # force offline response

    def offline_response(msgs):
        ctx = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        return {"choices": [{"message": {"content": f"Follow-up: {ctx} now with more details"}}]}

    monkeypatch.setattr(client, "_offline_response", offline_response)

    messages = client.build_prompt_with_memory([IMPROVEMENT_PATH], "What's next?")
    result = client.ask(messages, use_memory=False)
    text = result["choices"][0]["message"]["content"]

    assert mem.fetch_calls, "memory context was not retrieved"
    assert "Improve caching strategy" in text
