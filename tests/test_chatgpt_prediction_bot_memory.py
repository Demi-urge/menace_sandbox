import types
import sys

sys.modules.setdefault(
    "menace.database_manager",
    types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: []),
)
sys.modules.setdefault(
    "menace.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object)
)

import menace.chatgpt_prediction_bot as cpb


class FakeMemory:
    def __init__(self):
        self.logged = []
        self.context_calls = []

    def fetch_context(self, tags):
        self.context_calls.append(list(tags))
        return "ctx"

    def log_interaction(self, prompt, response, tags):
        self.logged.append((prompt, response, tags))


def test_enhancement_uses_memory(monkeypatch):
    mem = FakeMemory()
    monkeypatch.setattr(cpb, "joblib", None)
    monkeypatch.setattr(cpb, "sentiment_score", lambda text: 0.0)

    bot = cpb.ChatGPTPredictionBot(gpt_memory=mem)
    bot.client.session = None
    monkeypatch.setattr(
        bot.client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "resp"}}]},
    )

    bot.evaluate_enhancement("idea", "rationale")

    assert cpb.INSIGHT in mem.context_calls[0]
    assert mem.logged[0][0].startswith("Evaluate enhancement")
    assert mem.logged[0][1] == "resp"
    assert cpb.INSIGHT in mem.logged[0][2]
