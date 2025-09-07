import types
import types
import sys
from pathlib import Path
import pytest

pkg = types.ModuleType("menace")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
pkg.RAISE_ERRORS = False
sys.modules["menace"] = pkg

vector_service_pkg = types.ModuleType("vector_service")
vector_service_pkg.__path__ = []
vector_service_pkg.SharedVectorService = object
vector_service_pkg.CognitionLayer = object
class _StubContextBuilder:
    def refresh_db_weights(self):
        pass
ctx_mod = types.ModuleType("vector_service.context_builder")
ctx_mod.ContextBuilder = _StubContextBuilder
sys.modules["vector_service"] = vector_service_pkg
sys.modules["vector_service.context_builder"] = ctx_mod
sys.modules["menace.shared_gpt_memory"] = types.SimpleNamespace(GPT_MEMORY_MANAGER=None)

sys.modules.setdefault(
    "menace.database_manager",
    types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: []),
)
sys.modules.setdefault(
    "menace.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object)
)

import menace.chatgpt_prediction_bot as cpb

DummyBuilder = _StubContextBuilder

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

    bot = cpb.ChatGPTPredictionBot(gpt_memory=mem, context_builder=DummyBuilder())
    if bot.client is None:
        pytest.skip("ChatGPTClient unavailable")
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
