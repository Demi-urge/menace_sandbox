import sys
import types
import json

# stub modules required by ChatGPTClient and GPTMemory
sys.modules.setdefault(
    "menace.database_manager", types.SimpleNamespace(DB_PATH="db", search_models=lambda *a, **k: [])
)
sys.modules.setdefault(
    "menace.database_management_bot", types.SimpleNamespace(DatabaseManagementBot=object)
)

# lightweight in-memory MenaceMemoryManager
stub_mm = types.ModuleType("menace_memory_manager")

class _Entry:
    def __init__(self, key, data, tags=""):
        self.key = key
        self.data = data
        self.tags = tags

class _StubManager:
    def __init__(self):
        self.items: list[_Entry] = []

    def store(self, key, data, tags="", **_: object) -> int:
        self.items.append(_Entry(key, data, tags))
        return len(self.items)

    def search(self, text, limit=20):
        res = [e for e in self.items if text in e.data or text in e.key]
        return res[:limit]

    def summarise_memory(self, key, limit=20, ratio=0.2, store=True, condense=False):
        entries = [e for e in self.items if e.key == key][-limit:]
        if not entries:
            return ""
        text = "\n".join(e.data for e in entries)
        summary = stub_mm._summarise_text(text, ratio)
        if store:
            self.store(f"{key}:summary", summary, tags="summary")
        if condense:
            self.items = [e for e in self.items if e.key != key]
        return summary

stub_mm.MenaceMemoryManager = _StubManager
stub_mm._summarise_text = lambda text, ratio=0.2: text
sys.modules.setdefault("menace_memory_manager", stub_mm)

# stub sentence_transformers to avoid heavy import
stub_st = types.ModuleType("sentence_transformers")
class _DummyModel:
    def encode(self, text):
        return [0.0]
stub_st.SentenceTransformer = _DummyModel
sys.modules.setdefault("sentence_transformers", stub_st)

import menace.chatgpt_idea_bot as cib
from gpt_memory import GPTMemory


def test_build_prompt_injects_summary_and_logs(monkeypatch):
    mem = GPTMemory()
    # previous interaction stored for tag 'topic'
    mem.log_interaction("early prompt", "early resp", ["topic"])

    client = cib.ChatGPTClient(gpt_memory=mem)
    client.session = None  # offline mode
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "later resp"}}]},
    )

    msgs = client.build_prompt_with_memory(["topic"], "new question")
    assert msgs[0]["role"] == "system"
    assert "early prompt" in msgs[0]["content"]

    client.ask(msgs)
    # confirm interaction was logged with default tags
    logs = [
        json.loads(e.data)
        for e in mem.manager.items
        if e.key.startswith("gpt:")
    ]
    assert any(
        l["prompt"] == "new question" and l["response"] == "later resp" for l in logs
    )


def test_summarize_and_prune_via_client(monkeypatch):
    mem = GPTMemory()
    client = cib.ChatGPTClient(gpt_memory=mem)
    client.session = None
    monkeypatch.setattr(
        client,
        "_offline_response",
        lambda msgs: {"choices": [{"message": {"content": "resp"}}]},
    )

    for i in range(3):
        client.ask([{"role": "user", "content": f"ask{i}"}], tags=["insight"])

    assert len([e for e in mem.manager.items if e.key == "gpt:insight"]) == 3
    summary = mem.summarize_and_prune("insight")
    assert "ask0" in summary
    assert [e for e in mem.manager.items if e.key == "gpt:insight"] == []
    assert [e for e in mem.manager.items if e.key == "gpt:insight:summary"]
