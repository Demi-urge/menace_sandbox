import sys
import types
from datetime import datetime


class DummyEntry:
    def __init__(self, key, data, version, tags, ts=""):
        self.key = key
        self.data = data
        self.version = version
        self.tags = tags
        self.ts = ts or datetime.utcnow().isoformat()


class DummyManager:
    def __init__(self, path=":memory:"):
        import sqlite3
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory(rowid INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, data TEXT, version INTEGER, tags TEXT, ts TEXT)"
        )
        self.has_fts = False

    def log(self, entry, *, bot_id=None, info_id=None):
        self.conn.execute(
            "INSERT INTO memory(key, data, version, tags, ts) VALUES (?,?,?,?,?)",
            (entry.key, entry.data, entry.version, entry.tags, entry.ts),
        )
        self.conn.commit()

    def search(self, text, limit):
        cur = self.conn.execute(
            "SELECT key, data, version, tags, ts FROM memory WHERE data LIKE ? LIMIT ?",
            (f"%{text}%", limit),
        )
        rows = cur.fetchall()
        return [DummyEntry(*row) for row in rows]

    def store(self, key, data, tags="", *, bot_id=None, info_id=None):
        if isinstance(data, dict):
            import json

            data = json.dumps(data)
        entry = DummyEntry(key, data, 1, tags)
        self.log(entry)
        return 1


def _summarise_text(text: str, ratio: float = 0.2) -> str:
    return text[: max(1, int(len(text) * ratio))]


dummy_module = types.ModuleType("menace_memory_manager")
dummy_module.MenaceMemoryManager = DummyManager
dummy_module.MemoryEntry = DummyEntry
dummy_module._summarise_text = _summarise_text
sys.modules.setdefault("menace_memory_manager", dummy_module)

from gpt_memory import GPTMemory


def test_log_and_search():
    manager = DummyManager()
    memory = GPTMemory(manager)

    memory.log_interaction("hello", "hi", ["greeting"])
    results = memory.search_context("hello", 5)

    assert results, "expected at least one result"
    record = results[0]
    assert record["prompt"] == "hello"
    assert record["response"] == "hi"
    metadata = record["metadata"]
    assert metadata["feedback"] == []
    assert metadata["error_fixes"] == []
    assert metadata["improvement_paths"] == []


def test_summarises_long_entries():
    manager = DummyManager()
    memory = GPTMemory(manager, summary_threshold=10)

    prompt = "p" * 20
    response = "r" * 20
    memory.log_interaction(prompt, response, ["test"])

    results = memory.search_context("p" * 5, 5)
    assert results
    assert "summary" in results[0]
    assert results[0]["summary"].strip()


def test_retention_prunes_and_merges():
    manager = DummyManager()
    memory = GPTMemory(manager, max_entries=2, summary_threshold=100)

    memory.log_interaction("a1", "b1", [])
    memory.log_interaction("a2", "b2", [])
    memory.log_interaction("a3", "b3", [])  # triggers retention

    cur = manager.conn.execute("SELECT key FROM memory WHERE key LIKE 'a1%'")
    assert cur.fetchone() is None
    cur = manager.conn.execute("SELECT key FROM memory WHERE key='memory:summary'")
    assert cur.fetchone() is not None
    cur = manager.conn.execute("SELECT COUNT(*) FROM memory")
    count = cur.fetchone()[0]
    assert count <= 3


class DummyResp:
    status_code = 200

    def __init__(self, text):
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


class RecordingSession:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.messages = []

    def post(self, url, headers=None, json=None, timeout=0):
        self.messages.append(json["messages"])
        return DummyResp(next(self.texts))


class DummyClient:
    def __init__(self, session):
        self.session = session

    def ask(self, messages, knowledge=None):
        memory = knowledge
        user_prompt = messages[-1]["content"]
        messages_for_api = messages
        if memory:
            contexts = memory.search_context(user_prompt)
            if contexts:
                ctx_parts = [
                    f"Prompt: {c.get('prompt','')}\nResponse: {c.get('response','')}"
                    for c in contexts
                ]
                context_text = "\n\n".join(ctx_parts)
                messages_for_api = [{"role": "system", "content": context_text}] + messages
        resp = self.session.post("", json={"messages": messages_for_api})
        text = resp.json()["choices"][0]["message"]["content"]
        if memory:
            memory.log_interaction(user_prompt, text, [])
        return resp.json()


def test_chatgptclient_context_and_logging(tmp_path):
    session = RecordingSession(["r1", "r2"])
    memory = GPTMemory(DummyManager(path=tmp_path / "mem.db"))
    client = DummyClient(session)

    client.ask([{"role": "user", "content": "hello"}], knowledge=memory)
    client.ask([{"role": "user", "content": "hello"}], knowledge=memory)

    assert len(session.messages) == 2
    assert len(session.messages[0]) == 1
    ctx = session.messages[1][0]["content"]
    assert "Prompt: hello" in ctx
    assert "Response: r1" in ctx


def test_memory_persists_between_sessions(tmp_path):
    db = tmp_path / "mem.db"

    session1 = RecordingSession(["first"])
    mem1 = GPTMemory(DummyManager(path=db))
    client1 = DummyClient(session1)
    client1.ask([{"role": "user", "content": "persist"}], knowledge=mem1)
    mem1.manager.conn.close()

    session2 = RecordingSession(["second"])
    mem2 = GPTMemory(DummyManager(path=db))
    client2 = DummyClient(session2)
    client2.ask([{"role": "user", "content": "persist"}], knowledge=mem2)

    ctx = session2.messages[0][0]["content"]
    assert "Prompt: persist" in ctx
    assert "Response: first" in ctx
