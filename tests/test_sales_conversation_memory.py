import time
from sales_conversation_memory import SalesConversationMemory


def test_max_length():
    mem = SalesConversationMemory()
    start = time.time()
    for i in range(7):
        mem.add_message(f"msg {i}", "user", timestamp=start + i)
    recent = mem.get_recent()
    assert len(recent) == 6
    assert recent[0]["text"] == "msg 1"
    assert recent[-1]["text"] == "msg 6"


def test_ttl_expiration(monkeypatch):
    mem = SalesConversationMemory(ttl=10)
    mem.add_message("old", "user", timestamp=0)
    monkeypatch.setattr(time, "time", lambda: 11)
    assert mem.get_recent() == []
