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


def test_personal_data_removed_after_ttl(monkeypatch):
    mem = SalesConversationMemory(ttl=1)
    mem.add_message("Alice's email is alice@example.com", "user", timestamp=0)
    # Before TTL expires, message should be present without extra identifiers
    monkeypatch.setattr(time, "time", lambda: 0.5)
    assert mem.get_recent() == [
        {"text": "Alice's email is alice@example.com", "role": "user"}
    ]
    # After TTL expires, message should be purged
    monkeypatch.setattr(time, "time", lambda: 2)
    assert mem.get_recent() == []


def test_cta_push_pop():
    mem = SalesConversationMemory()
    mem.push_cta({"event": "message"})
    mem.push_cta({"event": "reply"})
    assert mem.pop_cta() == {"event": "reply"}
    assert mem.pop_cta() == {"event": "message"}
    assert mem.pop_cta() is None


def test_cta_reset():
    mem = SalesConversationMemory()
    mem.push_cta({"event": "message"})
    mem.push_cta({"event": "reply"})
    mem.clear_cta_stack()
    assert mem.cta_stack == []
