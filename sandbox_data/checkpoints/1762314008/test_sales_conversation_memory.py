import neurosales.memory_queue as mq
from sales_conversation_memory import SalesConversationMemory
import time


def test_queue_retains_latest_messages():
    q = mq.MemoryQueue(max_size=3)
    for i in range(5):
        q.add_message(f"m{i}")
    assert [m.text for m in q.get_recent_messages()] == ["m2", "m3", "m4"]

    q = mq.MemoryQueue(max_size=6)
    for i in range(10):
        q.add_message(f"n{i}")
    assert len(q.get_recent_messages()) == 6
    assert [m.text for m in q.get_recent_messages()] == [f"n{i}" for i in range(4, 10)]


def test_cta_stack_tracks_and_resets():
    mem = SalesConversationMemory()
    base = time.time()
    mem.push_cta({"step": 1}, timestamp=base)
    mem.push_cta({"step": 2}, timestamp=base + 1)
    assert mem.cta_stack == [{"step": 1}, {"step": 2}]
    assert mem.pop_cta() == {"step": 2}
    assert mem.cta_stack == [{"step": 1}]
    mem.clear_cta_stack()
    assert mem.cta_stack == []


def test_prune_removes_expired_entries():
    mem = SalesConversationMemory(ttl=5.0)
    mem.add_message("old", "user", timestamp=0.0)
    mem.push_cta({"step": "old"}, timestamp=0.0)
    assert len(mem._messages) == 1
    assert len(mem._cta_stack) == 1
    mem.prune(current_time=10.0)
    assert mem.get_recent() == []
    assert mem.cta_stack == []
