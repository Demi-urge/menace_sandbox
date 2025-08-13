from datetime import datetime, timedelta, timezone

import pytest

from in_memory_queue import InMemoryQueue, QueuedTask
from neurosales.memory_queue import MemoryQueue, MessageEntry
from neurosales.memory_stack import CTAChain, MemoryStack


def test_memory_queue_decay():
    q = MemoryQueue(decay_seconds=60)
    past = datetime.now(timezone.utc) - timedelta(seconds=120)
    q._queue.append(MessageEntry(text="old", timestamp=past))
    q.add_message("new")
    messages = q.get_recent_messages()
    assert all(m.text != "old" for m in messages)


def test_memory_stack_decay():
    stack = MemoryStack(decay_seconds=60)
    past = datetime.now(timezone.utc) - timedelta(seconds=120)
    stack._stack.append(
        CTAChain(
            message_ts=past,
            reply_ts=past,
            escalation_ts=past,
            created_at=past,
        )
    )
    stack.push_chain(past, past, past)
    assert len(stack._stack) == 1
    assert stack._stack[0].created_at >= past


def test_in_memory_queue_decay():
    q = InMemoryQueue(decay_seconds=60)
    past = datetime.now(timezone.utc) - timedelta(seconds=120)
    old_task = QueuedTask(name="old", kwargs=None, created_at=past)
    q.sent.append(old_task)
    q.executed.append(old_task)
    q._expire_old_records()
    assert old_task not in q.sent
    assert old_task not in q.executed


def test_memory_queue_max_size():
    q = MemoryQueue(max_size=3)
    for i in range(5):
        q.add_message(str(i))
    messages = q.get_recent_messages()
    assert len(messages) == 3
    assert messages[0].text == "2"
    with pytest.raises(ValueError):
        MemoryQueue(max_size=2)


def test_memory_stack_push_pop():
    stack = MemoryStack()
    t1 = datetime.now(timezone.utc)
    t2 = t1 + timedelta(seconds=1)
    stack.push_chain(t1, t1, t1)
    stack.push_chain(t2, t2, t2)
    top = stack.peek_chain()
    assert top and top.message_ts == t2
    popped = stack.pop_chain()
    assert popped and popped.message_ts == t2
    popped2 = stack.pop_chain()
    assert popped2 and popped2.message_ts == t1
    assert stack.pop_chain() is None
