import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurosales.memory import ConversationMemory


def test_queue_limits():
    mem = ConversationMemory(max_messages=3)
    for i in range(5):
        mem.add_message("user", f"msg {i}")
    msgs = mem.get_recent_messages()
    assert len(msgs) == 3
    assert msgs[0].content == "msg 2"
    assert msgs[-1].content == "msg 4"


def test_stack_push_pop():
    mem = ConversationMemory()
    mem.push_stack("user", "hello")
    mem.push_stack("assistant", "hi")
    assert len(mem.stack) == 2
    msg = mem.pop_stack()
    assert msg.content == "hi"
    assert len(mem.stack) == 1
