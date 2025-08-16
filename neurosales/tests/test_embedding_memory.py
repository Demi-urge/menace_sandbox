import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.embedding_memory import EmbeddingConversationMemory


def test_basic_storage():
    mem = EmbeddingConversationMemory(max_messages=3)
    mem.add_message('user', 'hello world')
    mem.add_message('assistant', 'hi there')
    mem.add_message('user', 'another message')
    recent = mem.get_recent_messages()
    assert len(recent) == 3
    assert recent[0].content == 'hello world'
    sim = mem.most_similar('hello')
    assert isinstance(sim, list)
