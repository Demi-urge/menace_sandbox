import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.mongo_memory import MongoMemorySystem
import time


def test_short_and_long_term_storage():
    mem = MongoMemorySystem(ttl_seconds=1)
    mem.add_message('u1', 'user', 'hello world')
    assert mem.short_term['u1'] and mem.long_term['u1']
    # expire
    mem.short_term['u1'][0].timestamp -= 2
    mem.add_message('u1', 'assistant', 'hi')
    recent = mem.recent_messages('u1')
    assert all(m.timestamp >= time.time() - 1 for m in recent)
    assert len(mem.long_term['u1']) == 2


def test_preferences_and_relations():
    mem = MongoMemorySystem()
    for _ in range(3):
        mem.add_message('u2', 'user', 'cats cats')
    prefs = mem.preferences['u2']
    assert prefs['cats'].score > 0.5
    mem.update_archetype_relation('a', 'b', 1.0)
    assert mem.get_relation('a', 'b') == 1.0
