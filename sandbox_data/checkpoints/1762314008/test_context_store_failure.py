import os
import logging
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import sys
sys.modules.setdefault(
    "menace.chatgpt_idea_bot",
    types.SimpleNamespace(ChatGPTClient=object),
)
sys.modules.setdefault("menace.database_manager", types.ModuleType("db"))

import menace.query_bot as qb


def test_add_logs_and_falls_back(caplog):
    store = qb.ContextStore()
    class FakeRedis:
        def rpush(self, *a, **k):
            raise RuntimeError('boom')
        def lrange(self, *a, **k):
            return []
    store.client = FakeRedis()
    caplog.set_level(logging.WARNING)
    store.add('c1', 'hello')
    store.client = None  # ensure history uses memory
    assert 'redis rpush failed' in caplog.text
    assert 'falling back to memory store' in caplog.text
    assert store.history('c1') == ['hello']


def test_history_logs_and_falls_back(caplog):
    store = qb.ContextStore()
    store.memory['c2'] = ['a']
    class FakeRedis:
        def lrange(self, *a, **k):
            raise RuntimeError('boom')
    store.client = FakeRedis()
    caplog.set_level(logging.WARNING)
    assert store.history('c2') == ['a']
    assert 'redis lrange failed' in caplog.text
    assert 'falling back to memory store' in caplog.text
