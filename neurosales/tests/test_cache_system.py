import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from neurosales.cache_system import (
    SessionMemoryCache,
    UserPreferenceCache,
    ResponseRankingCache,
    ArchetypeInfluenceCache,
    MemoryDecaySystem,
)
from neurosales.user_preferences import PreferenceProfile


def test_session_memory_cache_expiry():
    cache = SessionMemoryCache(max_messages=3, ttl_seconds=0.1)
    cache.add_message("u", "user", "m1")
    time.sleep(0.2)
    cache.add_message("u", "user", "m2")
    msgs = cache.get_messages("u")
    assert len(msgs) == 1
    assert msgs[0].content == "m2"


def test_user_preference_cache():
    cache = UserPreferenceCache(ttl_seconds=0.1)
    profile = PreferenceProfile(keyword_freq={"x": 1.0})
    cache.set("u", profile)
    assert cache.get("u").keyword_freq["x"] == 1.0
    time.sleep(0.2)
    assert cache.get("u") is None


def test_response_ranking_cache_decay():
    cache = ResponseRankingCache(decay_factor=0.5, ttl_seconds=1)
    cache.add_response("r1", 1.0)
    cache.add_response("r2", 0.8)
    top = cache.get_top(1)
    assert top[0][0] == "r1"
    cache.add_response("r3", 0.9)
    top2 = cache.get_top(1)
    assert top2[0][0] in {"r1", "r3"}


def test_archetype_influence_cache():
    cache = ArchetypeInfluenceCache(ttl_seconds=0.1)
    cache.set("a1", {"a2": 1.0})
    assert cache.get("a1")["a2"] == 1.0
    time.sleep(0.2)
    assert cache.get("a1") == {}


def test_memory_decay_system():
    mcache = SessionMemoryCache(ttl_seconds=0.1)
    pcache = UserPreferenceCache(ttl_seconds=0.1)
    mcache.add_message("u", "user", "hi")
    pcache.set("u", PreferenceProfile())
    system = MemoryDecaySystem(mcache, pcache)
    time.sleep(0.2)
    system.decay()
    assert mcache.get_messages("u") == []
    assert pcache.get("u") is None
