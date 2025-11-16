from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .user_preferences import PreferenceProfile


@dataclass
class CachedMessage:
    timestamp: float
    role: str
    content: str


class SessionMemoryCache:
    """Cache recent messages per session with TTL."""

    def __init__(self, max_messages: int = 6, ttl_seconds: Optional[int] = 3600) -> None:
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self.store: Dict[str, Deque[CachedMessage]] = {}

    def add_message(self, user_id: str, role: str, content: str) -> None:
        q = self.store.setdefault(user_id, deque())
        q.append(CachedMessage(time.time(), role, content))
        if len(q) > self.max_messages:
            q.popleft()
        self._prune(user_id)

    def get_messages(self, user_id: str) -> List[CachedMessage]:
        self._prune(user_id)
        return list(self.store.get(user_id, []))

    def _prune(self, user_id: str) -> None:
        if self.ttl_seconds is None:
            return
        q = self.store.get(user_id)
        if not q:
            return
        expiry = time.time() - self.ttl_seconds
        while q and q[0].timestamp < expiry:
            q.popleft()


class UserPreferenceCache:
    """Cache user preference profiles with TTL."""

    def __init__(self, ttl_seconds: Optional[int] = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.store: Dict[str, Tuple[PreferenceProfile, float]] = {}

    def set(self, user_id: str, profile: PreferenceProfile) -> None:
        self.store[user_id] = (profile, time.time())

    def get(self, user_id: str) -> Optional[PreferenceProfile]:
        entry = self.store.get(user_id)
        if not entry:
            return None
        profile, ts = entry
        if self.ttl_seconds is not None and time.time() - ts > self.ttl_seconds:
            del self.store[user_id]
            return None
        return profile


@dataclass(order=True)
class _HeapItem:
    priority: float
    timestamp: float
    response: str = field(compare=False)


class ResponseRankingCache:
    """Max-heap cache of responses that decays over time."""

    def __init__(self, decay_factor: float = 0.95, ttl_seconds: Optional[int] = 3600) -> None:
        self.decay_factor = decay_factor
        self.ttl_seconds = ttl_seconds
        self.heap: List[_HeapItem] = []

    def add_response(self, response: str, score: float) -> None:
        item = _HeapItem(-score, time.time(), response)
        heapq.heappush(self.heap, item)

    def get_top(self, n: int = 1) -> List[Tuple[str, float]]:
        self._prune()
        results: List[Tuple[str, float]] = []
        temp: List[_HeapItem] = []
        while self.heap and len(results) < n:
            item = heapq.heappop(self.heap)
            score = -item.priority
            results.append((item.response, score))
            decayed = _HeapItem(-(score * self.decay_factor), item.timestamp, item.response)
            temp.append(decayed)
        for t in temp:
            heapq.heappush(self.heap, t)
        return results

    def _prune(self) -> None:
        if self.ttl_seconds is None:
            return
        expiry = time.time() - self.ttl_seconds
        self.heap = [i for i in self.heap if i.timestamp >= expiry]
        heapq.heapify(self.heap)


class ArchetypeInfluenceCache:
    """Cache archetype rankings and faction influence."""

    def __init__(self, ttl_seconds: Optional[int] = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.store: Dict[str, Tuple[Dict[str, float], float]] = {}

    def set(self, archetype: str, influence: Dict[str, float]) -> None:
        self.store[archetype] = (influence, time.time())

    def get(self, archetype: str) -> Dict[str, float]:
        entry = self.store.get(archetype)
        if not entry:
            return {}
        data, ts = entry
        if self.ttl_seconds is not None and time.time() - ts > self.ttl_seconds:
            del self.store[archetype]
            return {}
        return data


class MemoryDecaySystem:
    """Apply TTL-based decay across multiple caches."""

    def __init__(self, *caches: Any) -> None:
        self.caches = caches

    def decay(self) -> None:
        for cache in self.caches:
            prune = getattr(cache, "_prune", None)
            if callable(prune):
                if hasattr(cache, "store") and isinstance(cache.store, dict):
                    for key in list(cache.store.keys()):
                        prune(key)
                else:
                    prune()

