import sys
import types
import pytest

pytest.importorskip("networkx")

from menace.unified_event_bus import UnifiedEventBus


class DummyRouter:
    def __init__(self):
        self.queries = []

    def query_all(self, term: str):
        self.queries.append(term)
        return {}


stub = types.ModuleType("menace.db_router")
stub.DBRouter = DummyRouter
sys.modules["menace.db_router"] = stub

from menace.bot_db_utils import wrap_bot_methods
from menace.bot_registry import BotRegistry


class BotA:
    name = "A"

    def act(self, other):
        return other.ping()


class BotB:
    name = "B"

    def ping(self):
        return "pong"


def test_event_bus_records_bot_call():
    router = DummyRouter()
    registry = BotRegistry()
    bus = UnifiedEventBus()
    events = []
    bus.subscribe("bot:call", lambda t, e: events.append(e))

    a = BotA()
    b = BotB()
    wrap_bot_methods(a, router, registry)
    wrap_bot_methods(b, router, registry)
    a.event_bus = bus
    b.event_bus = bus

    a.act(b)

    assert events and events[0] == {"from": "A", "to": "B"}
    assert registry.graph.has_edge("A", "B")
