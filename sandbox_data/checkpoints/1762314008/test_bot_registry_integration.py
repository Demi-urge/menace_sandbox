import sys
import types
import pytest

pytest.importorskip("networkx")


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


def test_registry_records_cross_bot_call():
    reg = BotRegistry()
    router = DummyRouter()
    a = BotA()
    b = BotB()
    wrap_bot_methods(a, router, reg)
    wrap_bot_methods(b, router, reg)

    a.act(b)

    assert reg.graph.has_edge("A", "B")
