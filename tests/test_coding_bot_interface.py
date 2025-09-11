import inspect
import sys
import types

import pytest


class DummyManager:
    pass

stub = types.ModuleType("self_coding_manager")
stub.SelfCodingManager = DummyManager
sys.modules["menace.self_coding_manager"] = stub

import menace.coding_bot_interface as cbi
from menace.coding_bot_interface import self_coding_managed
from menace.bot_registry import BotRegistry


pytest.importorskip("networkx")


def test_bot_registered_with_module_path():
    registry = BotRegistry()

    @self_coding_managed
    class SampleBot:
        def __init__(self, manager: DummyManager):
            self.manager = manager

    SampleBot(bot_registry=registry, manager=DummyManager())
    name = "SampleBot"
    assert name in registry.graph.nodes
    assert registry.graph.nodes[name]["module"] == inspect.getfile(SampleBot)
