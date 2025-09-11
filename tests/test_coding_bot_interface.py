import inspect

import pytest

from menace.coding_bot_interface import self_coding_managed
from menace.bot_registry import BotRegistry


pytest.importorskip("networkx")


def test_bot_registered_with_module_path():
    registry = BotRegistry()

    @self_coding_managed
    class SampleBot:
        pass

    SampleBot(bot_registry=registry)
    name = "SampleBot"
    assert name in registry.graph.nodes
    assert registry.graph.nodes[name]["module"] == inspect.getfile(SampleBot)
