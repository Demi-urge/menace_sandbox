"""Tests for :mod:`menace.coding_bot_interface`."""

import inspect
import sys
import types
import importlib.util
from pathlib import Path

import pytest


class DummyManager:
    """Lightweight stand‑in for :class:`SelfCodingManager`."""


class DummyDB:
    def log_eval(self, *_, **__):
        pass


class DummyDataBot:
    def __init__(self):
        self.db = DummyDB()

    def roi(self, _name):  # pragma: no cover - trivial
        return 0.0


menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = []
sys.modules["menace"] = menace_pkg

stub_manager = types.ModuleType("menace.self_coding_manager")
stub_manager.SelfCodingManager = DummyManager
sys.modules["menace.self_coding_manager"] = stub_manager

nx = pytest.importorskip("networkx")


class DummyRegistry:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def register_bot(self, name: str) -> None:
        self.graph.add_node(name)

    def update_bot(self, name: str, module: str) -> None:  # pragma: no cover - simple
        self.graph.nodes[name]["module"] = module


stub_registry = types.ModuleType("menace.bot_registry")
stub_registry.BotRegistry = DummyRegistry
sys.modules["menace.bot_registry"] = stub_registry

spec = importlib.util.spec_from_file_location(
    "menace.coding_bot_interface", Path(__file__).resolve().parents[1] / "coding_bot_interface.py"
)
cbi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cbi)
sys.modules["menace.coding_bot_interface"] = cbi

self_coding_managed = cbi.self_coding_managed
from menace.bot_registry import BotRegistry


def test_bot_registered_with_module_path():
    registry = BotRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed
    class SampleBot:
        def __init__(self, manager: DummyManager):
            self.manager = manager

    SampleBot(bot_registry=registry, data_bot=data_bot, manager=DummyManager())
    name = "SampleBot"
    assert name in registry.graph.nodes
    assert registry.graph.nodes[name]["module"] == inspect.getfile(SampleBot)


def test_resolve_helpers_requires_registry():
    data_bot = DummyDataBot()
    obj = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="BotRegistry"):
        cbi._resolve_helpers(obj, None, data_bot)


def test_resolve_helpers_requires_data_bot():
    registry = BotRegistry()
    obj = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="DataBot"):
        cbi._resolve_helpers(obj, registry, None)


def test_managed_bot_requires_registry():
    data_bot = DummyDataBot()

    @self_coding_managed
    class SampleBot:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="BotRegistry"):
        SampleBot(data_bot=data_bot)


def test_managed_bot_requires_data_bot():
    registry = BotRegistry()

    @self_coding_managed
    class SampleBot:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="DataBot"):
        SampleBot(bot_registry=registry)

