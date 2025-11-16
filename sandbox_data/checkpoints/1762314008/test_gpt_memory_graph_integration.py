import pytest

from menace_sandbox.gpt_memory import GPTMemoryManager
from knowledge_graph import KnowledgeGraph


class DummyBus:
    """Minimal event bus capturing subscriptions and synchronously dispatching."""

    def __init__(self) -> None:
        self._subs = {}

    def subscribe(self, topic, callback):
        self._subs.setdefault(topic, []).append(callback)

    def publish(self, topic, payload):
        for cb in self._subs.get(topic, []):
            cb(topic, payload)


def test_memory_event_updates_knowledge_graph(tmp_path):
    pytest.importorskip("networkx")
    bus = DummyBus()
    mgr = GPTMemoryManager(db_path=":memory:", event_bus=bus)
    kg = KnowledgeGraph(tmp_path / "kg.gpickle")
    kg.listen_to_memory(bus, mgr)
    mgr.log_interaction(
        "idea1",
        "result",
        tags=["bot:alpha", "code:module.py", "error:ValueError"],  # path-ignore
    )
    inode = "insight:idea1"
    assert inode in kg.graph
    assert (inode, "bot:alpha") in kg.graph.edges
    assert (inode, "code:module.py") in kg.graph.edges  # path-ignore
    assert (inode, "error_category:ValueError") in kg.graph.edges
