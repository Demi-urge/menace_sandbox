import pytest
pytest.importorskip("networkx")

import menace.event_collector as ec
from menace.unified_event_bus import UnifiedEventBus
from menace.bot_registry import BotRegistry
from menace.neuroplasticity import PathwayDB


def test_collector_updates_registry_and_pathway(tmp_path):
    bus = UnifiedEventBus()
    registry = BotRegistry()
    pdb = PathwayDB(tmp_path / "p.db")
    ec.EventCollector(bus, registry=registry, pathway_db=pdb)

    bus.publish("bot:call", {"from": "A", "to": "B"})
    bus.publish("workflows:new", {"id": 1})

    assert registry.graph.has_edge("A", "B")
    rows = list(pdb.conn.execute("SELECT actions FROM pathways"))
    actions = {r[0] for r in rows}
    assert "workflows:new" in actions


def test_subscribe_logs_exception(caplog):
    class BadBus:
        def subscribe(self, *a, **k):
            raise RuntimeError("boom")

    caplog.set_level("ERROR")
    ec.EventCollector(BadBus())
    assert "failed to subscribe" in caplog.text


def test_event_handler_logs_exceptions(caplog, tmp_path):
    bus = UnifiedEventBus()

    class BadRegistry:
        def register_interaction(self, *a, **k):
            raise RuntimeError("boom")

    class BadPathwayDB:
        def log(self, *a, **k):
            raise RuntimeError("boom")

    caplog.set_level("ERROR")
    collector = ec.EventCollector(bus, registry=BadRegistry(), pathway_db=BadPathwayDB())
    bus.publish("bot:call", {"from": "A", "to": "B"})
    bus.publish("workflows:new", {})
    text = caplog.text
    assert "failed registering interaction" in text
    assert "failed logging pathway record" in text

