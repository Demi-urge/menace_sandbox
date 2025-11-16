import pytest
from workflow_graph import WorkflowGraph
from menace.unified_event_bus import UnifiedEventBus
import threading


def test_workflow_graph_event_sync(tmp_path):
    bus = UnifiedEventBus()
    g = WorkflowGraph(path=str(tmp_path / "graph.json"))
    g.attach_event_bus(bus)

    impacts: list[dict] = []
    bus.subscribe("workflows:impact_wave", lambda t, e: impacts.append(e))

    bus.publish("workflows:new", {"workflow_id": 1})
    if g._backend == "networkx":
        assert g.graph.has_node("1")
    else:
        assert "1" in g.graph["nodes"]

    bus.publish(
        "workflows:update",
        {"workflow_id": 1, "roi": 2.0, "roi_delta": 0.5},
    )
    if g._backend == "networkx":
        assert g.graph.nodes["1"]["roi"] == 2.0
    else:
        assert g.graph["nodes"]["1"]["roi"] == 2.0

    assert impacts and impacts[0]["start_id"] == "1"
    assert impacts[0]["impact_map"]["1"]["roi"] == pytest.approx(0.5)

    bus.publish("workflows:deleted", {"workflow_id": 1})
    if g._backend == "networkx":
        assert "1" not in g.graph
    else:
        assert "1" not in g.graph["nodes"]

    bus.publish("workflows:new", {"workflow_id": 2})
    bus.publish("workflows:refactor", {"workflow_id": 2})
    if g._backend == "networkx":
        assert "2" not in g.graph
    else:
        assert "2" not in g.graph["nodes"]


def test_concurrent_workflow_updates(tmp_path):
    bus = UnifiedEventBus()
    g = WorkflowGraph(path=str(tmp_path / "graph.json"))
    g.attach_event_bus(bus)

    bus.publish("workflows:new", {"workflow_id": 1})

    impacts: list[dict] = []
    bus.subscribe("workflows:impact_wave", lambda t, e: impacts.append(e))

    def _publish(val: int) -> None:
        bus.publish("workflows:update", {"workflow_id": 1, "roi": val})

    threads = [threading.Thread(target=_publish, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if g._backend == "networkx":
        roi = g.graph.nodes["1"]["roi"]
    else:
        roi = g.graph["nodes"]["1"]["roi"]

    assert roi in range(5)
    assert len(impacts) == 5


def test_manual_weight_refresh_preserves_manual_edges(tmp_path):
    g = WorkflowGraph(path=str(tmp_path / "graph.json"))
    g.add_workflow("1")
    g.add_workflow("2")
    g.add_workflow("3")

    g.add_dependency("1", "2", impact_weight=0.3)
    g.add_dependency("2", "3", resource_weight=0.2)

    if g._backend == "networkx":
        manual_before = g.graph["1"]["2"]["impact_weight"]
        auto_before = g.graph["2"]["3"]["impact_weight"]
    else:
        manual_before = g.graph["edges"]["1"]["2"]["impact_weight"]
        auto_before = g.graph["edges"]["2"]["3"]["impact_weight"]

    assert manual_before == pytest.approx(0.3)
    assert auto_before == pytest.approx(0.2)

    g.refresh_edges()

    if g._backend == "networkx":
        manual_after = g.graph["1"]["2"]["impact_weight"]
        auto_after = g.graph["2"]["3"]["impact_weight"]
    else:
        manual_after = g.graph["edges"]["1"]["2"]["impact_weight"]
        auto_after = g.graph["edges"]["2"]["3"]["impact_weight"]

    assert manual_after == pytest.approx(0.3)
    assert auto_after != pytest.approx(0.2)


def test_refactor_rejects_cycles(tmp_path):
    bus = UnifiedEventBus(rethrow_errors=True)
    g = WorkflowGraph(path=str(tmp_path / "graph.json"))
    g.attach_event_bus(bus)

    bus.publish("workflows:new", {"workflow_id": "1"})
    bus.publish("workflows:new", {"workflow_id": "2"})
    g.add_dependency("1", "2")

    def _refactor_handler(_t: str, _e: object) -> None:
        g.add_workflow("1")
        g.add_dependency("1", "2")
        g.add_dependency("2", "1")

    bus.subscribe("workflows:refactor", _refactor_handler)

    with pytest.raises(ValueError):
        bus.publish("workflows:refactor", {"workflow_id": "1"})

    if g._backend == "networkx":
        assert not g.graph.has_edge("2", "1")
    else:
        assert "1" not in g.graph.get("edges", {}).get("2", {})
