from workflow_graph import WorkflowGraph
from menace.unified_event_bus import UnifiedEventBus


def test_workflow_graph_event_sync(tmp_path):
    bus = UnifiedEventBus()
    g = WorkflowGraph(path=str(tmp_path / "graph.json"))
    g.attach_event_bus(bus)

    bus.publish("workflows:new", {"workflow_id": 1})
    if g._backend == "networkx":
        assert g.graph.has_node("1")
    else:
        assert "1" in g.graph["nodes"]

    bus.publish("workflows:updated", {"workflow_id": 1, "roi": 2.0})
    if g._backend == "networkx":
        assert g.graph.nodes["1"]["roi"] == 2.0
    else:
        assert g.graph["nodes"]["1"]["roi"] == 2.0

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
