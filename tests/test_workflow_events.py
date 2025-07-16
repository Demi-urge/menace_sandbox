import menace.task_handoff_bot as thb
from menace.unified_event_bus import UnifiedEventBus

def test_workflow_events(tmp_path):
    bus = UnifiedEventBus()
    events = {"new": [], "update": []}
    bus.subscribe("workflows:new", lambda t, e: events["new"].append(e))
    bus.subscribe("workflows:update", lambda t, e: events["update"].append(e))

    db = thb.WorkflowDB(tmp_path / "wf.db", event_bus=bus)
    rec = thb.WorkflowRecord(workflow=["a"], title="t", description="d")
    wid = db.add(rec)
    assert events["new"] and events["new"][0]["wid"] == wid

    db.update_statuses([wid], "active")
    assert events["update"] and events["update"][0] == {"workflow_id": wid, "status": "active"}

