import menace.task_handoff_bot as thb
from menace.unified_event_bus import UnifiedEventBus

def test_workflow_events(tmp_path):
    bus = UnifiedEventBus()
    events = {"new": [], "update": [], "deleted": []}
    bus.subscribe("workflows:new", lambda t, e: events["new"].append(e))
    bus.subscribe("workflows:update", lambda t, e: events["update"].append(e))
    bus.subscribe("workflows:deleted", lambda t, e: events["deleted"].append(e))

    db = thb.WorkflowDB(tmp_path / "wf.db", event_bus=bus)
    rec = thb.WorkflowRecord(workflow=["a"], title="t", description="d")
    wid = db.add(rec)
    assert events["new"] and events["new"][0]["wid"] == wid

    db.update_statuses([wid], "active")
    assert events["update"] and events["update"][0] == {"workflow_id": wid, "status": "active"}

    db.remove(wid)
    assert events["deleted"] and events["deleted"][0] == {"workflow_id": wid}

    wid2 = db.add(thb.WorkflowRecord(workflow=["b"], title="t2", description="d2"))
    events["deleted"].clear()
    db.replace(wid2, thb.WorkflowRecord(workflow=["c"], title="t3", description="d3"))
    assert events["deleted"] and events["deleted"][0] == {"workflow_id": wid2}

