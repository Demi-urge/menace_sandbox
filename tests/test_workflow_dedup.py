import logging
import importlib
import sys


def test_duplicate_workflow_skipped(tmp_path, caplog, monkeypatch):
    sys.modules.pop("menace.task_handoff_bot", None)
    thb = importlib.import_module("menace.task_handoff_bot")
    router = thb.init_db_router(
        "wfdup", str(tmp_path / "local.db"), str(tmp_path / "shared.db")
    )
    monkeypatch.setattr(thb.WorkflowDB, "add_embedding", lambda *a, **k: None)
    db = thb.WorkflowDB(tmp_path / "wf.db", router=router)
    wf1 = thb.WorkflowRecord(
        workflow=["a"],
        action_chains=["x"],
        argument_strings=["y"],
        description="desc",
    )
    with caplog.at_level(logging.WARNING):
        wid1 = db.add(wf1)
        wid2 = db.add(
            thb.WorkflowRecord(
                workflow=["a"],
                action_chains=["x"],
                argument_strings=["y"],
                description="desc",
            )
        )
    assert wid1 == wid2
    assert db.conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 1
    assert "duplicate workflow" in caplog.text.lower()
