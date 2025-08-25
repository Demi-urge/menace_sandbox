import json
import logging
from types import SimpleNamespace

import pytest
import menace.task_handoff_bot as thb

zmq = pytest.importorskip("zmq")


def test_compile_to_json():
    info = thb.TaskInfo(
        name="t1",
        dependencies=["a"],
        resources={"cpu": 1.0},
        schedule="now",
        code="print('x')",
        metadata={"k": "v"},
    )
    pkg = thb.TaskPackage(tasks=[info])
    data = json.loads(pkg.to_json())
    assert data["tasks"][0]["name"] == "t1"


def test_send_package_fallback(monkeypatch):
    bot = thb.TaskHandoffBot(api_url="http://x")
    pkg = bot.compile([
        thb.TaskInfo(
            name="t",
            dependencies=[],
            resources={},
            schedule="",
            code="",
            metadata={},
        )
    ])
    monkeypatch.setattr(
        thb.requests,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(Exception("fail")),
    )  # noqa: E501
    sent = {}
    bot.channel = SimpleNamespace(
        basic_publish=lambda exc, queue, body: sent.update({"body": body})
    )
    bot.send_package(pkg)
    assert sent["body"] == pkg.to_json()
    bot.close()


def test_respond_to_queries():
    bot = thb.TaskHandoffBot()
    ctx = zmq.Context.instance()
    peer = ctx.socket(zmq.PAIR)
    peer.connect(bot.addr)
    peer.send_json({"q": 1})
    bot.respond_to_queries(lambda m: {"a": m["q"]})
    assert peer.recv_json()["a"] == 1
    peer.close()
    bot.close()


def test_workflowdb_add_fetch(tmp_path):
    db = thb.WorkflowDB(tmp_path / "wf.db")
    rec = thb.WorkflowRecord(workflow=["a", "b"], title="All", description="d")
    db.add(rec)
    items = db.fetch()
    assert items and items[0].title == "All"
    assert items[0].estimated_profit_per_bot == 0.0


def test_store_plan(tmp_path):
    db = thb.WorkflowDB(tmp_path / "wf.db")
    bot = thb.TaskHandoffBot(workflow_db=db)
    tasks = [
        thb.TaskInfo(
            name=f"t{i}",
            dependencies=[],
            resources={},
            schedule="",
            code="",
            metadata={},
        )
        for i in range(5)
    ]
    ids = bot.store_plan(tasks, enhancements=["e1"], title="Model", description="plan")
    assert ids
    stored = db.fetch()
    assert any(len(item.workflow) <= 3 for item in stored)


def test_workflowdb_duplicate(tmp_path, caplog, monkeypatch):
    router = thb.init_db_router("wfdup", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    monkeypatch.setattr(thb.WorkflowDB, "add_embedding", lambda *a, **k: None)
    captured: dict[str, int | None] = {"id": None}
    orig = thb.insert_if_unique

    def wrapper(*args, **kwargs):
        res = orig(*args, **kwargs)
        captured["id"] = res
        return res

    monkeypatch.setattr(thb, "insert_if_unique", wrapper)

    db = thb.WorkflowDB(tmp_path / "wf.db", router=router)
    wf = thb.WorkflowRecord(workflow=["a"], title="T", description="d")
    first = db.add(wf)
    captured["id"] = None
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        second = db.add(thb.WorkflowRecord(workflow=["a"], title="T", description="d"))
    assert first == second
    assert captured["id"] == first
    assert "duplicate" in caplog.text.lower()
    assert db.conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 1


def test_workflowdb_content_hash_unique_index(tmp_path, monkeypatch):
    old = thb.GLOBAL_ROUTER
    router = thb.init_db_router("wfidx", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    monkeypatch.setattr(thb.WorkflowDB, "add_embedding", lambda *a, **k: None)
    db = thb.WorkflowDB(tmp_path / "wf.db", router=router)
    indexes = {
        row[1]: row[2]
        for row in db.conn.execute("PRAGMA index_list('workflows')").fetchall()
    }
    assert indexes.get("idx_workflows_content_hash") == 1
    thb.GLOBAL_ROUTER = old
