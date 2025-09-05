import os
import shutil

import pytest

import menace.deployment_bot as db
import menace.chatgpt_enhancement_bot as ceb
import menace.contrarian_db as cdb
import menace.menace as mn

pytest.importorskip("sqlalchemy")


def test_links_and_status(tmp_path):
    p = tmp_path / "b.py"  # path-ignore
    p.write_text("def run():\n    pass\n")
    shutil.copy(p, "b.py")  # path-ignore

    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    eid = enh_db.add(ceb.Enhancement(idea="i", rationale="r"))

    menace = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with menace.engine.begin() as conn:
        conn.execute(
            menace.models.insert().values(
                model_id=1,
                model_name="m",
                current_status="inactive",
            )
        )

    contr_db = cdb.ContrarianDB(tmp_path / "c.db")
    cid = contr_db.add(cdb.ContrarianRecord(innovation_name="x", innovation_type="y"))

    bot = db.DeploymentBot(
        db.DeploymentDB(tmp_path / "dep.db"),
        bot_db=db.BotDB(tmp_path / "bots.db"),
        workflow_db=db.WorkflowDB(tmp_path / "wf.db"),
        enh_db=enh_db,
        code_db=db.CodeDB(tmp_path / "code.db"),
        menace_db=menace,
        contrarian_db=contr_db,
    )
    spec = db.DeploymentSpec(name="test", resources={}, env={})
    bot.deploy("run", ["b"], spec, model_id=1, enhancements=[eid], contrarian_id=cid)
    os.remove("b.py")  # path-ignore

    assert enh_db.bots_for(eid)
    with menace.engine.connect() as conn:
        row = conn.execute(menace.models.select()).mappings().fetchone()
    assert row["current_status"] == "active"
    assert contr_db.fetch()[0].status == "active"


def test_contrarian_timestamp_update(tmp_path):
    p = tmp_path / "c.py"  # path-ignore
    p.write_text("def run():\n    pass\n")
    shutil.copy(p, "c.py")  # path-ignore

    contr_db = cdb.ContrarianDB(tmp_path / "c.db")
    rec = cdb.ContrarianRecord(
        innovation_name="x",
        innovation_type="y",
        timestamp_last_evaluated="old",
    )
    cid = contr_db.add(rec)

    bot = db.DeploymentBot(
        db.DeploymentDB(tmp_path / "dep.db"),
        workflow_db=db.WorkflowDB(tmp_path / "wf.db"),
        code_db=db.CodeDB(tmp_path / "code.db"),
        contrarian_db=contr_db,
    )
    spec = db.DeploymentSpec(name="test", resources={}, env={})
    bot.deploy("run", ["c"], spec, contrarian_id=cid)
    os.remove("c.py")  # path-ignore

    assert contr_db.get(cid).timestamp_last_evaluated != "old"


def test_contrarian_new_strategy_links(tmp_path):
    p = tmp_path / "d.py"  # path-ignore
    p.write_text("def run():\n    pass\n")
    shutil.copy(p, "d.py")  # path-ignore

    contr_db = cdb.ContrarianDB(tmp_path / "c.db")
    cid = contr_db.add(cdb.ContrarianRecord(innovation_name="n", innovation_type="y"))

    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    eid = enh_db.add(ceb.Enhancement(idea="i", rationale="r"))

    err_db = db.ErrorDB(tmp_path / "err.db")
    menace_id = (
        err_db.router.menace_id if getattr(err_db, "router", None) else os.getenv("MENACE_ID", "")
    )
    cur = err_db.conn.execute(
        "INSERT INTO discrepancies(message, ts, source_menace_id) VALUES (?, ?, ?)",
        ("err", "2020", menace_id),
    )
    err_id = cur.lastrowid
    err_db.conn.commit()

    wf_db = db.WorkflowDB(tmp_path / "wf.db")
    wid = wf_db.add(db.WorkflowRecord(workflow=["step"]))

    bot = db.DeploymentBot(
        db.DeploymentDB(tmp_path / "dep.db"),
        bot_db=db.BotDB(tmp_path / "bots.db"),
        workflow_db=wf_db,
        enh_db=enh_db,
        code_db=db.CodeDB(tmp_path / "code.db"),
        error_db=err_db,
        contrarian_db=contr_db,
    )
    spec = db.DeploymentSpec(name="test", resources={}, env={})
    bot.deploy(
        "run",
        ["d"],
        spec,
        workflows=[wid],
        enhancements=[eid],
        contrarian_id=cid,
        errors=[err_id],
    )
    os.remove("d.py")  # path-ignore

    assert contr_db.workflows_for(cid)
    assert contr_db.enhancements_for(cid) == [eid]
    assert contr_db.errors_for(cid) == [err_id]
    assert wf_db.fetch()[0].status == "active"
    assert contr_db.fetch()[0].status == "active"


def test_workflow_status_failure(tmp_path, monkeypatch):
    p = tmp_path / "f.py"  # path-ignore
    p.write_text("def run():\n    pass\n")
    shutil.copy(p, "f.py")  # path-ignore

    wf_db = db.WorkflowDB(tmp_path / "wf.db")
    wid = wf_db.add(db.WorkflowRecord(workflow=["step"]))

    bot = db.DeploymentBot(
        db.DeploymentDB(tmp_path / "dep.db"),
        workflow_db=wf_db,
    )
    spec = db.DeploymentSpec(name="test", resources={}, env={})

    monkeypatch.setattr(bot, "run_tests", lambda **k: (False, []))

    bot.deploy("run", ["f"], spec, workflows=[wid])
    os.remove("f.py")  # path-ignore

    assert wf_db.fetch()[0].status == "failed"


def test_update_history(tmp_path):
    db_path = tmp_path / "dep.db"
    ddb = db.DeploymentDB(db_path)
    ddb.add_update(["pkg"], "success")
    row = ddb.conn.execute("SELECT packages, status FROM update_history").fetchone()
    assert row == ("pkg", "success")
