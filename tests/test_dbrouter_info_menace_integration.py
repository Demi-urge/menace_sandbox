import pytest

pytest.importorskip("sqlalchemy")

import menace.research_aggregator_bot as rab
import menace.db_router as dr
import menace.menace as mn


def test_dbrouter_insert_info(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with mdb.engine.begin() as conn:
        conn.execute(mdb.models.insert().values(model_id=1, model_name="m"))
        conn.execute(mdb.workflows.insert().values(workflow_id=2, workflow_name="w"))
    info = rab.InfoDB(tmp_path / "i.db")
    router = dr.DBRouter(info_db=info, menace_db=mdb)
    router.info_db.set_current_model(1)
    item = rab.ResearchItem(topic="t", content="c", timestamp=0.0, model_id=1)
    router.insert_info(item, workflows=[2])
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.information.select()).fetchone() is not None
        assert conn.execute(mdb.information_models.select()).fetchone() is not None
        assert conn.execute(mdb.information_workflows.select()).fetchone() is not None


def test_dbrouter_update_and_delete_info(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    info = rab.InfoDB(tmp_path / "i.db")
    router = dr.DBRouter(info_db=info, menace_db=mdb)
    item = rab.ResearchItem(topic="t", content="c", summary="s", timestamp=0.0)
    info_id = router.insert_info(item)

    router.update_info(info_id, summary="new")
    import sqlite3
    with sqlite3.connect(info.path) as conn:
        row = conn.execute("SELECT summary FROM info WHERE id=?", (info_id,)).fetchone()
    assert row and row[0] == "new"
    with mdb.engine.connect() as conn:
        row = conn.execute(mdb.information.select()).mappings().fetchone()
    assert row["summary"] == "new"

    router.delete_info(info_id)
    with sqlite3.connect(info.path) as conn:
        assert conn.execute("SELECT * FROM info").fetchone() is None
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.information.select()).fetchone() is None



