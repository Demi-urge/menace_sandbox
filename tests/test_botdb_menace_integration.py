import pytest

pytest.importorskip("sqlalchemy")

import menace.bot_database as bdbm
import menace.menace as mn


def test_botdb_menace_insert(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with mdb.engine.begin() as conn:
        conn.execute(mdb.models.insert().values(model_id=1, model_name="m"))
        conn.execute(mdb.workflows.insert().values(workflow_id=2, workflow_name="w"))
        conn.execute(
            mdb.enhancements.insert().values(
                enhancement_id=3,
                description_of_change="d",
                source_menace_id="",
            )
        )
    db = bdbm.BotDB(tmp_path / "b.db", menace_db=mdb)
    rec = bdbm.BotRecord(name="b", bid=1)
    db.add_bot(rec, models=[1], workflows=[2], enhancements=[3])
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.bots.select()).fetchone() is not None
        assert conn.execute(mdb.bot_models.select()).fetchone() is not None
        assert conn.execute(mdb.bot_workflows.select()).fetchone() is not None
        assert conn.execute(mdb.bot_enhancements.select()).fetchone() is not None


def test_botdb_invalid_model_warn(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    db = bdbm.BotDB(tmp_path / "b.db", menace_db=mdb)
    rec = bdbm.BotRecord(name="b", bid=1)
    with pytest.warns(UserWarning):
        db.add_bot(rec, models=[1])
