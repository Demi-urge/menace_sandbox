import pytest

pytest.importorskip("sqlalchemy")

import menace.research_aggregator_bot as rab
import menace.menace as mn


def test_infodb_menace_insert(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with mdb.engine.begin() as conn:
        conn.execute(mdb.models.insert().values(model_id=1, model_name="m"))
        conn.execute(mdb.workflows.insert().values(workflow_id=2, workflow_name="w"))
    db = rab.InfoDB(tmp_path / "i.db", menace_db=mdb)
    db.set_current_model(1)
    item = rab.ResearchItem(topic="t", content="c", timestamp=0.0, model_id=1)
    db.add(item, workflows=[2])
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.information.select()).fetchone() is not None
        assert conn.execute(mdb.information_models.select()).fetchone() is not None
        assert conn.execute(mdb.information_workflows.select()).fetchone() is not None


def test_infodb_invalid_refs_warn(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    db = rab.InfoDB(tmp_path / "i.db", menace_db=mdb)
    item = rab.ResearchItem(topic="t", content="c", timestamp=0.0, model_id=1)
    with pytest.warns(UserWarning):
        db.add(item, workflows=[99])
