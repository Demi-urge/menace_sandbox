import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("networkx")

from menace.bot_registry import BotRegistry
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.databases import MenaceDB
from menace.db_router import DBRouter


def _setup_menace(tmp_path):
    db = MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with db.engine.begin() as conn:
        conn.execute(db.bots.insert().values(bot_id=1, bot_name='A'))
        conn.execute(db.bots.insert().values(bot_id=2, bot_name='B'))
        conn.execute(db.workflows.insert().values(workflow_id=1, workflow_name='wfA'))
        conn.execute(db.workflows.insert().values(workflow_id=2, workflow_name='wfB'))
        conn.execute(db.workflow_bots.insert().values(workflow_id=1, bot_id=1))
        conn.execute(db.workflow_bots.insert().values(workflow_id=2, bot_id=2))
    return db


def test_router_related_workflows(tmp_path):
    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    reg.register_interaction('A', 'B')
    pdb = PathwayDB(tmp_path / 'p.db')
    pdb.log(PathwayRecord(actions='wfB', inputs='', outputs='', exec_time=1.0, resources='', outcome=Outcome.SUCCESS, roi=1.0))
    router = DBRouter(menace_db=db)

    results = router.related_workflows('A', registry=reg, pathway_db=pdb)
    assert 'wfA' in results
    assert 'wfB' in results
