import time
import pytest

pytest.importorskip("sqlalchemy")

from menace.bot_database import BotDB, BotRecord
from menace.databases import MenaceDB
from menace.bot_registry import BotRegistry
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.db_router import DBRouter


def _setup_menace(tmp_path):
    db = MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with db.engine.begin() as conn:
        conn.execute(db.bots.insert().values(bot_id=1, bot_name='A'))
        conn.execute(db.workflows.insert().values(workflow_id=1, workflow_name='wfA'))
        conn.execute(db.workflow_bots.insert().values(workflow_id=1, bot_id=1))
    return db


def test_execute_query_caching_and_flush(tmp_path):
    bdb = BotDB(tmp_path / 'b.db')
    router = DBRouter(bot_db=bdb, cache_seconds=1.0)

    bdb.add_bot(BotRecord(name='x'))
    res1 = router.execute_query('bot', 'SELECT name FROM bots')

    bdb.add_bot(BotRecord(name='y'))
    res2 = router.execute_query('bot', 'SELECT name FROM bots')
    assert res1 == res2

    router.flush_cache()
    res3 = router.execute_query('bot', 'SELECT name FROM bots')
    assert len(res3) == 2


def test_cross_query_cache_expiration(tmp_path):
    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    pdb = PathwayDB(tmp_path / 'p.db')
    pdb.log(PathwayRecord(actions='wfA', inputs='', outputs='', exec_time=1.0, resources='', outcome=Outcome.SUCCESS, roi=1.0))

    router = DBRouter(menace_db=db, cache_seconds=0.2)

    res1 = router.related_workflows('A', registry=reg, pathway_db=pdb)

    with db.engine.begin() as conn:
        conn.execute(db.workflows.insert().values(workflow_id=2, workflow_name='wfB'))
        conn.execute(db.workflow_bots.insert().values(workflow_id=2, bot_id=1))

    res2 = router.related_workflows('A', registry=reg, pathway_db=pdb)
    assert res1 == res2

    time.sleep(0.21)
    res3 = router.related_workflows('A', registry=reg, pathway_db=pdb)
    assert 'wfB' in res3
