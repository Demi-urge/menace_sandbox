import pytest

pytest.importorskip("sqlalchemy")

import importlib.util
import types
import sys
from pathlib import Path
import sqlite3

# Load modules without running package __init__
ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))

spec = importlib.util.spec_from_file_location(
    "menace.databases", ROOT / "databases.py", submodule_search_locations=[str(ROOT)]  # path-ignore
)
mn = importlib.util.module_from_spec(spec)
sys.modules["menace.databases"] = mn
spec.loader.exec_module(mn)

spec = importlib.util.spec_from_file_location(
    "menace.db_router", ROOT / "db_router.py", submodule_search_locations=[str(ROOT)]  # path-ignore
)
dr = importlib.util.module_from_spec(spec)
sys.modules["menace.db_router"] = dr
spec.loader.exec_module(dr)

spec = importlib.util.spec_from_file_location(
    "menace.task_handoff_bot", ROOT / "task_handoff_bot.py", submodule_search_locations=[str(ROOT)]  # path-ignore
)
thb = importlib.util.module_from_spec(spec)
sys.modules["menace.task_handoff_bot"] = thb
spec.loader.exec_module(thb)

spec = importlib.util.spec_from_file_location(
    "menace.bot_database", ROOT / "bot_database.py", submodule_search_locations=[str(ROOT)]  # path-ignore
)
bd = importlib.util.module_from_spec(spec)
sys.modules["menace.bot_database"] = bd
spec.loader.exec_module(bd)

spec = importlib.util.spec_from_file_location(
    "menace.research_aggregator_bot", ROOT / "research_aggregator_bot.py", submodule_search_locations=[str(ROOT)]  # path-ignore
)
rab = importlib.util.module_from_spec(spec)
sys.modules["menace.research_aggregator_bot"] = rab
spec.loader.exec_module(rab)


class FailingTM(dr.TransactionManager):
    def run(self, operation, rollback):
        rollback()
        raise RuntimeError("fail")


def test_insert_workflow_rollback(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    wfdb = thb.WorkflowDB(tmp_path / 'wf.db')
    router = dr.DBRouter(workflow_db=wfdb, menace_db=mdb, transaction_manager=FailingTM())
    wf = thb.WorkflowRecord(workflow=["a"], title="t")
    with pytest.raises(RuntimeError):
        router.insert_workflow(wf)
    with sqlite3.connect(wfdb.path) as conn:
        assert conn.execute("SELECT * FROM workflows").fetchone() is None
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.workflows.select()).fetchone() is None


def test_update_bot_rollback(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    bdb = bd.BotDB(tmp_path / 'b.db')
    router = dr.DBRouter(bot_db=bdb, menace_db=mdb)
    bid = router.insert_bot(bd.BotRecord(name="b"))
    router.transaction_manager = FailingTM()
    with pytest.raises(RuntimeError):
        router.update_bot(bid, name="B")
    row = bdb.conn.execute("SELECT name FROM bots WHERE id=?", (bid,)).fetchone()
    assert row and row[0] == "b"
    with mdb.engine.connect() as conn:
        row = conn.execute(mdb.bots.select()).mappings().fetchone()
    assert row["bot_name"] == "b"


def test_delete_info_rollback(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    info = rab.InfoDB(tmp_path / 'i.db')
    router = dr.DBRouter(info_db=info, menace_db=mdb)
    item = rab.ResearchItem(topic="t", content="c", timestamp=0.0)
    info_id = router.insert_info(item)
    router.transaction_manager = FailingTM()
    with pytest.raises(RuntimeError):
        router.delete_info(info_id)
    with sqlite3.connect(info.path) as conn:
        assert conn.execute("SELECT * FROM info").fetchone() is not None
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.information.select()).fetchone() is not None
