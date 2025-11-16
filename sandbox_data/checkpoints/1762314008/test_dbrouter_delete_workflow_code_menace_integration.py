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
    "menace.code_database", ROOT / "code_database.py", submodule_search_locations=[str(ROOT)]  # path-ignore
)
cdm = importlib.util.module_from_spec(spec)
sys.modules["menace.code_database"] = cdm
spec.loader.exec_module(cdm)


def test_delete_workflow_and_code_mirrors_to_menace(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with mdb.engine.begin() as conn:
        conn.execute(mdb.workflows.insert().values(workflow_id=1, workflow_name="w"))
        conn.execute(
            mdb.code.insert().values(code_id=1, template_type="t", language="py", version="1", complexity_score=0.0, code_summary="s")
        )
    wfdb = thb.WorkflowDB(tmp_path / "wf.db")
    wf_rec = thb.WorkflowRecord(workflow=["a"], title="t")
    router = dr.DBRouter(workflow_db=wfdb, menace_db=mdb, code_db=cdm.CodeDB(tmp_path/"c.db"))
    cid = router.insert_code(cdm.CodeRecord(code="print('x')", summary="s"))
    wid = router.insert_workflow(wf_rec)

    router.delete_workflow(wid)
    with sqlite3.connect(wfdb.path) as conn:
        assert conn.execute("SELECT * FROM workflows").fetchone() is None
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.workflows.select()).fetchone() is None

    router.delete_code(cid)
    with sqlite3.connect(router.code_db.path) as conn:
        assert (
            conn.execute(
                "SELECT * FROM code WHERE source_menace_id=?",
                (router.menace_id,),
            ).fetchone()
            is None
        )
    with mdb.engine.connect() as conn:
        assert conn.execute(mdb.code.select()).fetchone() is None
