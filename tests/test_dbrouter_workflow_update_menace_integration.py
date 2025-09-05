import pytest

pytest.importorskip("sqlalchemy")

import importlib.util
import types
import sys
from pathlib import Path

from dynamic_path_router import resolve_path

# Load modules without running package __init__
ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

spec = importlib.util.spec_from_file_location(
    "menace.databases",
    resolve_path("databases.py"),  # path-ignore
    submodule_search_locations=[str(ROOT)],
)
mn = importlib.util.module_from_spec(spec)
sys.modules["menace.databases"] = mn
spec.loader.exec_module(mn)

spec = importlib.util.spec_from_file_location(
    "menace.db_router",
    resolve_path("db_router.py"),  # path-ignore
    submodule_search_locations=[str(ROOT)],
)
dr = importlib.util.module_from_spec(spec)
sys.modules["menace.db_router"] = dr
spec.loader.exec_module(dr)


class DummyWF:
    def __init__(self):
        self.calls = []

    def update(self, wid: int, **fields):
        self.calls.append((wid, fields))


def test_update_workflow_mirrors_to_menace(tmp_path):
    mdb = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with mdb.engine.begin() as conn:
        conn.execute(
            mdb.workflows.insert().values(workflow_id=1, workflow_name="w", status="old")
        )
    wf = DummyWF()
    router = dr.DBRouter(workflow_db=wf, menace_db=mdb)
    router.update_workflow(1, status="new")
    assert wf.calls and wf.calls[0][0] == 1
    with mdb.engine.connect() as conn:
        row = conn.execute(mdb.workflows.select()).mappings().fetchone()
    assert row["status"] == "new"


