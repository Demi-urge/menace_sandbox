import importlib.util
import types
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck

pytest.importorskip("sqlalchemy")

# Load modules individually to avoid package side effects
ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg
sys.modules.setdefault("pandas", types.ModuleType("pandas"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"menace.{name}", ROOT / f"{name}.py", submodule_search_locations=[str(ROOT)]  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

mn = _load("databases")
dr = _load("db_router")
bd = _load("bot_database")


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(actions=st.lists(st.sampled_from(["insert", "delete"]), max_size=20))
def test_replication_checksums_consistency(tmp_path, actions):
    uid = uuid4().hex
    remote = f"sqlite:///{tmp_path / f'remote_{uid}.db'}"
    local_path = tmp_path / f'local_{uid}.db'
    router = dr.DBRouter(
        bot_db=bd.BotDB(local_path),
        remote_url=remote,
        auto_cross_link=False,
    )

    inserted: list[int] = []
    counter = 0
    for act in actions:
        if act == "insert":
            bid = router.insert_bot(bd.BotRecord(name=f"b{counter}"))
            inserted.append(bid)
            counter += 1
        elif inserted:
            bid = inserted.pop(0)
            router.delete_bot(bid)
        assert router.verify_replication()

    # Ensure final checksum state matches remote table
    mdb = mn.MenaceDB(url=remote)
    with mdb.engine.connect() as conn:
        rows = conn.execute(mdb.replication_checksums.select()).mappings().fetchall()
    remote_checksums = [(r["timestamp"], r["checksum"]) for r in rows]
    assert remote_checksums == router._local_checksums


