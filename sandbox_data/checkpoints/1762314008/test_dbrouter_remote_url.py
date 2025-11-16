import pytest

pytest.importorskip("sqlalchemy")

import importlib.util
import types
import sys
from pathlib import Path

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


def test_remote_url_mirrors_inserts(tmp_path):
    remote = f"sqlite:///{tmp_path / 'remote.db'}"
    router = dr.DBRouter(
        bot_db=bd.BotDB(tmp_path / 'local.db'),
        remote_url=remote,
        auto_cross_link=False,
    )

    bid = router.insert_bot(bd.BotRecord(name="bot"))

    # Verify local insert
    row = router.bot_db.conn.execute("SELECT name FROM bots WHERE id=?", (bid,)).fetchone()
    assert row and row[0] == "bot"

    # Verify remote replication by inspecting the database at *remote*
    mdb = mn.MenaceDB(url=remote)
    with mdb.engine.connect() as conn:
        mrow = conn.execute(mdb.bots.select()).mappings().fetchone()
    assert mrow["bot_id"] == bid and mrow["bot_name"] == "bot"
