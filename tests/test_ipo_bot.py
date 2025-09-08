import pytest

pytest.skip("optional dependencies not installed", allow_module_level=True)

from pathlib import Path  # noqa: E402
from db_router import GLOBAL_ROUTER, init_db_router  # noqa: E402
import menace.ipo_bot as ipb  # noqa: E402


BLUEPRINT = "BotA collects data using BotB. BotC processes results after BotB."


def make_db(path: Path) -> Path:
    init_db_router(
        "ipo_bot_test", local_db_path=str(path.parent / "local.db"), shared_db_path=str(path)
    )
    conn = GLOBAL_ROUTER.get_connection("bots")
    cur = conn.cursor()
    cur.execute(
        (
            "CREATE TABLE IF NOT EXISTS bots ("
            "id INTEGER PRIMARY KEY, name TEXT, keywords TEXT, reuse INTEGER, "
            "source_menace_id TEXT NOT NULL)"
        )
    )
    cur.execute(
        (
            "INSERT INTO bots (name, keywords, reuse, source_menace_id) "
            "VALUES ('BotB', 'collect data scrape web', 1, ?)"
        ),
        (GLOBAL_ROUTER.menace_id,),
    )
    cur.execute(
        (
            "INSERT INTO bots (name, keywords, reuse, source_menace_id) "
            "VALUES ('BotC', 'process results', 1, ?)"
        ),
        (GLOBAL_ROUTER.menace_id,),
    )
    conn.commit()
    return path


def test_ingest():
    ing = ipb.BlueprintIngestor()
    bp = ing.ingest(BLUEPRINT)
    assert [t.name for t in bp.tasks] == ["BotA", "BotB", "BotC"]


def test_generate_plan(tmp_path: Path):
    db = make_db(tmp_path / "models.db")
    bot = ipb.IPOBot(db_path=str(db), enhancements_db=tmp_path / "enh.db")
    plan = bot.generate_plan(BLUEPRINT, "bp1")
    assert len(plan.actions) == 3
    assert plan.graph.number_of_nodes() == 3


def test_search_scope(tmp_path: Path):
    db = make_db(tmp_path / "models.db")
    searcher = ipb.BotDatabaseSearcher(str(db))
    results = searcher.search(["BotB"], "local")
    assert results and results[0].name == "BotB"
