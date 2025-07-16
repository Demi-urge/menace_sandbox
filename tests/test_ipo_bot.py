import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import sqlite3
from pathlib import Path

import menace.ipo_bot as ipb


BLUEPRINT = "BotA collects data using BotB. BotC processes results after BotB."


def make_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE bots (id INTEGER PRIMARY KEY, name TEXT, keywords TEXT, reuse INTEGER)")
    cur.execute("INSERT INTO bots (name, keywords, reuse) VALUES ('BotB', 'collect data scrape web', 1)")
    cur.execute("INSERT INTO bots (name, keywords, reuse) VALUES ('BotC', 'process results', 1)")
    conn.commit()
    conn.close()
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
