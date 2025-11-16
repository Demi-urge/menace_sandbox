import sqlite3
from pathlib import Path

import menace.competitive_intelligence_bot as cib


def _create_v1_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            source TEXT,
            timestamp TEXT,
            sentiment REAL,
            entities TEXT,
            ai_signals INTEGER
        )
        """
    )
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()


def test_migrate_v1_to_v2(tmp_path: Path) -> None:
    db_path = tmp_path / "int.db"
    _create_v1_db(db_path)
    db = cib.IntelligenceDB(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            cols = [
                r[1]
                for r in conn.execute("PRAGMA table_info(updates)").fetchall()
            ]
    finally:
        db.close_all()
    assert version == cib.IntelligenceDB.SCHEMA_VERSION
    assert "category" in cols


def test_add_and_fetch_category(tmp_path: Path) -> None:
    db = cib.IntelligenceDB(tmp_path / "int.db")
    up = cib.CompetitorUpdate(
        title="t", content="c", source="s", timestamp="t", category="news"
    )
    db.add(up)
    res = db.fetch()
    db.close_all()
    assert res and res[0].category == "news"
