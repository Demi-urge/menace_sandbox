import importlib
import sqlite3
from pathlib import Path

import db_router
import embedding_stats_db


def test_embedding_stats_via_router(tmp_path):
    local = tmp_path / "l.db"
    shared = tmp_path / "s.db"
    db_router.init_db_router("emb", local_db_path=str(local), shared_db_path=str(shared))
    importlib.reload(embedding_stats_db)
    stats = embedding_stats_db.EmbeddingStatsDB()
    stats.log("db", 10, 1.0, 2.0)
    conn = db_router.GLOBAL_ROUTER.get_connection("embedding_stats")
    row = conn.execute(
        "SELECT db_name, tokens FROM embedding_stats"
    ).fetchone()
    assert row == ("db", 10)
    shared_conn = db_router.GLOBAL_ROUTER.get_connection("information")
    db_path = Path(
        shared_conn.execute("PRAGMA database_list").fetchone()[2]
    )
    assert db_path == shared


def test_embedding_stats_no_direct_sqlite(tmp_path, monkeypatch):
    local = tmp_path / "l2.db"
    shared = tmp_path / "s2.db"
    db_router.init_db_router("emb2", local_db_path=str(local), shared_db_path=str(shared))
    importlib.reload(embedding_stats_db)

    calls: list[object] = []

    def bad_connect(*a, **k):  # pragma: no cover - should not run
        calls.append(1)
        raise AssertionError("sqlite3.connect should not be called directly")

    monkeypatch.setattr(sqlite3, "connect", bad_connect)

    stats = embedding_stats_db.EmbeddingStatsDB()
    stats.log("db", 1, 1.0, 1.0)

    assert not calls
