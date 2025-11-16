import sqlite3

from db_router import DBRouter, SHARED_TABLES, LOCAL_TABLES
from menace.information_db import InformationDB, InformationRecord


def _db_path(conn):
    return conn.execute("PRAGMA database_list").fetchall()[0][2]


def test_informationdb_uses_shared_db(tmp_path):
    shared = tmp_path / "shared.db"
    local = tmp_path / "local.db"
    router = DBRouter("info", str(local), str(shared))
    try:
        db = InformationDB(router=router, vector_index_path=str(tmp_path / "idx"))
        assert _db_path(db.conn) == str(shared)
        db.add(InformationRecord(data_type="news"))
        with sqlite3.connect(shared) as conn:
            assert conn.execute("SELECT count(*) FROM information").fetchone()[0] == 1
    finally:
        router.close()


def test_informationdb_uses_local_db_when_configured(tmp_path):
    shared = tmp_path / "shared.db"
    local = tmp_path / "local.db"
    SHARED_TABLES.discard("information")
    LOCAL_TABLES.add("information")
    router = DBRouter("info", str(local), str(shared))
    try:
        db = InformationDB(router=router, vector_index_path=str(tmp_path / "idx"))
        assert _db_path(db.conn) == str(local)
        db.add(InformationRecord(data_type="news"))
        with sqlite3.connect(local) as conn:
            assert conn.execute("SELECT count(*) FROM information").fetchone()[0] == 1
    finally:
        router.close()
        LOCAL_TABLES.discard("information")
        SHARED_TABLES.add("information")
