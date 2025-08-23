import importlib
import sqlite3

import db_router


def test_retrieval_cache_initialises_router(tmp_path, monkeypatch):
    importlib.reload(db_router)
    calls = {}

    def fake_init(menace_id, local_db_path=None, shared_db_path=None):
        calls['id'] = menace_id
        db_router.GLOBAL_ROUTER = db_router.DBRouter(
            menace_id, str(tmp_path / 'l.db'), str(tmp_path / 's.db')
        )
        return db_router.GLOBAL_ROUTER

    monkeypatch.setattr(db_router, 'init_db_router', fake_init)
    rc_mod = importlib.reload(importlib.import_module('retrieval_cache'))

    calls2 = {}

    def fake_get_connection(table, operation='read'):
        assert db_router.GLOBAL_ROUTER is not None
        calls2['table'] = table
        return sqlite3.connect(':memory:')

    monkeypatch.setattr(rc_mod.DB_ROUTER, 'get_connection', fake_get_connection)
    rc_mod.RetrievalCache(ttl=1)
    assert calls['id'] == 'retrieval_cache'
    assert calls2['table'] == 'retrieval_cache'
