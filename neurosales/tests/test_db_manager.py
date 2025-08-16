import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.db_manager import DatabaseConnectionManager
from unittest.mock import patch, MagicMock
from neurosales.neuro_etl import NeuroToken


def test_default_connections():
    mgr = DatabaseConnectionManager()
    pg = mgr.get_postgres()
    mongo = mgr.get_mongo()
    vect = mgr.get_vector_db()
    neo = mgr.get_neo4j()
    assert pg is not None and mongo is not None
    assert vect.get_recent_messages() == []
    token = NeuroToken("s1", "hello", "", "", 0.0)
    pg.insert(token)
    mongo.insert("s1", "paragraph")
    vect.add_message("user", "hi")
    assert pg.rows and mongo.docs
    assert vect.get_recent_messages()
    assert neo is not None


def test_load_balancing_toggle():
    cfg = {"postgres_urls": ["memory", "memory"], "enable_load_balancing": False}
    mgr = DatabaseConnectionManager(cfg)
    c1 = mgr.get_postgres("u1")
    c2 = mgr.get_postgres("u2")
    assert c1 is c2
    mgr_lb = DatabaseConnectionManager(
        {"postgres_urls": ["memory", "memory"], "enable_load_balancing": True}
    )
    a = mgr_lb.get_postgres("u1")
    b = mgr_lb.get_postgres("u2")
    assert a is not b or len(mgr_lb.pg_conns) == 1


def test_preference_engine_in_manager(tmp_path):
    url = f"sqlite:///{tmp_path/'prefs.db'}"
    mgr = DatabaseConnectionManager({"preference_db_url": url})
    engine = mgr.get_preference_engine()
    engine.add_message("u1", "hi there")
    mgr2 = DatabaseConnectionManager({"preference_db_url": url})
    engine2 = mgr2.get_preference_engine()
    prof = engine2.get_profile("u1")
    assert prof.keyword_freq.get("hi", 0) > 0


def test_persistent_memories_in_manager(tmp_path):
    emb_url = f"sqlite:///{tmp_path/'emb.db'}"
    vect_url = f"sqlite:///{tmp_path/'vect.db'}"
    mgr = DatabaseConnectionManager(
        {"embedding_db_url": emb_url, "vector_db_url": vect_url}
    )
    emb = mgr.get_embedding_memory()
    vect = mgr.get_vector_db()
    emb.add_message("user", "hi")
    vect.add_message("user", "hello")
    mgr2 = DatabaseConnectionManager(
        {"embedding_db_url": emb_url, "vector_db_url": vect_url}
    )
    assert mgr2.get_embedding_memory().get_recent_messages()[0].content == "hi"
    assert mgr2.get_vector_db().get_recent_messages()[0].content == "hello"


def test_db_manager_vector_env(monkeypatch):
    monkeypatch.setenv("NEURO_PINECONE_INDEX", "idx")
    monkeypatch.setenv("NEURO_PINECONE_KEY", "k")
    monkeypatch.setenv("NEURO_PINECONE_ENV", "env")
    with patch("neurosales.vector_db.PineconeLogger") as Logger:
        mgr = DatabaseConnectionManager()
        mgr.get_vector_db()
    Logger.assert_called_with(None, api_key=None, environment=None)


def test_db_manager_neo4j_env(monkeypatch):
    monkeypatch.setenv("NEURO_NEO4J_URI", "bolt://env")
    monkeypatch.setenv("NEURO_NEO4J_USER", "u")
    monkeypatch.setenv("NEURO_NEO4J_PASS", "p")
    updater = MagicMock()
    with patch("neurosales.db_manager.InfluenceGraphUpdater", return_value=updater) as Upd:
        mgr = DatabaseConnectionManager()
    Upd.assert_called_with("bolt://env", ("u", "p"))
    assert mgr.get_neo4j() is updater


def test_manager_conn_lists_from_env(monkeypatch):
    monkeypatch.setenv("NEURO_POSTGRES_URLS", "memory,memory")
    monkeypatch.setenv("NEURO_MONGO_URLS", "memory,memory")
    mgr = DatabaseConnectionManager()
    assert len(mgr.pg_conns) == 2
    assert len(mgr.mongo_conns) == 2


def test_enable_load_balancing_env(monkeypatch):
    monkeypatch.setenv("NEURO_ENABLE_DB_LOAD_BALANCING", "1")
    monkeypatch.setenv("NEURO_POSTGRES_URLS", "memory,memory")
    mgr = DatabaseConnectionManager()
    assert mgr.enable_load_balancing is True
    a = mgr.get_postgres("u1")
    b = mgr.get_postgres("u2")
    assert a is not b or len(mgr.pg_conns) == 1


def test_manager_pooling_config(monkeypatch):
    called = {}

    def fake_create_session(url, **kw):
        called.update(kw)

        class F:
            def __call__(self):
                return "sess"

        return F

    monkeypatch.setattr("neurosales.db_manager.psycopg2", object())
    monkeypatch.setattr("neurosales.db_manager.create_session", fake_create_session)
    cfg = {
        "postgres_urls": ["postgresql://db"],
        "pool_size": 4,
        "max_overflow": 8,
    }
    DatabaseConnectionManager(cfg)
    assert called["pool_size"] == 4
    assert called["max_overflow"] == 8


def test_manager_pooling_env(monkeypatch):
    called = {}

    def fake_create_session(url, **kw):
        called.update(kw)

        class F:
            def __call__(self):
                return "sess"

        return F

    monkeypatch.setattr("neurosales.db_manager.psycopg2", object())
    monkeypatch.setattr("neurosales.db_manager.create_session", fake_create_session)
    monkeypatch.setenv("NEURO_POSTGRES_URLS", "postgresql://db")
    monkeypatch.setenv("NEURO_DB_POOL_SIZE", "3")
    monkeypatch.setenv("NEURO_DB_MAX_OVERFLOW", "6")
    DatabaseConnectionManager()
    assert called["pool_size"] == 3
    assert called["max_overflow"] == 6
