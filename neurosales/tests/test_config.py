import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import neurosales.config as config


def test_load_config_parses_env(monkeypatch):
    monkeypatch.setenv("NEURO_OPENAI_KEY", "ok")
    monkeypatch.setenv("NEURO_PINECONE_INDEX", "idx")
    monkeypatch.setenv("NEURO_PINECONE_KEY", "key")
    monkeypatch.setenv("NEURO_PINECONE_ENV", "env")
    monkeypatch.setenv("NEURO_NEO4J_URI", "bolt://x")
    monkeypatch.setenv("NEURO_NEO4J_USER", "u")
    monkeypatch.setenv("NEURO_NEO4J_PASS", "p")
    monkeypatch.setenv("NEURO_REDIS_URL", "redis://x")
    monkeypatch.setenv("NEURO_MEMCACHED_SERVERS", "a,b")
    monkeypatch.setenv("NEURO_PROXY_LIST", "http://a,http://b")

    cfg = config.load_config()
    assert cfg.openai_key == "ok"
    assert cfg.pinecone_index == "idx"
    assert cfg.pinecone_key == "key"
    assert cfg.pinecone_env == "env"
    assert cfg.neo4j_uri == "bolt://x"
    assert cfg.neo4j_user == "u"
    assert cfg.neo4j_pass == "p"
    assert cfg.redis_url == "redis://x"
    assert cfg.memcached_servers == ["a", "b"]
    assert cfg.proxy_list == ["http://a", "http://b"]


def test_is_openai_enabled_warns(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.delenv("NEURO_OPENAI_KEY", raising=False)
    assert not config.is_openai_enabled()
    assert "openai" in caplog.text.lower()
