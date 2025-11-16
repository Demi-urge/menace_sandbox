import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
from neurosales.vector_db import VectorDB
sys.modules.setdefault("pinecone", MagicMock())


def test_vector_db_basic_operations():
    with patch("neurosales.vector_db.PineconeLogger") as Logger:
        logger = Logger.return_value
        db = VectorDB(
            max_messages=2,
            pinecone_index="idx",
            pinecone_key="k",
            pinecone_env="us-east",
            sync_interval=1,
        )
        db.add_message("user", "hello")
        db.add_message("assistant", "hi")
        recent = db.get_recent_messages()
        assert len(recent) == 2
        db.sync()
        assert logger.log.called


def test_vector_db_prune_decay():
    db = VectorDB(max_messages=5, ttl_seconds=1)
    db.add_message("user", "old")
    # force expiry
    db._messages[0].timestamp -= 2
    db.add_message("assistant", "new")
    assert len(db.get_recent_messages()) == 1


def test_vector_db_env_defaults(monkeypatch):
    monkeypatch.setenv("NEURO_PINECONE_INDEX", "idx")
    monkeypatch.setenv("NEURO_PINECONE_KEY", "k")
    monkeypatch.setenv("NEURO_PINECONE_ENV", "us")
    with patch("neurosales.vector_db.PineconeLogger") as Logger:
        VectorDB()
        Logger.assert_called_with(None, api_key=None, environment=None)


