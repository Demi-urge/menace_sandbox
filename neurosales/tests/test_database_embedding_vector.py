import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch
from neurosales.embedding_memory import DatabaseEmbeddingMemory
from neurosales.vector_db import DatabaseVectorDB
from neurosales.sql_db import create_session as create_sql_session


def test_database_embedding_memory_persistence():
    Session = create_sql_session("sqlite://")
    mem = DatabaseEmbeddingMemory(session_factory=Session, max_messages=2)
    mem.add_message("user", "hello")
    mem2 = DatabaseEmbeddingMemory(session_factory=Session, max_messages=2)
    msgs = mem2.get_recent_messages()
    assert [m.content for m in msgs] == ["hello"]


def test_database_vector_db_persistence():
    Session = create_sql_session("sqlite://")
    with patch("neurosales.vector_db.PineconeLogger"):
        db = DatabaseVectorDB(session_factory=Session, max_messages=2, sync_interval=1)
        db.add_message("user", "hi")
        db2 = DatabaseVectorDB(session_factory=Session, max_messages=2, sync_interval=1)
        msgs = db2.get_recent_messages()
        assert [m.content for m in msgs] == ["hi"]
