import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.memory import DatabaseConversationMemory
from neurosales.sql_db import create_session as create_sql_session


def test_persistent_history():
    Session = create_sql_session("sqlite://")
    mem = DatabaseConversationMemory(user_id="u1", session_factory=Session, max_messages=3)
    mem.add_message("user", "hi")
    mem.add_message("assistant", "hello")

    mem2 = DatabaseConversationMemory(user_id="u1", session_factory=Session, max_messages=3)
    msgs = mem2.get_recent_messages()
    assert [m.content for m in msgs] == ["hi", "hello"]


def test_memory_db_url(tmp_path):
    db_file = tmp_path / "conv.db"
    url = f"sqlite:///{db_file}"
    mem = DatabaseConversationMemory(user_id="u1", db_url=url, max_messages=2)
    mem.add_message("user", "hi")
    mem2 = DatabaseConversationMemory(user_id="u1", db_url=url, max_messages=2)
    msgs = mem2.get_recent_messages()
    assert [m.content for m in msgs] == ["hi"]
