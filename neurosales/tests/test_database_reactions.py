import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.reactions import DatabaseReactionHistory
from neurosales.sql_db import create_session as create_sql_session


def test_persistent_pairs():
    Session = create_sql_session("sqlite://")
    hist = DatabaseReactionHistory(
        user_id="u1", session_factory=Session, ttl_seconds=100
    )
    hist.add_pair("hi", "wave")

    hist2 = DatabaseReactionHistory(
        user_id="u1", session_factory=Session, ttl_seconds=100
    )
    pairs = hist2.get_pairs()
    assert pairs == [("hi", "wave")]


def test_history_db_url(tmp_path):
    db_file = tmp_path / "react.db"
    url = f"sqlite:///{db_file}"
    hist = DatabaseReactionHistory(user_id="u1", db_url=url, ttl_seconds=100)
    hist.add_pair("hello", "smile")

    hist2 = DatabaseReactionHistory(user_id="u1", db_url=url, ttl_seconds=100)
    pairs = hist2.get_pairs()
    assert pairs == [("hello", "smile")]
