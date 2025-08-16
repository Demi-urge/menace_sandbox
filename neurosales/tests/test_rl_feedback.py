import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.self_learning import SelfLearningEngine
from neurosales.reactions import ReactionHistory
from neurosales.rl_integration import DatabaseReplayBuffer
from neurosales.sql_db import create_session as create_sql_session, RLFeedback


def test_feedback_persistence_and_loading():
    Session = create_sql_session("sqlite://")
    eng = SelfLearningEngine(session_factory=Session)
    hist = ReactionHistory(ttl_seconds=None, session_factory=Session)
    eng.log_interaction("hi", "issue")
    hist.add_pair("hello", "smile")

    buf = DatabaseReplayBuffer("u1", session_factory=Session)
    with Session() as s:
        rows = s.query(RLFeedback).all()
    assert len(rows) == 2
    assert all(not r.processed for r in rows)
    assert len(buf) >= 2
