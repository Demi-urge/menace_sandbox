import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.rlhf import DatabaseRLHFPolicyManager
from neurosales.rl_integration import DatabaseRLResponseRanker
from neurosales.sql_db import create_session as create_sql_session


def test_persistent_policy_weights():
    Session = create_sql_session("sqlite://")
    mgr1 = DatabaseRLHFPolicyManager(session_factory=Session, exploration_rate=0.0)
    mgr1.record_result("hi", ctr=0.5, sentiment=0.5, session=0.5)

    mgr2 = DatabaseRLHFPolicyManager(session_factory=Session, exploration_rate=0.0)
    assert mgr2.weights.get("hi") is not None
    assert mgr2.counts["hi"] == 1


def test_persistent_replay_experience():
    Session = create_sql_session("sqlite://")
    ranker1 = DatabaseRLResponseRanker(session_factory=Session)
    ranker1.log_outcome("u1", (0,), "a", 1.0, (1,), ["a"])

    ranker2 = DatabaseRLResponseRanker(session_factory=Session)
    buf = ranker2._buffer("u1")
    assert len(buf) == 1
    exp = buf.sample(1)[0]
    assert exp.action == "a"
