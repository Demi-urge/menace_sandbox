import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.self_learning import SelfLearningEngine
from neurosales.rl_integration import DatabaseRLResponseRanker
from neurosales.sql_db import create_session as create_sql_session, RLFeedback


def test_export_corrections_updates_ranker():
    Session = create_sql_session("sqlite://")
    eng = SelfLearningEngine(session_factory=Session)
    ranker = DatabaseRLResponseRanker(session_factory=Session)

    eng.log_interaction("hello", "resp_a")
    count = eng.export_corrections(ranker)
    assert count == 1

    q_val = ranker._module("trainer").predict((len("hello"),), "resp_a")
    assert q_val > 0.0

    with Session() as s:
        row = s.query(RLFeedback).first()
        assert row.processed
