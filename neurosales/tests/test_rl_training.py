import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

from neurosales.self_learning import SelfLearningEngine
from neurosales.rl_integration import DatabaseRLResponseRanker
from neurosales.sql_db import create_session as create_sql_session


def test_ranker_learns_from_logged_feedback():
    data_path = Path(__file__).parent / "data" / "rl_feedback.json"
    records = json.loads(data_path.read_text())

    Session = create_sql_session("sqlite://")
    engine = SelfLearningEngine(session_factory=Session)
    for rec in records:
        engine.log_interaction(rec["text"], rec["feedback"])

    ranker = DatabaseRLResponseRanker(session_factory=Session)
    scores = {"resp_a": 0.0, "resp_b": 0.0}
    ranker.log_outcome("u1", (0,), "resp_b", 0.0, (1,), list(scores))
    q_val = ranker._module("u1").predict((len(records[0]["text"]),), "resp_a")
    assert q_val > 0.0
