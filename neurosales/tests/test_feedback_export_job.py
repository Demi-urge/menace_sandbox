import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.self_learning import SelfLearningEngine
from neurosales.sql_db import create_session as create_sql_session, RLFeedback
from neurosales.rl_training import schedule_feedback_export


def test_schedule_feedback_export(tmp_path):
    dataset = tmp_path / "fb.json"
    Session = create_sql_session("sqlite://")
    eng = SelfLearningEngine(session_factory=Session)
    eng.log_interaction("hi", "resp")

    schedule_feedback_export(
        interval=0.1, dataset_path=str(dataset), session_factory=Session
    )
    time.sleep(0.2)

    assert dataset.exists()
    with Session() as s:
        rows = s.query(RLFeedback).all()
        assert all(r.processed for r in rows)
