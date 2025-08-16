import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.emotion import EmotionResult, DatabaseEmotionMemory
from neurosales.sql_db import create_session as create_sql_session


def test_persistent_emotions():
    Session = create_sql_session("sqlite://")
    mem = DatabaseEmotionMemory(session_factory=Session)
    res = EmotionResult(timestamp=1.0, primary="joy", secondary="joy", intensity=0.8, confidence=1.0)
    mem.log("p1", res)

    mem2 = DatabaseEmotionMemory(session_factory=Session)
    avg = mem2.average_intensity("p1")
    assert abs(avg - 0.8) < 1e-6


def test_emotion_db_url(tmp_path):
    db_file = tmp_path / "emo.db"
    url = f"sqlite:///{db_file}"
    mem = DatabaseEmotionMemory(db_url=url)
    res = EmotionResult(timestamp=2.0, primary="sadness", secondary="sadness", intensity=-0.5, confidence=1.0)
    mem.log("p2", res)

    mem2 = DatabaseEmotionMemory(db_url=url)
    avg = mem2.average_intensity("p2")
    assert abs(avg + 0.5) < 1e-6
