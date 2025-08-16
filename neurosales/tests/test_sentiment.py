import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.sentiment import SentimentAnalyzer, SentimentMemory


def test_analyser_basic():
    analyser = SentimentAnalyzer()
    score, emotions = analyser.analyse("I am very happy!")
    assert score > 0
    assert "joy" in emotions


def test_memory_logging_and_average():
    mem = SentimentMemory()
    mem.log("u1", 0.8, ["joy"], timestamp=0)
    mem.log("u1", -0.2, ["sadness"], timestamp=1)
    avg = mem.average_user_sentiment("u1")
    assert abs(avg - 0.3) < 1e-6
