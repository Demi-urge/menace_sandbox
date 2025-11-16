import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.behavioral_shift import BehavioralShiftDetector


def test_training_and_detection():
    det = BehavioralShiftDetector()
    baseline = ["hello there", "how are you", "nice weather"]
    det.fit(baseline)
    score = det.detect("u1", "FUCK OFF", timestamp=3)
    assert score > 0


def test_session_score():
    det = BehavioralShiftDetector()
    det.fit(["hi"] * 3)
    score = det.session_anomaly_score(["hi", "HI", "sir please", "damn"])
    assert score >= 0
