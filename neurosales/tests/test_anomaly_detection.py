import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.anomaly_detection import AnomalyDetector


def test_anomaly_detects_change():
    det = AnomalyDetector(short_window=3)
    for i in range(3):
        det.detect("u1", "hello", timestamp=i)
    score = det.detect("u1", "WHAT IS THIS?!", timestamp=3)
    assert score > 1


def test_anomaly_low_for_consistent_messages():
    det = AnomalyDetector()
    for i in range(5):
        score = det.detect("u2", "hi there", timestamp=i)
    assert score < 1
