import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.intent_detection import IntentDetector


def test_basic_intent_detection():
    messages = [
        "I want to buy shoes",
        "Where is my order",
        "Love the new product line",
        "How much does it cost",
        "Any coupons available",
    ]
    detector = IntentDetector(num_clusters=2, threshold=0.3)
    detector.fit(messages)
    res = detector.detect_intents("I want a refund for my order")
    assert res
    labels = [label for label, _ in res]
    assert "clarify" in labels or len(labels) > 0


def test_incremental_fit():
    msgs = ["buy shoes", "need help"]
    detector = IntentDetector(num_clusters=2)
    detector.fit(msgs)
    detector.partial_fit(["discount codes"])
    assert detector.messages and len(detector.messages) == 3
