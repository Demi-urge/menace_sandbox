import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.intent_classifier import IntentClassifier


def test_basic_classification():
    data = [
        (["hi", "i want to buy shoes"], ["buy"]),
        (["hello", "need a refund"], ["support"]),
        (["any discounts"], ["discount"]),
    ]
    clf = IntentClassifier(context_size=2, threshold=0.3)
    clf.fit(data)
    res = clf.classify(["i really need", "a refund"])
    assert res
    labels = [l for l, _ in res]
    assert "support" in labels or "clarify" in labels


def test_threshold_adjustment():
    data = [(["ping"], ["misc"])]
    clf = IntentClassifier()
    clf.fit(data)
    old = clf.threshold
    clf.adjust_threshold(0.4)
    assert clf.threshold < old

