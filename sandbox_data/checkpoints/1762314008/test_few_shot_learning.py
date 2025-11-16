import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.few_shot_learning import FewShotZeroShotClassifier


def test_basic_classification():
    clf = FewShotZeroShotClassifier(threshold=0.2)
    clf.add_examples("buy", ["i want to buy", "purchase item"])
    clf.add_examples("greet", ["hello there", "hi friend"])
    label, score = clf.classify("i would like to buy shoes")
    assert label == "buy"
    assert score > 0


def test_zero_shot_and_feedback():
    clf = FewShotZeroShotClassifier(threshold=0.4, micro_epoch=1)
    clf.add_examples("greet", ["hi there"])
    label, score = clf.classify("i want to purchase things")
    assert label.startswith("category_")
    clf.log_feedback("i want to purchase things", label, "buy")
    label2, _ = clf.classify("i want to purchase things")
    assert label2 == "buy"
