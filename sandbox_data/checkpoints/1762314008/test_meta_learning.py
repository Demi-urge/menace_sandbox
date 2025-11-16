import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.meta_learning_preprocess import MetaLearningPreprocessor


def test_data_utility_score():
    meta = MetaLearningPreprocessor(base_threshold=0.0)
    s1 = meta.data_utility_score("hello hello")
    s2 = meta.data_utility_score("hello world")
    assert s2 > s1


def test_feedback_threshold_adjustment():
    meta = MetaLearningPreprocessor(base_threshold=0.1)
    assert meta.add_sample("x", "good sample")
    thr = meta.category_thresholds["x"]
    meta.update_feedback("x", confirmed=True)
    assert meta.category_thresholds["x"] > thr
    thr2 = meta.category_thresholds["x"]
    meta.update_feedback("x", confirmed=False)
    assert meta.category_thresholds["x"] < thr2


def test_self_evaluate_logs_refinements():
    meta = MetaLearningPreprocessor()
    meta.add_sample("greet", "hello there")
    # force misclassification by clearing classifier centroids
    meta.classifier.centroids = {}
    meta.self_evaluate()
    assert meta.refinement_log
