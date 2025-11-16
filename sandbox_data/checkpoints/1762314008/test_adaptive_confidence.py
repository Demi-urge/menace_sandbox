import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.adaptive_confidence import AdaptiveConfidenceScorer


def test_overall_confidence_average():
    scorer = AdaptiveConfidenceScorer()
    scores = {"intent": 0.8, "sentiment": 0.6, "entity": 0.7}
    conf = scorer.overall_confidence(scores)
    assert abs(conf - 0.7) < 1e-6


def test_threshold_adjustment_and_clarify():
    scorer = AdaptiveConfidenceScorer(base_threshold=0.6)
    scores = {"intent": 0.5, "sentiment": 0.4, "entity": 0.6}
    assert scorer.should_clarify("u1", scores)
    scorer.update_with_feedback("u1", corrected=True, latency_increase=0.5)
    higher = scorer.get_threshold("u1")
    assert higher > 0.6
    scorer.update_with_feedback("u1", corrected=False, response_length_drop=0.5)
    assert scorer.get_threshold("u1") < higher


def test_modulate_voice_and_topic_decay():
    scorer = AdaptiveConfidenceScorer()
    hi = {"intent": 0.9, "sentiment": 0.9, "entity": 0.9}
    mid = {"intent": 0.6, "sentiment": 0.5, "entity": 0.5}
    low = {"intent": 0.2, "sentiment": 0.3, "entity": 0.4}
    assert scorer.modulate_voice(hi, user_id="u") == "cta"
    assert scorer.modulate_voice(mid, user_id="u") == "hedge"
    assert scorer.modulate_voice(low, user_id="u") == "clarify"

    base = scorer.overall_confidence(mid, topic="sales")
    scorer.record_topic_result("sales", success=True)
    boosted = scorer.overall_confidence(mid, topic="sales")
    assert boosted > base
    scorer.topic_last["sales"] -= 7200
    scorer.decay_topics()
    decayed = scorer.overall_confidence(mid, topic="sales")
    assert decayed < boosted


def test_stamp_response_labels():
    scorer = AdaptiveConfidenceScorer(base_threshold=0.5)
    hi = {"semantic": 0.9, "emotional": 0.9, "history": 0.9}
    low = {"semantic": 0.2, "emotional": 0.2, "history": 0.2}
    conf, style = scorer.stamp_response("u1", hi)
    assert style == "cta"
    assert conf > 0.8
    _, style_low = scorer.stamp_response("u1", low)
    assert style_low == "clarify"
