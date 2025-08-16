import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.confidence_action import ConfidenceBasedActionSelector
from neurosales.user_preferences import PreferenceProfile


def test_certainty_and_styles():
    selector = ConfidenceBasedActionSelector()
    profile = PreferenceProfile(embedding=[0.1] * 384)
    style_hi, cert_hi = selector.select_style("u1", "hi", "hello", profile)
    assert style_hi in {"fire", "velvet", "intel"}
    assert 0.0 <= cert_hi <= 1.0
    selector.record_feedback("hello", success=False)
    style_low, cert_low = selector.select_style("u1", "bye", "hello", profile)
    assert cert_low <= cert_hi
    if cert_low < selector.base_soften:
        wrapped = selector.disclaimer("text", cert_low)
        assert "help me out" in wrapped

