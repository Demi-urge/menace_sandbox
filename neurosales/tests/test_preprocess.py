import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.preprocess import TextPreprocessor


def test_basic_preprocess():
    pre = TextPreprocessor(["buy"])
    text = "I'm buying now! u won't regret 😊 email me at t@example.com visit https://a.com"
    result = pre.preprocess(text)
    assert "you" in result.tokens  # slang converted
    assert "buy" in result.lemmas  # lemmatization
    assert result.trigger_flags.get("buy", 0) >= 1
    assert "https://a.com" in result.urls
    assert "t@example.com" in result.emails


def test_subword_trigger_detection():
    pre = TextPreprocessor(["buy"])
    result = pre.preprocess("buyer buys buying")
    assert result.trigger_flags.get("buy", 0) == 3
