import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.ner import IntentEntityExtractor
import time


def test_basic_extraction():
    extractor = IntentEntityExtractor(ttl_seconds=10)
    triggers = extractor.extract("I want to buy things from reddit")
    assert "buy" in triggers
    assert "reddit" in triggers
    profile = extractor.get_intent_profile()
    assert profile.get("buy") == 1


def test_alias_standardisation():
    extractor = IntentEntityExtractor(ttl_seconds=10)
    extractor.extract("found on ig")
    profile = extractor.get_intent_profile()
    assert "instagram" in profile


def test_prune_unrepeated():
    extractor = IntentEntityExtractor(ttl_seconds=0.1)
    extractor.extract("need this")
    time.sleep(0.2)
    extractor.extract("nothing here")
    profile = extractor.get_intent_profile()
    assert "need" not in profile


def test_repeated_survives():
    extractor = IntentEntityExtractor(ttl_seconds=1)
    extractor.extract("buy now")
    extractor.extract("I will buy again")
    profile = extractor.get_intent_profile()
    assert profile.get("buy", 0) >= 2
