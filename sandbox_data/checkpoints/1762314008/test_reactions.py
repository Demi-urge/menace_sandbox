import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.reactions import ReactionHistory


def test_add_and_order():
    hist = ReactionHistory(ttl_seconds=100)
    hist.add_pair("hello", "smile")
    hist.add_pair("bye", "wave")
    pairs = hist.get_pairs()
    assert pairs == [("hello", "smile"), ("bye", "wave")]


def test_prune_old_pairs():
    hist = ReactionHistory(ttl_seconds=5)
    now = time.time()
    hist.add_pair("old", "thumbs", timestamp=now - 10)
    hist.add_pair("new", "nod", timestamp=now)
    hist.prune()
    pairs = hist.get_pairs()
    assert pairs == [("new", "nod")]
