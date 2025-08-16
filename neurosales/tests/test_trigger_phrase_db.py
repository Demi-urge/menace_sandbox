import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.trigger_phrase_db import TriggerPhraseDB


def test_add_and_best_phrase():
    db = TriggerPhraseDB()
    db.add_phrase("only 3 left", ["amygdala"])
    db.record_feedback("only 3 left", context="email", clicks=2, conversions=1)
    db.add_phrase("limited offer", ["amygdala"])
    db.record_feedback("limited offer", context="email", clicks=1)
    best = db.best_phrase(["only 3 left", "limited offer"], context="email")
    assert best == "only 3 left"


def test_prune_and_mutate():
    db = TriggerPhraseDB()
    db.add_phrase("weak phrase", ["cortex"])
    db.record_feedback("weak phrase", context="sms", clicks=0)
    db.prune(threshold=1)
    assert "weak phrase" not in db.phrases
    db.add_phrase("strong", ["amygdala"])
    db.record_feedback("strong", context="sms", conversions=2)
    muts = db.mutate(top_n=1)
    assert muts == ["strong!"]
